import torch
from collections import OrderedDict
import os
from os.path import join as jn
import logging
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from numpy.typing import NDArray
from typing import Optional
from functions.graph_construction import OnDiskStencilGraph, VariableStencilGraph, CustomLoader
from functions.stencil_growth import StencilGrowth
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from functions.Plots import plot_training_pytorch
from functions.gnn_preproc import load
from functions.labfm_moments import calc_moments_torch, monomial_power
from models.NEMDO_mod import NEMDO
from scipy.special import factorial
from torch_geometric.nn.aggr import SumAggregation
import torch._dynamo
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    # change these to increase performance
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('highest') # options are highest, high and medium
    torch._dynamo.config.capture_scalar_outputs = True # required because of attention aggregation method in gnn

def construct_data_loader(cpu_cores: int,
                          batch_size: int,
                          train_idx: NDArray,
                           val_idx: NDArray,
                           test_idx: NDArray,
                          distances: NDArray,
                          prefetch_factor: int,
                          root: Optional[str] = '',
                          variable_stencil: bool = False,
                          initial_stencil: int = 5):

    test_root = os.path.join(root, 'test_graphs')
    val_root  = os.path.join(root, 'val_graphs')
    train_root = os.path.join(root, 'train_graphs')

    pin_memory = True

    logger.info('Creating graphs')
    if variable_stencil:
        # every point gets its own stencil size, starting at initial_stencil (grown during training)
        test_ds = VariableStencilGraph(features=distances[test_idx], initial_size=initial_stencil)
        val_ds = VariableStencilGraph(features=distances[val_idx], initial_size=initial_stencil)
        train_ds = VariableStencilGraph(features=distances[train_idx], initial_size=initial_stencil)
    else:
        test_ds = OnDiskStencilGraph(features=distances[test_idx],
                               root=test_root)

        val_ds = OnDiskStencilGraph(features=distances[val_idx],
                               root=val_root)

        train_ds = OnDiskStencilGraph(features=distances[train_idx],
                               root=train_root)

    logger.info('Creating data loader')
    test_loader = CustomLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=cpu_cores,
                             pin_memory=pin_memory,
                             drop_last=False,
                             prefetch_factor=prefetch_factor,
                             in_order=True)

    val_loader = CustomLoader(val_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=cpu_cores,
                             pin_memory=pin_memory,
                             drop_last=True,
                             prefetch_factor=prefetch_factor,
                             in_order=True,
                             persistent_workers=True)

    train_loader = CustomLoader(train_ds,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=cpu_cores,
                             pin_memory=pin_memory,
                             drop_last=True,
                             prefetch_factor=prefetch_factor,
                             in_order=True,
                             persistent_workers=True)


    return test_loader, val_loader, train_loader


def train_model(model_id: int,
                epochs: int,
                input_size: int,
                embedding_size: int,
                layers: list | int,
                lr: float,
                out_path: str,
                train_loader: DataLoader,
                val_loader: DataLoader,
                batch_size: int,
                derivative: str,
                checkpoint_p_epoch: int,
                checkpoint_path: str,
                approximation_order: int,
                resume_training: str,
                growth: Optional[StencilGrowth] = None):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    print(f"PID: {os.getpid()} started.")

    #torch.manual_seed(1222)

    # if model is resuming training
    if resume_training:
        logger.info(f'Resuming training for model {resume_training}')

        attrs = torch.load(resume_training,
                           map_location='cpu',
                           weights_only=False)

        layers = attrs['layers']
        embedding_size = attrs['embedding_size']
        lr = attrs['lr']

        model = NEMDO(input_size=input_size,
                         output_size=1,
                         embedding_size=embedding_size,
                        layers=layers).to(device)


        weight_dict = OrderedDict()

        weight_dict.update(
            (k[len("module."):], v) if k.startswith("module.")
            else (k, v) for k, v in attrs['weights'].items())

        model.load_state_dict(weight_dict)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        optimizer.load_state_dict(attrs['optimizer'])


        train_history = attrs['train_history']
        val_history   = attrs['val_history']
        resume_epoch  = attrs['epochs']
        best_val_loss = attrs['best_val_loss']
        model_id      = attrs['model_id']

        if growth is not None and 'stencil_sizes' in attrs:
            train_loader.data.stencil_sizes[:] = attrs['stencil_sizes']['train']
            val_loader.data.stencil_sizes[:]   = attrs['stencil_sizes']['val']

    else:
        model = NEMDO(input_size=input_size,
                         output_size=1,
                         embedding_size=embedding_size,
                        layers=layers).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_history = []
        val_history = []
        best_val_loss = torch.inf
        resume_epoch = 0
        linear_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)

    # Scheduler
    plateau_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                          patience=13,
                                          factor=0.4,
                                          cooldown=15,
                                          eps=1e-12)


    n = int((approximation_order ** 2 + 3 * approximation_order) / 2)
    target_moments = torch.zeros((n, 1), dtype=torch.float32)
    if derivative == 'laplace':
        target_moments[2] = 1.0
        target_moments[4] = 1.0
    elif derivative == 'x':
        target_moments[0] = 1.0
    elif derivative == 'y':
        target_moments[1] = 1.0
    elif derivative == 'hyp':
        if approximation_order != 4: raise ValueError('For hyperviscosity, operator must be 4th order')
        target_moments[9]  = -1.0
        target_moments[11] = -2.0
        target_moments[13] = -1.0
    else:
        raise ValueError("derivative must be either 'laplace', 'x', or 'y'")

    # Pre-computing data that will be used to compute the moments
    target_moments = target_moments.expand(-1, batch_size).to(device=device)

    mon_power = monomial_power(approximation_order)
    inv_factorial = 1 / (factorial(mon_power[:, 0]) * factorial(mon_power[:, 1]))
    inv_factorial = torch.tensor(inv_factorial, dtype=torch.float32, device=device)
    mon_power = torch.tensor(mon_power, dtype=torch.float32, device=device).T
    sum_aggr = SumAggregation().to(device=device)

    # stencil sizes (and so graph sizes) change between batches and epochs when they are grown
    model.compile(dynamic=growth is not None)

    if growth is not None:
        train_sizes = train_loader.data.stencil_sizes
        val_sizes   = val_loader.data.stencil_sizes
        # most recent loss of every point since its stencil last changed (NaN = not measured yet)
        train_pt_loss = torch.full((len(train_sizes),), float('nan'), device=device)
        val_pt_loss   = torch.full((len(val_sizes),), float('nan'), device=device)

    # get list of CPUs available to attach processes, can also be manually picked
    allowed = sorted(os.sched_getaffinity(0))
    workers = allowed[:cpu_cores]
    logger.info('Entering training loop')

    loss_scaling = 1

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        model.train()

        total_loss = torch.tensor(0.0, device=device)

        with train_loader.enable_cpu_affinity(loader_cores=workers):

            for num_batches, batch in enumerate(train_loader):

                batch = batch.to(device, non_blocking=True)

                optimizer.zero_grad()

                out = model(batch.x,
                            batch.edge_index,
                            batch.batch)

                pred_m = calc_moments_torch(batch.x,
                                            out,
                                            batch.batch,
                                            mon_power,
                                            inv_factorial,
                                            sum_aggr)

                loss = F.mse_loss(target_moments, pred_m)

                if growth is not None:
                    train_pt_loss[batch.point_idx] = ((target_moments - pred_m) ** 2).mean(dim=0).detach()

                loss = loss #* loss_scaling

                loss.backward()

                optimizer.step()

                total_loss += loss.detach()

        train_loss = total_loss / (num_batches + 1)


        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        num_batches = 0
        with torch.no_grad():
            with val_loader.enable_cpu_affinity(loader_cores=workers):
                for batch in val_loader:
                    num_batches += 1
                    batch = batch.to(device, non_blocking=True)
                    out = model(batch.x,
                                batch.edge_index,
                                batch.batch)

                    pred_m = calc_moments_torch(batch.x,
                                                out,
                                                batch.batch,
                                                mon_power,
                                                inv_factorial,
                                                sum_aggr)

                    val_loss = F.mse_loss(target_moments, pred_m)

                    if growth is not None:
                        val_pt_loss[batch.point_idx] = ((target_moments - pred_m) ** 2).mean(dim=0)

                    total_loss += val_loss#.detach()

                val_loss = total_loss / num_batches

                if val_loss < best_val_loss:
                    e = epoch + resume_epoch
                    check_epoch = e
                    best_val_loss = val_loss
                    save_weights = model.state_dict()
                    save_optimizer = optimizer.state_dict()

        train_loss /= loss_scaling
        train_history.append(float(train_loss))
        val_history.append(float(val_loss))

        elapsed = time.perf_counter() - t0
        e = epoch + resume_epoch
        logger.info(f'Epoch {e:3d} — Train Loss: {train_loss:.5e} || Val Loss: {val_loss:.5e} || '
              f'time per epoch: {elapsed:.3f}s')

        # The scheduler step is purposely taken with the training loss
        plateau_scheduler.step(train_loss)

        if not resume_training: linear_scheduler.step()

        if growth is not None and growth.stalled(float(train_loss)):
            # loss has stalled: give the worst points (per dataset) a bigger stencil
            grew_train = growth.grow(train_sizes, train_pt_loss.cpu())
            grew_val   = growth.grow(val_sizes, val_pt_loss.cpu())
            train_pt_loss[grew_train.to(device)] = float('nan')
            val_pt_loss[grew_val.to(device)] = float('nan')
            growth.reset()
            # validation losses on different stencils are not comparable, so best-model tracking restarts
            best_val_loss = torch.inf
            logger.info(f'Loss stalled: grew {int(grew_train.sum())}/{len(train_sizes)} train stencils || '
                        f'mean stencil size train {train_sizes.float().mean():.2f} '
                        f'(min {int(train_sizes.min())}, max {int(train_sizes.max())}) || '
                        f'val {val_sizes.float().mean():.2f}')

        if epoch % checkpoint_p_epoch == 0:
            save_dict = {'train_history' : train_history,
                         'val_history'   : val_history,
                         'best_val_loss' : best_val_loss,
                         'weights'       : save_weights,
                         'optimizer'     : save_optimizer,
                         'epochs'        : e,
                         'batch_size'    : batch_size,
                         'layers'        : layers,
                         'input_size'    : input_size,
                         'lr'            : lr,
                         'embedding_size': embedding_size,
                         'approximation_order': approximation_order,
                         'model_id'      : model_id,
                         'loss_scaling'  : loss_scaling}
            if growth is not None:
                save_dict['stencil_sizes'] = {'train': train_sizes.clone(), 'val': val_sizes.clone()}
            e = epoch + resume_epoch
            save_path = jn(checkpoint_path, f'attrs{model_id}_epoch{check_epoch}.pth')
            torch.save(save_dict, save_path)
            logger.info(f'Checkpoint model saved at {save_path} in epoch {e} from epoch {check_epoch}')


    save_dict = {'train_history' : train_history,
                 'val_history'   : val_history,
                 'best_val_loss' : best_val_loss,
                 'weights'       : save_weights,
                 'optimizer'     : save_optimizer,
                 'epochs'        : e,
                 'batch_size'    : batch_size,
                 'layers'        : layers,
                 'input_size'    : input_size,
                 'lr'            : lr,
                 'embedding_size': embedding_size,
                 'approximation_order': approximation_order,
                 'model_id'      : model_id,
                 'loss_scaling'  : loss_scaling}
    if growth is not None:
        save_dict['stencil_sizes'] = {'train': train_sizes.clone(), 'val': val_sizes.clone()}

    save_path = jn(out_path, f'attrs{model_id}.pth')


    torch.save(save_dict, save_path)
    logger.info(f'Saved model at {save_path}')


if __name__=='__main__':
    # to isolate the host and the cores used for dataloader run the code with
    # numactl -C 4-7 --localalloc python3 main_train.py
    cpu_cores   = 4                                        # number of cpu cores to load data for gpu
    batch_size  = 128                                      #
    prefetch_factor = 10                                    # number of batches for cpu to prefetch
    model_id    = 67                                      # id of the model to save
    epochs      = 1000                                      # total of number of epochs to run
    lr          = 1e-3                                   # initial learning rate
    input_size  = 2                                        # 2 dimensional input (x and y)
    layers      = 2                                        # num of gnn layers
    embedding_size = 64                                    # embedding size
    data_iteration = 4                                     # which data iteration to use
    checkpoint_p_epoch = 250                                # every how many epochs to save checkpoint
    approximation_order = 2                                # order of approximation for loss moments
    continue_train_model = ''                              # set to checked model full path to resume training, leave empty string for new model
    derivative         = 'x'                               # the differential operator the gnn will learn ('x', 'y', 'laplace', or 'hyp')
    base_model_path    = 'saved_models'                    # root dir to save models and checkpoints
    out_path           = jn(base_model_path, derivative)   # dir to save best model trained
    checkpoint_path    = jn(base_model_path, 'checkpoint') # dir to save checkpoint model
    root_dir_graphs    = 'graphs'                          # root dir for graphs to be saved
    base_path          = 'preproc_data'                    # root dir to get imported preproc data

    variable_stencil = True                                # each point gets its own stencil size, grown when the loss stalls
    initial_stencil    = 5                                 # starting stencil size of every point
    stencil_step       = 2                                 # neighbours added to a point's stencil each time it grows
    growth_patience    = 10                                # epochs without improvement before the stencils grow
    growth_rel_tol     = 1e-2                              # relative train loss improvement that counts as progress
    growth_cooldown    = 5                                 # epochs to wait after growing before checking for a stall
    growth_quantile    = 0.5                               # only points with loss above this quantile grow (0 = all)
    converged_tol      = 0.0                               # points with per-point loss below this stop growing

    train = True                                         # set train=False and plot=True to only visualise training loss
    plot  = True                                          # plots training-validation curve when finished training

    f_path = jn(base_path, f'iter{data_iteration}')
    root_dir_graphs = jn(root_dir_graphs, f'{data_iteration}')

    if train:
        distances = load(os.path.join(f_path, 'distances.pk'))
        train_idx = load(os.path.join(f_path, 'train_idx.pk'))
        val_idx = load(os.path.join(f_path, 'val_idx.pk'))
        test_idx = load(os.path.join(f_path, 'test_idx.pk'))

        (test_loader,
         val_loader,
         train_loader) = construct_data_loader(cpu_cores=cpu_cores,
                                               batch_size=batch_size,
                                               train_idx=train_idx,
                                               val_idx=val_idx,
                                               test_idx=test_idx,
                                               distances=distances,
                                               prefetch_factor=prefetch_factor,
                                               root=root_dir_graphs,
                                               variable_stencil=variable_stencil,
                                               initial_stencil=initial_stencil)

        growth = None
        if variable_stencil:
            growth = StencilGrowth(max_size=distances.shape[1],
                                   step=stencil_step,
                                   patience=growth_patience,
                                   rel_tol=growth_rel_tol,
                                   cooldown=growth_cooldown,
                                   quantile=growth_quantile,
                                   converged_tol=converged_tol)

        os.makedirs(out_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        train_model(model_id,
                    epochs,
                    input_size,
                    embedding_size,
                    layers,
                    lr,
                    out_path,
                    train_loader,
                    val_loader,
                    batch_size,
                    derivative,
                    checkpoint_p_epoch,
                    checkpoint_path,
                    approximation_order,
                    continue_train_model,
                    growth)


    if plot:
        path = jn(out_path, f'attrs{model_id}.pth') if not continue_train_model else continue_train_model
        attrs = torch.load(path,
                           map_location='cpu',
                           weights_only=False)
        h = {'history': (attrs['train_history'], attrs['val_history'])}
        plot_training_pytorch(h, log_x=True, log_y=True)
        print(
            f"Model Summary:\n"
            f"best_val_loss: {attrs['best_val_loss']}\n"
            f"Max epoch: {attrs['epochs']}\n"
            f"Batch size: {attrs['batch_size']}\n"
            f"Layers: {attrs['layers']}\n"
            f"Embedding size: {attrs['embedding_size']}\n"
            f"Approx order: {attrs['approximation_order']}\n"
            f"Model ID: {attrs['model_id']}"
        )


from functions.gnn_preproc import save, split_data_by_index
from functions.Plots import *
import os
import logging
from typing import Optional


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_parallel(data_path: str,
                    data_iteration: int | str,
                    n_cores: int,
                    save_path: str,
                    max_neighbours: Optional[int] = None,
                    plot_stencils: Optional[bool] = False,
                    variable_stencil: Optional[bool] = False) -> None:

    from functions.parallel_load import load_and_stack_ij_links

    distances = load_and_stack_ij_links(data_path,
                                        data_iteration=data_iteration,
                                        n_cores=n_cores)

    if max_neighbours:
        max_neighbours = min(max_neighbours, distances.shape[1])
        distances   = distances[:, :max_neighbours, :]
        # with a variable stencil the normalisation depends on each point's current stencil size, so it is done on
        # the fly by VariableStencilGraph and the raw offsets are stored here
        if not variable_stencil:
            r_distances = (distances[..., 0] ** 2 + distances[..., 1] ** 2) ** .5
            max_r       = np.max(r_distances, axis=1)
            distances   = distances / (max_r[..., None, None])

    if plot_stencils:
        plot_kernel(distances)


    train_size = int(distances.shape[0] * 0.7)
    val_size  = int(distances.shape[0] * 0.2)
    test_size = int(distances.shape[0] * 0.1)

    (train_idx,
     val_idx,
     test_idx) = split_data_by_index(0, distances.shape[0], (train_size, val_size, test_size), seed=42)

    print('Training dataset size: ', train_idx.shape[0])
    print('Validation dataset size: ', val_idx.shape[0])
    print('Test dataset size: ', test_idx.shape[0])
    print('Total dataset size: ', train_idx.shape[0] + val_idx.shape[0] + test_idx.shape[0])
    print('Number of neighbours: ', max_neighbours)


    train_idx_dir = os.path.join(save_path, 'train_idx.pk')
    val_idx_dir = os.path.join(save_path, 'val_idx.pk')
    test_idx_dir = os.path.join(save_path, 'test_idx.pk')
    distances_dir = os.path.join(save_path, 'distances.pk')

    # path, obj
    save(train_idx_dir, train_idx,
         val_idx_dir, val_idx,
         test_idx_dir, test_idx,
         distances_dir, distances)



if __name__ == '__main__':
    # This routine doesn't use h, it normalises the distance and wrt the maximum distance of the neighbours
    data_path         = './dataset'                         # directory to read data
    data_iteration    = 4                                   # which file iteration to read
    n_cores           = 4                                   # number of threads to read data
    root              = 'preproc_data'                      # directory to save preprocessed data
    max_neighbours    = 20                                  # stencil size (maximum stencil size if variable_stencil)
    plot_stencil      = False                               # plots stencil
    variable_stencil  = True                                # keep raw offsets so training can grow each point's stencil

    # directories where each dataset will be saved
    test_root  = os.path.join(root, 'test_graphs')
    val_root   = os.path.join(root, 'val_graphs')
    train_root = os.path.join(root, 'train_graphs')


    save_path = os.path.join('./preproc_data_no_w', f'iter{data_iteration}')
    os.makedirs(save_path, exist_ok=True)
    import_parallel(data_path=data_path,
                    data_iteration=data_iteration,
                    save_path=save_path,
                    n_cores=n_cores,
                    max_neighbours=max_neighbours,
                    plot_stencils=plot_stencil,
                    variable_stencil=variable_stencil)


# training_bc — learning NeMDO operators on domains with boundaries

`training/` learns mesh-free operators from interior stencils taken from a perturbed
square grid: every centre node is fully surrounded by neighbours. A node that sits
**on a wall** is not like that. Its stencil is one-sided, the local Taylor system is
several times worse conditioned, and the weights the operator needs there are much
larger. A model that never saw such a stencil has to extrapolate, and the shipped
`nemdo_x` / `nemdo_lap` checkpoints should not be trusted on boundary rows.

This package trains on node clouds that have real boundaries, and adds the operator a
flux boundary condition needs.

Training stays **self-supervised** — the loss is still Taylor/polynomial consistency,
so no PDE solutions and no labelled weights are required. "Boundary conditions" here
means the geometry and the operators that let you *impose* a condition
(Dirichlet / Neumann / Robin), not boundary *values* in the training data. Values
enter at assembly time; see [Assembling rows](#assembling-rows).

---

## What is different from `training/`

| | `training/` | `training_bc/` |
|---|---|---|
| Node sets | perturbed square grid, interior only | square, disk, annulus, L-shape, star blobs — with wall node layers |
| Stencils | fully surrounded | one-sided at walls, including a re-entrant corner |
| Frame | global axes; rotational robustness learned from data diversity | canonical frame with the outward normal on `+x`, so orientation is removed by construction |
| Operators | one model per operator | one shared trunk, one head per operator (`d/dx`, `d/dy`, Laplacian) |
| Normal derivative | not available | the `d/dx` head *is* `d/dn` at a boundary node |
| Node features | `dx/h, dy/h` | `dx/h, dy/h, is_centre, boundary_proximity` |
| Loss | moment MSE | moment MSE, reweighted toward walls, plus a weight-norm penalty |

Three of these are worth expanding on.

**Canonical frame.** Each stencil is rotated so the outward normal of the nearest
boundary node points along local `+x`. Orientation stops being something the network
must learn from data diversity (the original NeMDO is not rotation-equivariant), and
the first-derivative head along the normal direction becomes exactly the operator a
Neumann row needs. Weights rotate back to global axes afterwards:

```
w_x_global = cos θ · w'_x − sin θ · w'_y
w_y_global = sin θ · w'_x + cos θ · w'_y
```

with the Laplacian unchanged, since it is rotation invariant.

**Boundary proximity.** A continuous channel, `1` on the wall decaying to `0` one
stencil radius inside, rather than a binary interior/boundary flag. The stencil of a
node two layers in is also partly one-sided, and the network does better knowing how
close it is than being told it is simply "interior".

**Weight-norm penalty.** On a one-sided stencil the consistency conditions no longer
pin the weights down tightly, and the network will happily drift to large oscillatory
solutions. Classical methods avoid this by taking the *minimum-norm* solve; a small
`λ‖w‖²` term is the learned equivalent.

---

## Quick start

```bash
cd training_bc
pip install -r requirements.txt

python main_build_dataset.py   # generate clouds, extract stencils  (~1 min)
python main_train_bc.py        # train d/dx, d/dy and the Laplacian (~45 min on CPU)
python main_eval_bc.py         # evaluate on a held-out shape
```

Defaults are CPU-friendly. For a production operator, lower `spacing` in
`main_build_dataset.py` (more stencils) and raise `embedding_size` / `epochs`.

---

## Files

```
training_bc/
├── main_build_dataset.py    # stage 1: node clouds -> stencil dataset
├── main_train_bc.py         # stage 2: multi-operator self-supervised training
├── main_eval_bc.py          # stage 3: held-out evaluation vs classical weights
├── test_sanity.py           # invariant checks (run this after any change)
├── functions/
│   ├── domains.py           # SDF-defined domains, node placement, outward normals
│   ├── boundary.py          # boundary detection, canonical frame, weight rotation
│   ├── stencils.py          # kNN stencils, features, fixed-K batching
│   ├── moments.py           # multi-operator Taylor targets and loss
│   ├── classical.py         # least-squares reference weights, conditioning
│   ├── infer.py             # trained model -> assembly-ready weights
│   ├── mesh_io.py           # read an external node set (your own mesh)
│   └── plots.py             # geometry check and error-field figures
└── models/
    └── nemdo_bc.py          # shared trunk, one head per operator
```

`python test_sanity.py` (seconds, no trained model needed) checks the things that
would otherwise fail *silently*: that classical weights hit their moments exactly,
that weights solved in the canonical frame and rotated back equal weights solved
directly in global axes, that the canonical frame really puts the normal on `+x`,
that the torch loss agrees with the numpy moment matrix, and that the `h` scaling is
the right power. A wrong rotation convention would still train to a plausible loss
while producing quietly wrong operators, so run this after touching any of it.

---

## Assembling rows

The operator form is unchanged from the rest of the repository,

```
L u(x_i) = Σ_j (u_j − u_i) w_ji
```

so the centre weight is always implied: `w_ii = −Σ_j w_ji`. With `w` from
`functions.infer.predict_weights`:

| Condition | Row |
|---|---|
| Interior | `Σ_j (u_j − u_i) w_lap_ji = f_i` |
| Dirichlet `u = g` | `u_i = g_i` (no stencil) |
| Neumann `∂u/∂n = q` | `Σ_j (u_j − u_i) w_n_ji = q_i` |
| Robin `a u + b ∂u/∂n = g` | `a u_i + b Σ_j (u_j − u_i) w_n_ji = g_i` |

`w_n` is `predict_weights(...)["normal"]`. Variable conductivity is **not** learned —
put `k` on the edges as a harmonic mean when assembling `∇·(k∇u)`, so one trained
operator serves every material.

```python
from functions.infer import predict_weights
from functions.mesh_io import load_csv_cloud
from functions.stencils import extract_stencils
from models.nemdo_bc import load_bc_model

model, _ = load_bc_model("saved_models_bc/nemdo_bc_1.pth")
cloud    = load_csv_cloud("my_mesh.csv")          # x,y[,is_boundary[,nx,ny]]
stencils = extract_stencils(cloud, stencil_size=35)
w        = predict_weights(model, stencils)

w_lap    = w["global"][:, :, model.operators.index("laplace")]
w_normal = w["normal"]
```

---

## Using your own mesh

`mesh_io.load_csv_cloud` reads `x, y[, is_boundary[, nx, ny]]`. Supply
`is_boundary` and the normals when the mesh knows them. The fallback estimates both
from neighbour asymmetry, which found every boundary node on a test shape but with a
median normal error around 6° and a 95th percentile near 20° — good enough to orient
a stencil, not good enough for a Neumann row at a sharp corner.

Plot the cloud before trusting anything from it. A flipped normal or a mislabelled
interior node is obvious in a picture and invisible in a loss curve:

```python
from functions.plots import plot_cloud
plot_cloud(load_csv_cloud("my_mesh.csv"), path="mesh_check.png")
```

There is no COMSOL `.mphtxt` reader here — writing one blind risks mis-parsing
silently, so export node coordinates (and the boundary flag, if you can) to CSV, or
send a sample export and the parser can be added.

To specialise the operator to one geometry, add the cloud to the mix in
`domains.default_domain_mix` and retrain. Training is self-supervised, so this needs
the mesh only, not a solution on it.

---

## What to expect, honestly

The loss is polynomial consistency, which the classical local solve satisfies
*exactly*. A learned operator trained on that objective therefore cannot beat it on
accuracy — the realistic goals are:

- **Usable boundary rows at all**, which the shipped checkpoints do not provide.
- **Bounded weights on badly conditioned stencils.** The exact solve answers a
  near-singular one-sided system with large oscillatory weights; the penalised
  network answers with smaller ones. That can matter more for solver stability than
  the last digit of consistency.
- **One forward pass instead of a local solve per node**, which only pays off when
  nodes move every step. On a fixed mesh you build the weights once and the
  advantage disappears.

`main_eval_bc.py` prints both side by side on a held-out shape, so the comparison is
visible rather than asserted.

### Measured on a reference run

Shipped defaults: 10k stencils, `K = 35`, 196k parameters, 300 epochs, ~50 min on
CPU. Held-out star blob not in the training mix, 1499 nodes, 120 on the boundary.
Relative L2 error on `u = sin(3x)cos(2y) + x³`:

| operator | NeMDO interior | NeMDO wall | classical interior | classical wall |
|---|---|---|---|---|
| `d/dx` | 1.4e-2 | 3.2e-2 | 1.5e-3 | 1.6e-3 |
| `d/dy` | 8.3e-2 | 8.0e-2 | 2.3e-3 | 4.1e-3 |
| Laplacian | 1.7e-1 | 1.9e-1 | 3.9e-3 | 2.2e-2 |

Normal derivative at the 120 boundary nodes: 5.6e-2 relative L2 (classical 1.3e-3).

Read these honestly. **The learned operator is well short of classical in absolute
accuracy** — one to two orders of magnitude on this run, worst on the Laplacian. That
is what the section above predicts, and this run is also under-trained: validation
loss was still falling at epoch 300.

What *does* hold up is the boundary-specific behaviour, which is the point of the
module:

- **Error is flat from interior to wall.** The learned Laplacian goes 1.7e-1 → 1.9e-1
  (1.1×) while the classical one goes 3.9e-3 → 2.2e-2 (5.5×). The learned operator
  degrades far less at the boundary, which is the robustness claim and the thing the
  shipped interior-only checkpoints cannot do at all.
- **Weights stay bounded on bad stencils.** Largest Laplacian weight: NeMDO 307
  interior / 333 wall; classical 977 / 1710. The wall stencils are 3.3× worse
  conditioned, and the penalised network barely notices.
- **The Neumann row exists**, which it did not before.

To close the accuracy gap: more data first (lower `spacing` in
`main_build_dataset.py`; 10k stencils is small), then longer training, then give the
Laplacian its own model rather than sharing a trunk with the first derivatives — the
shared trunk is convenient but the original NeMDO trains one network per operator and
its Laplacian model alone is ~3× the size of this whole one.

To actually beat classical weights on accuracy, the operator has to know something a
polynomial constraint does not — i.e. train a *correction* on solution data from the
PDE you care about, on top of these consistency-trained weights. That keeps
convergence on fine meshes while buying accuracy per degree of freedom on coarse
ones, at the cost of making the correction PDE-specific. This package is the base
layer that such a correction would sit on; it is deliberately not included here.

Other current limits: 2D only; fixed stencil size `K` per trained model; the operator
is not symmetric or sign-constrained, so a heat operator assembled from it is not
guaranteed monotone (constraining the Laplacian head to symmetric non-negative edge
conductances is the natural next change).

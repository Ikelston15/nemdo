"""Invariant checks for the boundary-aware pipeline.

These guard the parts that would fail *silently*: a wrong rotation convention, a
moment matrix that disagrees with the torch loss, or a weight scaling that is off by
a power of h. Each would still train to a plausible-looking loss while producing
operators that are quietly wrong.

Run:
    python test_sanity.py
"""

from __future__ import annotations

import numpy as np
import torch

from functions.boundary import rotate_offsets, weights_to_global
from functions.classical import classical_weights, moment_matrix, moment_residual
from functions.domains import holdout_domain
from functions.moments import (calc_moments_multihead, monomial_basis,
                               physical_scale, target_moments)
from functions.stencils import extract_stencils, star_edge_index

OPS = ("x", "y", "laplace")
ORDER = 2
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> None:
    rng = np.random.default_rng(0)
    cloud = holdout_domain(spacing=0.04)
    st = extract_stencils(cloud, stencil_size=25)
    offs_local = st.offsets_local
    wall = np.flatnonzero(st.is_boundary)
    print(f"Holdout cloud: {cloud.n_nodes} nodes, {len(wall)} on the boundary\n")

    # 1. The classical solve must satisfy the consistency conditions exactly.
    print("1. classical weights reproduce their target moments")
    res = np.array([np.abs(moment_residual(classical_weights(offs_local[i], OPS, ORDER),
                                           offs_local[i], OPS, ORDER)).max()
                    for i in rng.choice(len(st), 60, replace=False)])
    check("interior + wall stencils", res.max() < 1e-9, f"max residual {res.max():.2e}")

    # 2. A degree-2 polynomial must be differentiated exactly by those weights.
    print("\n2. exactness on a quadratic (the consistency order)")
    i = int(wall[0])
    nb = offs_local[i]
    w = classical_weights(nb, OPS, ORDER)
    f = lambda d: 1.0 + 2 * d[:, 0] - 3 * d[:, 1] + 0.5 * d[:, 0] ** 2 + 4 * d[:, 0] * d[:, 1] - d[:, 1] ** 2
    u = f(nb)
    du = u - u[0]
    got = {op: float(du @ w[:, c]) for c, op in enumerate(OPS)}
    want = {"x": 2.0, "y": -3.0, "laplace": 1.0 - 2.0}      # at the centre, d = 0
    for op in OPS:
        check(f"{op} on a wall stencil", abs(got[op] - want[op]) < 1e-8,
              f"got {got[op]:.6f}, want {want[op]:.6f}")

    # 3. Rotation round-trip. Weights solved in the canonical frame and rotated back
    #    must equal weights solved directly on the global-frame offsets.
    print("\n3. canonical-frame weights rotate back to the global frame")
    coords = cloud.coords
    from scipy.spatial import cKDTree
    _, idx = cKDTree(coords).query(coords, k=25)
    offs_global = coords[idx] - coords[:, None, :]
    sel = rng.choice(wall, size=min(25, len(wall)), replace=False)
    err = 0.0
    for i in sel:
        w_loc = classical_weights(offs_local[i], OPS, ORDER)
        gx, gy = weights_to_global(w_loc[None, :, 0], w_loc[None, :, 1],
                                   np.array([st.theta[i]], dtype=float))
        w_glob = classical_weights(offs_global[i], OPS, ORDER)
        err = max(err, np.abs(gx[0] - w_glob[:, 0]).max(), np.abs(gy[0] - w_glob[:, 1]).max())
    scale = np.abs(classical_weights(offs_global[int(sel[0])], OPS, ORDER)).max()
    check("d/dx and d/dy agree", err < 1e-6 * max(scale, 1.0),
          f"max abs diff {err:.2e} (weights are O({scale:.0f}))")

    lap_loc = np.array([classical_weights(offs_local[i], OPS, ORDER)[:, 2] for i in sel])
    lap_glob = np.array([classical_weights(offs_global[i], OPS, ORDER)[:, 2] for i in sel])
    check("laplacian is rotation invariant",
          np.abs(lap_loc - lap_glob).max() < 1e-6 * max(np.abs(lap_glob).max(), 1.0),
          f"max abs diff {np.abs(lap_loc - lap_glob).max():.2e}")

    # 4. Canonical frame really does put the outward normal on +x.
    print("\n4. canonical frame orients the outward normal along +x")
    n_local = rotate_offsets(cloud.normals[wall][:, None, :], st.theta[wall].astype(float))[:, 0, :]
    check("normal maps to (1, 0)", np.abs(n_local - np.array([1.0, 0.0])).max() < 1e-6,
          f"max deviation {np.abs(n_local - np.array([1.0, 0.0])).max():.2e}")

    # 5. The torch loss must compute the same moments as the numpy reference.
    print("\n5. torch moment computation matches the numpy moment matrix")
    k = st.stencil_size
    b = 8
    sel = rng.choice(len(st), b, replace=False)
    x = torch.from_numpy(st.features[sel].reshape(b * k, -1))
    w_np = np.stack([classical_weights(st.features[i][:, :2].astype(np.float64), OPS, ORDER)
                     for i in sel])                                  # normalised offsets
    out = torch.from_numpy(w_np.reshape(b * k, len(OPS)).astype(np.float32))
    batch = torch.arange(b).repeat_interleave(k)
    mp, inv = monomial_basis(ORDER)
    got_t = calc_moments_multihead(x, out, batch, mp, inv, b).numpy()
    want_np = np.stack([moment_matrix(st.features[i][:, :2].astype(np.float64), ORDER) @ w_np[j]
                        for j, i in enumerate(sel)])
    check("moments agree", np.abs(got_t - want_np).max() < 1e-4,
          f"max abs diff {np.abs(got_t - want_np).max():.2e}")

    tgt = target_moments(OPS, ORDER)
    check("and they hit the target", np.abs(got_t - tgt[None]).max() < 1e-4,
          f"max abs diff {np.abs(got_t - tgt[None]).max():.2e}")

    # 6. h-scaling: normalised weights divided by h**q must equal physical weights.
    print("\n6. h-normalisation scaling")
    i = int(wall[1])
    w_norm = classical_weights(st.features[i][:, :2].astype(np.float64), OPS, ORDER)
    w_phys = classical_weights(offs_local[i], OPS, ORDER)
    scaled = w_norm / physical_scale(OPS, st.h[i:i + 1])[0][None, :]
    check("w_normalised / h**order == w_physical",
          np.abs(scaled - w_phys).max() < 1e-6 * np.abs(w_phys).max(),
          f"max rel diff {np.abs(scaled - w_phys).max() / np.abs(w_phys).max():.2e}")

    # 7. Star graph wiring: every neighbour connects to the centre, both ways.
    print("\n7. batched star-graph edge index")
    ei = star_edge_index(5, 3).numpy()
    expect_edges = 3 * 2 * 4
    centres = set()
    for s, d in ei.T:
        centres.add(s if s % 5 == 0 else d)
    check("edge count", ei.shape[1] == expect_edges, f"{ei.shape[1]} == {expect_edges}")
    check("centres are node 0 of each graph", centres == {0, 5, 10}, str(sorted(centres)))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()

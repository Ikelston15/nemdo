import torch
import logging

logger = logging.getLogger(__name__)


class StencilGrowth:
    """
    Decides *when* the stencils should grow (training loss stalled) and *which* points grow (the ones that
    are still doing worst). Every point ends up with its own stencil size.

    A stall is declared when the training loss has not improved by at least `rel_tol` (relative) for `patience`
    consecutive epochs. After a growth event the detector is reset and left alone for `cooldown` epochs so the
    network can adapt to the new stencils.
    """
    def __init__(self,
                 max_size: int,
                 step: int = 2,
                 patience: int = 10,
                 rel_tol: float = 1e-2,
                 cooldown: int = 5,
                 quantile: float = 0.5,
                 converged_tol: float = 0.0):
        self.max_size = max_size
        self.step = step
        self.patience = patience
        self.rel_tol = rel_tol
        self.cooldown = cooldown
        self.quantile = quantile
        self.converged_tol = converged_tol
        self.reset()

    def reset(self):
        self.best = float('inf')
        self.wait = 0
        self.cool = self.cooldown

    def stalled(self, train_loss: float) -> bool:
        """Call once per epoch with the training loss."""
        if self.cool > 0:
            self.cool -= 1
            self.best = min(self.best, train_loss)
            return False
        if train_loss < self.best * (1.0 - self.rel_tol):
            self.best = train_loss
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience

    def grow(self, sizes: torch.Tensor, point_loss: torch.Tensor) -> torch.Tensor:
        """
        Grows (in place) the stencil of the worst-performing points that are not yet at the maximum size.

        :param sizes: shared [N] tensor of per-point stencil sizes (cpu)
        :param point_loss: [N] most recent per-point loss, NaN if the point has not been seen since it last grew
        :return: boolean mask [N] of the points that grew
        """
        point_loss = point_loss.to(sizes.device)
        growable = (sizes < self.max_size) & ~torch.isnan(point_loss) & (point_loss > self.converged_tol)
        if not growable.any():
            return growable

        # sort-based quantile (torch.quantile is limited to 16M elements)
        vals = torch.sort(point_loss[growable].float()).values
        threshold = vals[int(self.quantile * (vals.numel() - 1))]
        mask = growable & (point_loss >= threshold)
        sizes[mask] = torch.clamp(sizes[mask] + self.step, max=self.max_size)
        return mask

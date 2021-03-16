from torch.optim import Adam as AdamOptimizer

from .algorithm_abc import Algorithm

from typing import Tuple
import attr


@attr.s
class Adam(Algorithm):
    lr: float = attr.ib(default=0.001)
    betas: Tuple[float, float] = attr.ib(default=(0.9, 0.999))
    eps: float = attr.ib(default=1e-08)
    weight_decay: float = attr.ib(default=0)
    amsgrad: bool = attr.ib(default=False)

    def instance(self, params):
        return AdamOptimizer(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

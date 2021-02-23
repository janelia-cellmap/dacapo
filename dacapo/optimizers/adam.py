from torch.optim import Adam

from typing import Tuple
import attr


@attr.s
class AdamConfig:
    lr: float = attr.ib(default=0.001)
    betas: Tuple[float, float] = attr.ib(default=(0.9, 0.999))
    eps: float = attr.ib(default=1e-08)
    weight_decay: float = attr.ib(default=0)
    amsgrad: bool = attr.ib(default=False)

    def optimizer(self, params):
        return Adam(
            params,
            self.lr,
            self.betas,
            self.eps,
            self.weight_decay,
            self.degenerated_to_sgd,
        )

import attr

from .optimizers import AnyOptimizer


@attr.s
class OptimizerConfig:
    name: str = attr.ib()
    optimizer: AnyOptimizer = attr.ib()

    def optimizer(self, params):
        return self.optimizer(params)
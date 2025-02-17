from .trainer_config import TrainerConfig
import warnings


class Trainer:
    def __new__(cls, trainer_config: TrainerConfig):
        warnings.warn(
            "Trainer is depricated and doesn't need to be used", DeprecationWarning
        )
        return trainer_config

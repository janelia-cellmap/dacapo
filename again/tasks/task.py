from .data import Data


class Task:

    def __init__(self, task_config):

        self.data = Data(task_config.data)
        self.predictor_type = task_config.predictor
        self.augmentations = task_config.augmentations

        loss_args = {}
        if hasattr(task_config, 'loss_args'):
            loss_args = task_config.loss_args
        self.loss = task_config.loss(**loss_args)

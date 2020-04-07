class Task:

    def __init__(self, data, model, task_config):

        self.data = data

        predictor_args = {}
        if hasattr(task_config, 'predictor_args'):
            predictor_args = task_config.predictor_args
        self.predictor = task_config.predictor(
            self.data,
            model,
            **predictor_args)

        self.augmentations = task_config.augmentations

        loss_args = {}
        if hasattr(task_config, 'loss_args'):
            loss_args = task_config.loss_args
        self.loss = task_config.loss(**loss_args)

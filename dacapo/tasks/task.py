from dacapo.tasks.post_processors import PostProcessingParameterRange


class Task:

    def __init__(self, data, model, task_config):

        self.data = data

        post_processor = None
        if hasattr(task_config, 'post_processor'):
            if hasattr(task_config, 'post_processing_parameter_range'):
                kwargs = task_config.post_processing_parameter_range.to_dict()
                del kwargs['id']
                del kwargs['task']
            else:
                kwargs = {}
            parameter_range = PostProcessingParameterRange(**kwargs)
            post_processor = task_config.post_processor(parameter_range)

        predictor_args = {}
        if hasattr(task_config, 'predictor_args'):
            predictor_args = task_config.predictor_args
        self.predictor = task_config.predictor(
            self.data,
            model,
            post_processor=post_processor,
            **predictor_args)

        self.augmentations = task_config.augmentations

        loss_args = {}
        if hasattr(task_config, 'loss_args'):
            loss_args = task_config.loss_args
        self.loss = task_config.loss(**loss_args)

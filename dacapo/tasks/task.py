from dacapo.tasks.post_processors import PostProcessingParameterRange


class AuxLoss:
    def __init__(self, regular_loss, aux_loss, predictor, aux_predictor):
        self.regular_loss = regular_loss
        self.aux_loss = aux_loss
        self.predictor = predictor
        self.aux_predictor = aux_predictor

    def forward(self, model_output, target, weights=None, aux_target=None):
        predictor_output = self.predictor.forward(model_output)
        aux_output = self.aux_predictor.forward(model_output)
        if weights is not None:
            weights = weights.to("cuda")
            regular_loss = self.regular_loss(predictor_output, target, weights)
        else:
            regular_loss = self.regular_loss(predictor_output, target)
        aux_loss = self.aux_loss(aux_output, aux_target)
        return regular_loss + aux_loss

    def add_weights(self, *args, **kwargs):
        return self.regular_loss.add_weights(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class PredictorLoss:
    def __init__(self, regular_loss, predictor):
        self.regular_loss = regular_loss
        self.predictor = predictor

    def forward(self, model_output, target, weights=None):
        predictor_output = self.predictor.forward(model_output)
        if weights is None:
            return self.regular_loss(predictor_output, target)
        else:
            return self.regular_loss(predictor_output, target, weights)

    def add_weights(self, *args, **kwargs):
        return self.regular_loss.add_weights(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class AuxTask:
    pass


class Task:
    def __init__(self, data, model, task_config):

        post_processor = None
        if hasattr(task_config, "post_processor"):
            if hasattr(task_config, "post_processing_parameter_range"):
                kwargs = task_config.post_processing_parameter_range.to_dict()
                del kwargs["id"]
                del kwargs["task"]
            else:
                kwargs = {}
            parameter_range = PostProcessingParameterRange(**kwargs)
            post_processor = task_config.post_processor(parameter_range)

        predictor_args = {}
        if hasattr(task_config, "predictor_args"):
            predictor_args = task_config.predictor_args
        self.predictor = task_config.predictor(
            data, model, post_processor=post_processor, **predictor_args
        )
        self.model = model

        self.augmentations = task_config.augmentations

        loss_args = {}
        if hasattr(task_config, "loss_args"):
            loss_args = task_config.loss_args
        self.loss = task_config.loss(**loss_args)

        self.aux_task = None
        if hasattr(task_config, "aux_task"):
            self.aux_task = AuxTask()
            predictor_args = {}
            if hasattr(task_config.aux_task, "predictor_args"):
                predictor_args = task_config.aux_task.predictor_args
            self.aux_task.predictor = task_config.aux_task.predictor(
                data, model, **predictor_args
            )

            loss_args = {}
            if hasattr(task_config, "loss_args"):
                loss_args = task_config.loss_args
            self.aux_task.loss = task_config.aux_task.loss(**loss_args)

            self.loss = AuxLoss(
                self.loss, self.aux_task.loss, self.predictor, self.aux_task.predictor
            )
        else:
            self.loss = PredictorLoss(self.loss, self.predictor)

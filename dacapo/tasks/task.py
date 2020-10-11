from dacapo.tasks.post_processors import PostProcessingParameterRange
import torch


class AgglomeratedLoss(torch.nn.Module):
    def __init__(self, predictor, loss, aux_tasks):
        self.predictor = predictor
        self.loss = loss
        self.aux_tasks = aux_tasks

    @property
    def predictors(self):
        yield self.predictor
        for aux_task in self.aux_tasks:
            yield aux_task[1]

    def forward(self, model_output, target, weights=None, **kwargs):
        predictor_output = self.predictor(model_output)
        if weights is not None:
            loss = self.loss(predictor_output, target)
        else:
            loss = self.loss(predictor_output, target, weights)

        aux_outputs = [predictor(model_output) for _, predictor, _ in self.aux_tasks]
        aux_losses = []
        for i in range(self.aux_tasks):
            name, loss, prediction = (
                self.aux_tasks[i][0],
                self.aux_tasks[i][2],
                aux_outputs[i],
            )
            loss_inputs = {
                key[len(name) + 1 :]: value
                for key, value in kwargs
                if key.startswith(f"{name}_")
            }
            aux_loss = loss.forward(prediction, **loss_inputs)
            aux_losses.append(aux_loss)

        loss += torch.sum(aux_losses)
        
        return loss

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)


class AuxTask:
    pass


class Task:
    def __init__(self, data, model, task_config):

        post_processor = None
        if hasattr(task_config, "post_processor"):
            if hasattr(task_config, "post_processing_parameter_range"):
                kwargs = task_config.post_processing_parameter_range.to_dict(
                    default_only=True
                )
                del kwargs["id"]
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

        auxiliary_task_keys = []
        for key in task_config.to_dict().keys():
            if key.startswith("aux_task_"):
                auxiliary_task_keys.append(key[9:])

        # stores tuples of name, predictor, loss
        self.aux_tasks = []
        for auxiliary_task_key in auxiliary_task_keys:
            aux_task_config = getattr(task_config, f"aux_task_{auxiliary_task_key}")
            predictor_args = {}
            if hasattr(aux_task_config, "predictor_args"):
                predictor_args = aux_task_config.predictor_args
            self.aux_tasks.append(
                (
                    auxiliary_task_key,
                    aux_task_config.predictor(**predictor_args),
                    aux_task_config.loss,
                )
            )

        self.total_loss = AgglomeratedLoss(self.predictor, self.loss, self.aux_tasks)

from .task import Task

import torch


class PretrainedTask(Task):
    """
    A task that uses a pretrained model. The model is loaded from a file
    and the weights are loaded from a file.

    Attributes:
        weights (Path): The path to the weights file.
    Methods:
        create_model(self, architecture) -> Model: This method creates a model using the given architecture.
    Notes:
        This is a base class for all tasks that use pretrained models.

    """

    def __init__(self, task_config):
        """
        Initialize the PretrainedTask object.

        Args:
            task_config (TaskConfig): The configuration of the task.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> task = PretrainedTask(task_config)
        """
        sub_task = task_config.sub_task_config.task_type(task_config.sub_task_config)
        self.weights = task_config.weights

        self.predictor = sub_task.predictor
        self.loss = sub_task.loss
        self.post_processor = sub_task.post_processor
        self.evaluator = sub_task.evaluator

    def create_model(self, architecture):
        """
        Create a model using the given architecture.

        Args:
            architecture (str): The architecture of the model.
        Returns:
            Model: The model created using the given architecture.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> model = task.create_model(architecture)

        """
        model = self.predictor.create_model(architecture)

        saved_state_dict = torch.load(str(self.weights))
        model.chain.load_state_dict(saved_state_dict["model"])
        return model

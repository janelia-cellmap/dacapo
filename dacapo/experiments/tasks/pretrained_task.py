from .task import Task

import torch

class PretrainedTask(Task):
    """
    PretrainedTask is a specialized task that initializes a model weights using a pretrained model.
    
    This task uses a pretrained model weights which can have a different head channels
    and then loads pretrained weights into the model created by the predictor.

    Attributes:
        weights (str): The path to the pretrained weights file.
        predictor (Predictor): Inherits the Predictor instance from the sub-task.
        loss (Loss): Inherits the Loss instance from the sub-task.
        post_processor (PostProcessor): Inherits the PostProcessor instance from the sub-task.
        evaluator (Evaluator): Inherits the Evaluator instance from the sub-task.
    """

    def __init__(self, task_config):
        """
        Initializes the PretrainedTask with the specified task configuration.

        The constructor initializes the task by setting up a sub-task based on the provided
        task configuration and then loading the pretrained weights.

        Args:
            task_config: A configuration object for the task, which includes the sub-task 
                         configuration and the path to the pretrained weights.
        """
        sub_task = task_config.sub_task_config.task_type(task_config.sub_task_config)
        self.weights = task_config.weights

        self.predictor = sub_task.predictor
        self.loss = sub_task.loss
        self.post_processor = sub_task.post_processor
        self.evaluator = sub_task.evaluator

    def create_model(self, architecture):
        """
        Creates and returns a model based on the given architecture, with pretrained weights loaded.

        This method creates a model using the predictor's `create_model` method and then loads
        the pretrained weights into the model.

        Args:
            architecture: The architecture specification for the model to be created.

        Returns:
            The model instance with pretrained weights loaded.
        """
        model = self.predictor.create_model(architecture)

        saved_state_dict = torch.load(str(self.weights))
        model.chain.load_state_dict(saved_state_dict["model"])
        return model

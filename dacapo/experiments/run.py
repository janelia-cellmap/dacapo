"""
This class defines a 'Run' object which is mainly used for model training and validation.
All the components like tasks, architectures, trainers, are set with this object.

Attributes:
    name (str): The name of the run.
    train_until (int): The total number of iterations for training.
    validation_interval (int): The interval to conduct validation during training.
    task (Task): The Task object for the run.
    architecture (Architecture): The Architecture object for the model
    trainer (Trainer): The Trainer object for the run.
    datasplit (DataSplit): The DataSplit object for the run.
    model (Model): The Model object for the run.
    optimizer (torch.optim.Optimizer): The optimizer for model training.
    training_stats (TrainingStats): The TrainingStats object for tracking training statistics.
    validation_scores (ValidationScores): The ValidationScores object for tracking validation scores.
    start (Start): The Start object containing weights from a previous run if any.

Methods:
    __init__(run_config): Initializes the Run object with configurations.
    get_validation_scores(run_config): A static method to get validation scores.
    move_optimizer(device, empty_cuda_cache): Moves the optimizer to a specified device.
"""

class Run:
    ...
    def __init__(self, run_config):
        """
        Initializes the Run object with the provided configurations.

        Args:
            run_config: An object containing the configurations for the run.
        """
        ...

    @staticmethod
    def get_validation_scores(run_config) -> ValidationScores:
        """
        Static method to avoid having to initialize model, optimizer, trainer, etc.
        This method is used to compute and return validation scores.

        Args:
            run_config: An object containing the configurations for the run.

        Returns:
            The ValidationScores object containing validation scores.
        """
        ...

    def move_optimizer(
        self, device: torch.device, empty_cuda_cache: bool = False
    ) -> None:
        """
        Moves the optimizer to a certain device which can be cpu or gpu.
        Also, it has an option to clear the GPU memory/cache.

        Args:
            device (torch.device): The device to which the optimizer needs to be moved.
            empty_cuda_cache (bool): If True, it will clear the GPU memory/cache.

        Returns:
            None
        """
        ...
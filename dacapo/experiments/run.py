from .datasplits import DataSplit
from .tasks.task import Task
from .architectures import Architecture
from .trainers import Trainer, GunpowderTrainer
from .training_stats import TrainingStats
from .validation_scores import ValidationScores
from .starts import Start
from .model import Model
from typing import Optional

import torch


class Run:
    """
    Class representing a run in the experiment. A run is a combination of a task, architecture, trainer, datasplit,
    model, optimizer, training stats, and validation scores. It also contains the name of the run, the number of
    iterations to train for, and the interval at which to validate. It also contains a start object that can be used to
    initialize the model with preloaded weights. The run object can be used to move the optimizer to a specified device.

    Attributes:
        name (str): The name of the run.
        train_until (int): The number of iterations to train for.
        validation_interval (int): The interval at which to validate.
        task (Task): The task object.
        architecture (Architecture): The architecture object.
        trainer (Trainer): The trainer object.
        datasplit (DataSplit): The datasplit object.
        model (Model): The model object.
        optimizer (torch.optim.Optimizer): The optimizer object.
        training_stats (TrainingStats): The training stats object.
        validation_scores (ValidationScores): The validation scores object.
        start (Start): The start object.
    Methods:
        move_optimizer(device: torch.device, empty_cuda_cache: bool) -> None:
            Moves the optimizer to the specified device.
        get_validation_scores(run_config) -> ValidationScores:
            Static method to get the validation scores without initializing model, optimizer, trainer, etc.
    Note:
        The iteration stats list is structured as follows:
        - The outer list contains the stats for each iteration.
        - The inner list contains the stats for each training iteration.
    """

    name: str
    train_until: int
    validation_interval: int

    task: Task
    architecture: Architecture
    trainer: Trainer
    _datasplit: Optional[DataSplit]

    model: Model
    optimizer: torch.optim.Optimizer

    training_stats: TrainingStats
    _validation_scores: Optional[ValidationScores]

    def __init__(self, run_config, load_starter_model: bool = True):
        """
        Initializes a Run object.

        Args:
            run_config: The configuration for the run.
        Raises:
            AssertionError: If the task, architecture, trainer, or datasplit types are not specified in the run_config.
        Examples:
            >>> run = Run(run_config)
            >>> run.name
            'run_name'
            >>> run.train_until
            100
            >>> run.validation_interval
            10
            >>> run.task
            Task object
            >>> run.architecture
            Architecture object
            >>> run.trainer
            Trainer object
            >>> run.datasplit
            DataSplit object
            >>> run.model
            Model object
            >>> run.optimizer
            Optimizer object
            >>> run.training_stats
            TrainingStats object
            >>> run.validation_scores
            ValidationScores object
            >>> run.start
            Start object

        """
        self.name = run_config.name
        self._config = run_config
        self.train_until = run_config.num_iterations
        self.validation_interval = run_config.validation_interval

        # config types
        task_type = run_config.task_config.task_type
        architecture_type = run_config.architecture_config.architecture_type
        trainer_type = run_config.trainer_config.trainer_type
        datasplit_type = run_config.datasplit_config.datasplit_type

        # run components
        self.task = task_type(run_config.task_config)
        self.architecture = architecture_type(run_config.architecture_config)
        self.trainer = trainer_type(run_config.trainer_config)

        # lazy load datasplit
        self._datasplit = None

        # combined pieces
        self.model = self.task.create_model(self.architecture)
        self.optimizer = self.trainer.create_optimizer(self.model)

        # tracking
        self.training_stats = TrainingStats()
        self._validation_scores = None

        if not load_starter_model:
            self.start = None
            return

        # preloaded weights from previous run
        self.start = (
            (
                run_config.start_config.start_type(run_config.start_config)
                if hasattr(run_config.start_config, "start_type")
                else Start(run_config.start_config)
            )
            if run_config.start_config is not None
            else None
        )
        if self.start is None:
            return

        new_head = None
        if hasattr(run_config, "task_config"):
            if hasattr(run_config.task_config, "channels"):
                new_head = run_config.task_config.channels

        self.start.initialize_weights(self.model, new_head=new_head)

    @property
    def datasplit(self):
        if self._datasplit is None:
            self._datasplit = self._config.datasplit_config.datasplit_type(
                self._config.datasplit_config
            )
        return self._datasplit

    @property
    def validation_scores(self):
        if self._validation_scores is None:
            self._validation_scores = ValidationScores(
                self.task.parameters,
                self.datasplit.validate,
                self.task.evaluation_scores,
            )
        return self._validation_scores

    @staticmethod
    def get_validation_scores(run_config) -> ValidationScores:
        """
        Static method to get the validation scores without initializing model, optimizer, trainer, etc.

        Args:
            run_config: The configuration for the run.
        Returns:
            The validation scores.
        Raises:
            AssertionError: If the task or datasplit types are not specified in the run_config.
        Examples:
            >>> validation_scores = Run.get_validation_scores(run_config)
            >>> validation_scores
            ValidationScores object

        """
        task_type = run_config.task_config.task_type
        datasplit_type = run_config.datasplit_config.datasplit_type

        task = task_type(run_config.task_config)
        datasplit = datasplit_type(run_config.datasplit_config)

        return ValidationScores(
            task.parameters, datasplit.validate, task.evaluation_scores
        )

    def move_optimizer(
        self, device: torch.device, empty_cuda_cache: bool = False
    ) -> None:
        """
        Moves the optimizer to the specified device.

        Args:
            device: The device to move the optimizer to.
            empty_cuda_cache: Whether to empty the CUDA cache after moving the optimizer.
        Raises:
            AssertionError: If the optimizer state is not a dictionary.
        Examples:
            >>> run.move_optimizer(device)
            >>> run.optimizer
            Optimizer object

        """
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        if empty_cuda_cache:
            torch.cuda.empty_cache()

    def __str__(self):
        return self.name

    def visualize_pipeline(self, bind_address="0.0.0.0", bind_port=0):
        """
        Visualizes the pipeline for the run, including all produced arrays.

        Args:
            bind_address : str
                Bind address for Neuroglancer webserver
            bind_port : int
                Bind port for Neuroglancer webserver

        Examples:
            >>> run.visualize_pipeline()

        """
        if not isinstance(self.trainer, GunpowderTrainer):
            raise NotImplementedError(
                "Only GunpowderTrainer is supported for visualization"
            )
        if not hasattr(self.trainer, "_pipeline"):
            from ..store.create_store import create_array_store

            array_store = create_array_store()
            self.trainer.build_batch_provider(
                self.datasplit.train,
                self.model,
                self.task,
                array_store.snapshot_container(self.name),
            )
        self.trainer.visualize_pipeline(bind_address, bind_port)

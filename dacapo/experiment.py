from datetime import datetime
from typing import Optional
import attr
from .experiments import (
    DataConfig,
    TaskConfig,
    ArchitectureConfig,
    TrainerConfig,
    StartConfig,
)
from .experiments.architectures import CNNectomeUNetConfig, CNNectomeUNet
from .experiments.trainers import GunpowderTrainerConfig


@attr.s
class Experiment:
    """Class for experiment configuration. This class takes in the necessary configurations for an experiment and runs, logs and saves the experiment configuration and results."""

    data_config: DataConfig = attr.ib(
        metadata={
            "help_text": "A config defining the data to use for this experiment, including training and validation split."
        }
    )
    task_config: TaskConfig = attr.ib(
        metadata={
            "help_text": "A config defining the Task to run. The task defines the target output "
            "of your model. Do you want semantic segmentations, instance segmentations, "
            "or something else? The task also lets you choose from different methods of "
            "achieving each of these goals. The parameters for the task are also defined here."
        }
    )
    architecture_config: ArchitectureConfig = attr.ib(
        default=None,
        metadata={
            "help_text": "A config defining the Architecture to train. The architecture defines "
            "the backbone of your model. The majority of your models weights will be "
            "defined by the Architecture and will be very impactful on your models "
            "performance. There is no need to worry about the output since depending "
            "on the chosen task, additional layers will be appended to make sure "
            "the output conforms to the expected format."
        },
    )
    trainer_config: TrainerConfig = attr.ib(
        default=None,
        metadata={
            "help_text": "The trainer config defines everything related to how batches are generated "
            "and passed to the model for training. Things such as augmentations (adding noise, "
            "random rotations, transposing, etc.), batch size, learning rate, number of workers "
            "and snapshot logging will be configured here."
        },
    )
    start_config: Optional[StartConfig] = attr.ib(
        default=None,
        metadata={
            "help_text": "The start config defines the starting point of the experiment. This can be a pretrained model, a checkpoint, or a new model."
        },
    )
    num_iterations: int = attr.ib(
        default=1000,
        metadata={"help_text": "The number of iterations to train for."},
    )
    num_replicates: int = attr.ib(
        default=1,
        metadata={"help_text": "The number of replicates to run for this experiment."},
    )
    validation_interval: int = attr.ib(
        default=1000, metadata={"help_text": "How often to perform validation."}
    )
    overwrite: bool = attr.ib(
        default=False,
        metadata={
            "help_text": "Whether to overwrite the results of a previous experiment with the same name."
        },
    )
    name: str = attr.ib(
        default=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        metadata={
            "help_text": "A unique name for this experiment. This will be saved so "
            "you and others can find and reuse this experiment. Avoid special "
            "characters."
        },
    )
    description: Optional[str] = attr.ib(
        default="Unspecified experiment.",
        metadata={"help_text": "A brief description of this experiment."},
    )

    def __attrs_post_init__(self):
        if self.architecture_config is None:
            # figure out necessary parameters for architecture
            options = {} TODO
            self.architecture_config = CNNectomeUNetConfig(**options)
        if self.trainer_config is None:
            # figure out necessary parameters for trainer
            options = {} TODO
            self.trainer_config = GunpowderTrainerConfig(**options)
        ...


    
    def run(self):
        """
        Run the experiment
        """
        ...

    def report(self):
        """
        Report the results of the experiment
        """
        ...

    def get_model(self, criterion: Optional[str] = None, validation_dataset: Optional[str] = None):
        """
        Get the model from the experiment that has performed the best on the validation dataset for the given criterion. Default is to use the tasks default criterion and the entire validation dataset list from the training datasplit.
        """
        if criterion is None:
            criterion = self.task_config.default_criterion
        if validation_dataset is not None:
            # make sure the given dataset has been validated for the given criterion, otherwise warn and then run validation
            ...
        ...
        return model
    
    def export(self, path: str):
        """
        Export the experiment to a file for publication.
        """
        ...

    def verify(self) -> bool:
        """
        Check whether this is a valid experiment
        """
        return True

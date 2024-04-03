import attr

from .architectures import ArchitectureConfig
from .datasplits import DataSplitConfig
from .tasks import TaskConfig
from .trainers import TrainerConfig
from .starts import StartConfig

from typing import Optional


@attr.s
class RunConfig:
    """
    A class to represent a configuration of a run that helps to structure all the tasks,
    architecture, training, and datasplit configurations.

    ...

    Attributes:
    -----------
    task_config: `TaskConfig`
        A config defining the Task to run that includes deciding the output of the model and
        different methods to achieve the goal.

    architecture_config: `ArchitectureConfig`
         A config that defines the backbone architecture of the model. It impacts the model's
         performance significantly.

    trainer_config: `TrainerConfig`
        Defines how batches are generated and passed for training the model along with defining
        configurations like batch size, learning rate, number of cpu workers and snapshot logging.

    datasplit_config: `DataSplitConfig`
        Configures the data available for the model during training or validation phases.

    name: str
        A unique name for this run to distinguish it.

    repetition: int
        The repetition number of this run.

    num_iterations: int
        The total number of iterations to train for during this run.

    validation_interval: int
        Specifies how often to perform validation during the run. It defaults to 1000.

    start_config : `Optional[StartConfig]`
        A starting point for continued training. It is optional and can be left out.
    """

    task_config: TaskConfig = attr.ib(
        metadata={
            "help_text": "A config defining the Task to run. The task defines the output "
            "of your model. Do you want semantic segmentations, instance segmentations, "
            "or something else? The task also lets you choose from different methods of "
            "achieving each of these goals."
        }
    )
    architecture_config: ArchitectureConfig = attr.ib(
        metadata={
            "help_text": "A config defining the Architecture to train. The architecture defines "
            "the backbone of your model. The majority of your models weights will be "
            "defined by the Architecture and will be very impactful on your models "
            "performance. There is no need to worry about the output since depending "
            "on the chosen task, additional layers will be appended to make sure "
            "the output conforms to the expected format."
        }
    )
    trainer_config: TrainerConfig = attr.ib(
        metadata={
            "help_text": "The trainer config defines everything related to how batches are generated "
            "and passed to the model for training. Things such as augmentations (adding noise, "
            "random rotations, transposing, etc.), batch size, learning rate, number of cpu_workers "
            "and snapshot logging will be configured here."
        }
    )
    datasplit_config: DataSplitConfig = attr.ib(
        metadata={
            "help_text": "The datasplit config defines what data will be available for your model during "
            "training or validation. Usually this involves simply reading data from a zarr, "
            "but if there is any preprocessing that needs to be done, that can be configured here."
        }
    )

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this run. This will be saved so you and "
            "others can find this run. Keep it short and avoid special "
            "characters."
        }
    )

    repetition: int = attr.ib(
        metadata={"help_text": "The repetition number of this run."}
    )
    num_iterations: int = attr.ib(
        metadata={"help_text": "The number of iterations to train for."}
    )

    validation_interval: int = attr.ib(
        default=1000, metadata={"help_text": "How often to perform validation."}
    )

    start_config: Optional[StartConfig] = attr.ib(
        default=None, metadata={"help_text": "A starting point for continued training."}
    )

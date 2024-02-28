import attr

from .architectures import ArchitectureConfig
from .datasplits import DataSplitConfig
from .tasks import TaskConfig
from .trainers import TrainerConfig
from .starts import StartConfig

from typing import Optional


@attr.s
class RunConfig:
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
    dataset_adapter_config: DatasetAdapterConfig = attr.ib(...)

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

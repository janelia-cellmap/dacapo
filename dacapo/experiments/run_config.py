import attr
import tempfile
import hashlib
import numpy as np

from funlib.geometry import Coordinate

from dacapo.store.converter import converter
from .architectures import ArchitectureConfig
from .datasplits import DataSplitConfig, DataSplit
from .tasks import TaskConfig, Task
from .trainers import TrainerConfig, Trainer, GunpowderTrainer
from .starts import StartConfig
from .training_stats import TrainingStats
from .validation_scores import ValidationScores
from .model import Model

from bioimageio.core import test_model
from bioimageio.spec import save_bioimageio_package
from bioimageio.spec.model.v0_5 import (
    ModelDescr,
    WeightsDescr,
    PytorchStateDictWeightsDescr,
    Author,
    CiteEntry,
    LicenseId,
    HttpUrl,
    ArchitectureFromLibraryDescr,
    OutputTensorDescr,
    InputTensorDescr,
    BatchAxis,
    ChannelAxis,
    SpaceInputAxis,
    SpaceOutputAxis,
    Identifier,
    AxisId,
    TensorId,
    SizeReference,
    FileDescr,
    Doi,
    IntervalOrRatioDataDescr,
    ParameterizedSize,
    Version,
)

import torch

from typing import Optional
from pathlib import Path


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
    task_config: TaskConfig | None = attr.ib(
        default=None,
        metadata={
            "help_text": "A config defining the Task to run. The task defines the output "
            "of your model. Do you want semantic segmentations, instance segmentations, "
            "or something else? The task also lets you choose from different methods of "
            "achieving each of these goals."
        },
    )
    trainer_config: TrainerConfig | None = attr.ib(
        default=None,
        metadata={
            "help_text": "The trainer config defines everything related to how batches are generated "
            "and passed to the model for training. Things such as augmentations (adding noise, "
            "random rotations, transposing, etc.), batch size, learning rate, number of cpu_workers "
            "and snapshot logging will be configured here."
        },
    )
    datasplit_config: DataSplitConfig | None = attr.ib(
        default=None,
        metadata={
            "help_text": "The datasplit config defines what data will be available for your model during "
            "training or validation. Usually this involves simply reading data from a zarr, "
            "but if there is any preprocessing that needs to be done, that can be configured here."
        },
    )

    name: str | None = attr.ib(
        default=None,
        metadata={
            "help_text": "A unique name for this run. This will be saved so you and "
            "others can find this run. Keep it short and avoid special "
            "characters."
        },
    )

    repetition: int | None = attr.ib(
        default=None, metadata={"help_text": "The repetition number of this run."}
    )
    num_iterations: int | None = attr.ib(
        default=None, metadata={"help_text": "The number of iterations to train for."}
    )

    validation_interval: int = attr.ib(
        default=1000, metadata={"help_text": "How often to perform validation."}
    )

    start_config: Optional[StartConfig] = attr.ib(
        default=None, metadata={"help_text": "A starting point for continued training."}
    )

    _optimizer: Optional[torch.optim.Optimizer] = None
    _model: Optional[torch.nn.Module] = None
    _datasplit: Optional[DataSplitConfig] = None
    _task: Optional[Task] = None
    _trainer: Optional[Trainer] = None
    _training_stats: Optional[TrainingStats] = None
    _validation_scores: Optional[ValidationScores] = None

    @property
    def train_until(self) -> int:
        return self.num_iterations

    @property
    def task(self) -> Task | None:
        if self._task is None and self.task_config is not None:
            self._task = self.task_config.task_type(self.task_config)
        return self._task

    @property
    def architecture(self) -> ArchitectureConfig:
        return self.architecture_config

    @property
    def trainer(self) -> Trainer:
        if self._trainer is None:
            self._trainer = self.trainer_config.trainer_type(self.trainer_config)
        return self._trainer

    @property
    def datasplit(self) -> DataSplit:
        if self._datasplit is None:
            self._datasplit = self.datasplit_config.datasplit_type(
                self.datasplit_config
            )
        return self._datasplit

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            if self.task is not None:
                self._model = self.task.create_model(self.architecture)
            else:
                self._model = Model(self.architecture, torch.nn.Identity(), None)
            if self.start_config is not None:
                self.start_config.start_type(self.start_config).initialize_weights(
                    self._model, None
                )
        return self._model

    @model.setter
    def model(self, value: torch.nn.Module):
        self._model = value

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = self.trainer.create_optimizer(self.model)
        return self._optimizer

    @property
    def training_stats(self):
        if self._training_stats is None:
            self._training_stats = TrainingStats()
        return self._training_stats

    @training_stats.setter
    def training_stats(self, value: TrainingStats):
        self._training_stats = value

    @property
    def validation_scores(self):
        if self._validation_scores is None and self.task is not None:
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

    def save_bioimage_io_model(
        self,
        path: Path,
        authors: list[Author],
        cite: list[CiteEntry] | None = None,
        license: str = "MIT",
        input_test_image_path: Path | None = None,
        output_test_image_path: Path | None = None,
        checkpoint: int | str | None = "latest",
        in_voxel_size: Coordinate | None = None,
    ):
        # TODO: Fix this import. Importing here due to circular imports.
        # The weights store takes a Run to figure out where weights are saved,
        # but Run needs to import the weights store to load the weights.
        from dacapo.store.create_store import create_weights_store

        weights_store = create_weights_store()
        if checkpoint == "latest":
            checkpoint = weights_store.latest_iteration(self)
        if checkpoint is not None:
            weights_store.load_weights(self, checkpoint)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_axes = [
                BatchAxis(),
                ChannelAxis(
                    channel_names=[
                        Identifier(f"in_c{i}")
                        for i in range(self.architecture.num_in_channels)
                    ]
                ),
            ]
            input_shape = self.architecture.input_shape
            in_voxel_size = (
                np.array(in_voxel_size)
                if in_voxel_size is not None
                else np.array((1,) * input_shape.dims)
            )

            input_axes += [
                SpaceInputAxis(
                    id=AxisId(f"d{i}"),
                    size=ParameterizedSize(min=s, step=s),
                    scale=scale,
                )
                for i, (s, scale) in enumerate(zip(input_shape, in_voxel_size))
            ]
            data_descr = IntervalOrRatioDataDescr(type="float32")

            if input_test_image_path is None:
                input_test_image_path = tmp / "input_test_image.npy"
                test_image = np.random.random(
                    (
                        1,
                        self.architecture.num_in_channels,
                        *self.architecture.input_shape,
                    )
                ).astype(np.float32)
                np.save(input_test_image_path, test_image)

            input_descr = InputTensorDescr(
                id=TensorId("raw"),
                axes=input_axes,
                test_tensor=FileDescr(source=str(input_test_image_path)),
                data=data_descr,
            )

            output_shape = self.model.compute_output_shape(input_shape)[1]
            out_voxel_size = self.model.scale(in_voxel_size)
            context_units = Coordinate(
                np.array(input_shape) * in_voxel_size
            ) - Coordinate(np.array(output_shape) * out_voxel_size)
            context_out_voxels = Coordinate(np.array(context_units) / out_voxel_size)

            output_axes = [
                BatchAxis(),
                ChannelAxis(
                    channel_names=(
                        self.task.channels
                        if self.task is not None
                        else [
                            f"c{i}" for i in range(self.architecture.num_out_channels)
                        ]
                    )
                ),
            ]
            output_axes += [
                SpaceOutputAxis(
                    id=AxisId(f"d{i}"),
                    size=SizeReference(
                        tensor_id=TensorId("raw"),
                        axis_id=AxisId(f"d{i}"),
                        offset=-c,
                    ),
                    scale=s,
                )
                for i, (c, s) in enumerate(zip(context_out_voxels, out_voxel_size))
            ]
            if output_test_image_path is None:
                output_test_image_path = tmp / "output_test_image.npy"
                test_out_image = (
                    self.model.eval()(torch.from_numpy(test_image).float())
                    .detach()
                    .numpy()
                )
                np.save(output_test_image_path, test_out_image)
            output_descr = OutputTensorDescr(
                id=TensorId(
                    self.task.__class__.__name__.lower().replace("task", "")
                    if self.task is not None
                    else self.architecture_config.name
                ),
                axes=output_axes,
                test_tensor=FileDescr(source=str(output_test_image_path)),
            )

            pytorch_architecture = ArchitectureFromLibraryDescr(
                callable="from_yaml",
                kwargs={"config_yaml": converter.unstructure(self)},
                import_from="dacapo.experiments.run_config",
            )

            weights_path = tmp / "model.pth"
            torch.save(self.model.state_dict(), weights_path)
            with open(weights_path, "rb", buffering=0) as f:
                weights_hash = hashlib.file_digest(f, "sha256").hexdigest()

            my_model_descr = ModelDescr(
                name=self.name,
                description="A model trained with DaCapo",
                authors=authors,
                cite=[
                    CiteEntry(
                        text="paper",
                        doi=Doi("10.1234something"),
                    )
                ],
                license=LicenseId(license),
                documentation=HttpUrl(
                    "https://github.com/janelia-cellmap/dacapo/blob/main/README.md"
                ),
                git_repo=HttpUrl("https://github.com/janelia-cellmap/dacapo"),
                inputs=[input_descr],
                outputs=[output_descr],
                weights=WeightsDescr(
                    pytorch_state_dict=PytorchStateDictWeightsDescr(
                        source=weights_path,
                        sha256=weights_hash,
                        architecture=pytorch_architecture,
                        pytorch_version=Version(torch.__version__),
                    ),
                ),
            )

            summary = test_model(my_model_descr)
            summary.display()

            print(
                "package path:",
                save_bioimageio_package(my_model_descr, output_path=path),
            )


def from_yaml(config_yaml: dict) -> torch.nn.Module:
    run_config: RunConfig = converter.structure(config_yaml, RunConfig)
    return run_config.model

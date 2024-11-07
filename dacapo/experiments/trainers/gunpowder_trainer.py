from ..training_iteration_stats import TrainingIterationStats
from .trainer import Trainer
from dacapo.tmp import (
    create_from_identifier,
    open_from_identifier,
    gp_to_funlib_array,
    np_to_funlib_array,
)

from dacapo.gp import (
    GraphSource,
    DaCapoTargetFilter,
    CopyMask,
)

from funlib.geometry import Coordinate
from funlib.persistence import Array
import gunpowder as gp

import zarr
import torch
import numpy as np

import time
import logging

logger = logging.getLogger(__name__)


class GunpowderTrainer(Trainer):
    """
    GunpowderTrainer class for training a model using gunpowder. This class is a subclass of the Trainer class. It
    implements the abstract methods defined in the Trainer class. The GunpowderTrainer class is used to train a model
    using gunpowder, a data loading and augmentation library. It is used to train a model on a dataset using a specific
    task.

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The size of the training batch.
        num_data_fetchers (int): The number of data fetchers.
        print_profiling (int): The number of iterations after which to print profiling stats.
        snapshot_iteration (int): The number of iterations after which to save a snapshot.
        min_masked (float): The minimum value of the mask.
        augments (List[Augment]): The list of augmentations to apply to the data.
        mask_integral_downsample_factor (int): The downsample factor for the mask integral.
        clip_raw (bool): Whether to clip the raw data.
        scheduler (torch.optim.lr_scheduler.LinearLR): The learning rate scheduler.
    Methods:
        create_optimizer(model: Model) -> torch.optim.Optimizer:
            Creates an optimizer for the model.
        build_batch_provider(datasets: List[Dataset], model: Model, task: Task, snapshot_container: LocalContainerIdentifier) -> None:
            Initializes the training pipeline using various components.
        iterate(num_iterations: int, model: Model, optimizer: torch.optim.Optimizer, device: torch.device) -> Iterator[TrainingIterationStats]:
            Performs a number of training iterations.
        __iter__() -> Iterator[None]:
            Initializes the training pipeline.
        next() -> Tuple[NumpyArray, NumpyArray, NumpyArray, NumpyArray, NumpyArray]:
            Fetches the next batch of data.
        __enter__() -> GunpowderTrainer:
            Enters the context manager.
        __exit__(exc_type, exc_val, exc_tb) -> None:
            Exits the context manager.
        can_train(datasets: List[Dataset]) -> bool:
            Checks if the trainer can train with a specific set of datasets.
    Note:
        The GunpowderTrainer class is a subclass of the Trainer class. It is used to train a model using gunpowder.

    """

    iteration = 0

    def __init__(self, trainer_config):
        """
        Initializes the GunpowderTrainer object.

        Args:
            trainer_config (TrainerConfig): The trainer configuration.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> trainer = GunpowderTrainer(trainer_config)

        """
        self.learning_rate = trainer_config.learning_rate
        self.batch_size = trainer_config.batch_size
        self.num_data_fetchers = trainer_config.num_data_fetchers
        self.print_profiling = 100
        self.snapshot_iteration = trainer_config.snapshot_interval
        self.min_masked = trainer_config.min_masked

        self.augments = trainer_config.augments
        self.mask_integral_downsample_factor = 4
        self.clip_raw = trainer_config.clip_raw
        self.gt_min_reject = trainer_config.gt_min_reject

        self.scheduler = None

    def create_optimizer(self, model):
        """
        Creates an optimizer for the model.

        Args:
            model (Model): The model for which the optimizer will be created.
        Returns:
            torch.optim.Optimizer: The optimizer created for the model.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> optimizer = trainer.create_optimizer(model)

        """
        optimizer = torch.optim.RAdam(
            lr=self.learning_rate,
            params=model.parameters(),
            decoupled_weight_decay=True,
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=1000,
            last_epoch=-1,
        )
        return optimizer

    def build_batch_provider(self, datasets, model, task, snapshot_container=None):
        """
        Initializes the training pipeline using various components.

        Args:
            datasets (List[Dataset]): The list of datasets.
            model (Model): The model to be trained.
            task (Task): The task to be performed.
            snapshot_container (LocalContainerIdentifier): The snapshot container.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> trainer.build_batch_provider(datasets, model, task, snapshot_container)

        """
        input_shape = Coordinate(model.input_shape)
        output_shape = Coordinate(model.output_shape)

        # get voxel sizes
        raw_voxel_size = datasets[0].raw.voxel_size
        prediction_voxel_size = model.scale(raw_voxel_size)

        # define input and output size:
        # switch to world units
        input_size = raw_voxel_size * input_shape
        output_size = prediction_voxel_size * output_shape

        # define keys:
        raw_key = gp.ArrayKey("RAW")
        gt_key = gp.ArrayKey("GT")
        mask_key = gp.ArrayKey("MASK")

        # make requests such that the mask placeholder is not empty. request single voxel
        # this means we can pad gt and mask as much as we want and not worry about
        # never retrieving an empty gt.
        # as long as the gt is large enough to accomidate one voxel we shouldn't have errors
        mask_placeholder = gp.ArrayKey("MASK_PLACEHOLDER")

        target_key = gp.ArrayKey("TARGET")
        weight_key = gp.ArrayKey("WEIGHT")
        sample_points_key = gp.GraphKey("SAMPLE_POINTS")

        # Get source nodes
        dataset_sources = []
        weights = []
        for dataset in datasets:
            weights.append(dataset.weight)
            assert isinstance(dataset.weight, int), dataset

            raw_source = gp.ArraySource(raw_key, dataset.raw)
            if self.clip_raw:
                raw_source += gp.Crop(
                    raw_key, dataset.gt.roi.snap_to_grid(dataset.raw.voxel_size)
                )
            gt_source = gp.ArraySource(gt_key, dataset.gt)
            sample_points = dataset.sample_points
            points_source = None
            if sample_points is not None:
                graph = gp.Graph(
                    [gp.Node(i, np.array(loc)) for i, loc in enumerate(sample_points)],
                    [],
                    gp.GraphSpec(dataset.gt.roi),
                )
                points_source = GraphSource(sample_points_key, graph)
            if dataset.mask is not None:
                mask_source = gp.ArraySource(mask_key, dataset.mask)
            else:
                # Always provide a mask. By default it is simply an array
                # of ones with the same shape/roi as gt. Avoids making us
                # specially handle no mask case and allows padding of the
                # ground truth without worrying about training on incorrect
                # data.
                mask_source = gp.ArraySource(
                    mask_key,
                    Array(
                        np.ones(dataset.gt.data.shape, dtype=dataset.gt.data.dtype),
                        offset=dataset.gt.roi.offset,
                        voxel_size=dataset.gt.voxel_size,
                        axis_names=dataset.gt.axis_names,
                        units=dataset.gt.units,
                    ),
                )
            array_sources = [raw_source, gt_source, mask_source] + (
                [points_source] if points_source is not None else []
            )

            dataset_source = (
                tuple(array_sources)
                + gp.MergeProvider()
                + CopyMask(
                    mask_key,
                    mask_placeholder,
                    drop_channels=True,
                )
                + gp.Pad(raw_key, None)
                + gp.Pad(gt_key, None)
                + gp.Pad(mask_key, None)
                + gp.RandomLocation(
                    ensure_nonempty=(
                        sample_points_key if points_source is not None else None
                    ),
                    ensure_centered=(
                        sample_points_key if points_source is not None else None
                    ),
                )
            )

            dataset_source += gp.Reject(mask_placeholder, 1e-6)
            if self.gt_min_reject is not None:
                dataset_source += gp.Reject(gt_key, self.gt_min_reject)

            for augment in self.augments:
                dataset_source += augment.node(raw_key, gt_key, mask_key)

            dataset_sources.append(dataset_source)
        pipeline = tuple(dataset_sources) + gp.RandomProvider(weights)

        # Add predictor nodes to pipeline
        pipeline += DaCapoTargetFilter(
            task.predictor,
            gt_key=gt_key,
            target_key=target_key,
            weights_key=weight_key,
            mask_key=mask_key,
        )

        # Trainer attributes:
        if self.num_data_fetchers > 1:
            pipeline += gp.PreCache(num_workers=self.num_data_fetchers)

        # stack to create a batch dimension
        pipeline += gp.Stack(self.batch_size)

        # print profiling stats
        pipeline += gp.PrintProfilingStats(every=self.print_profiling)

        # generate request for all necessary inputs to training
        request = gp.BatchRequest()
        request.add(raw_key, input_size)
        request.add(target_key, output_size)
        request.add(weight_key, output_size)
        request.add(
            mask_placeholder,
            prediction_voxel_size * self.mask_integral_downsample_factor,
        )
        # request additional keys for snapshots
        request.add(gt_key, output_size)
        request.add(mask_key, output_size)
        request[mask_placeholder].roi = request[mask_placeholder].roi.snap_to_grid(
            prediction_voxel_size * self.mask_integral_downsample_factor
        )

        self._request = request
        self._pipeline = pipeline
        self._raw_key = raw_key
        self._gt_key = gt_key
        self._mask_key = mask_key
        self._weight_key = weight_key
        self._target_key = target_key
        self._loss = task.loss

        self.snapshot_container = snapshot_container

    def iterate(self, num_iterations, model, optimizer, device):
        """
        Performs a number of training iterations.

        Args:
            num_iterations (int): The number of training iterations.
            model (Model): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            device (torch.device): The device (GPU/CPU) where the model will be trained.
        Returns:
            Iterator[TrainingIterationStats]: An iterator of the training statistics.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> for iteration_stats in trainer.iterate(num_iterations, model, optimizer, device):
            >>>     print(iteration_stats)

        """
        t_start_fetch = time.time()

        logger.debug("Starting iteration!")

        for iteration in range(self.iteration, self.iteration + num_iterations):
            raw, gt, target, weight, mask = self.next()
            logger.debug(
                f"Trainer fetch batch took {time.time() - t_start_fetch} seconds"
            )

            for param in model.parameters():
                param.grad = None

            t_start_prediction = time.time()
            predicted = model.forward(torch.as_tensor(raw[raw.roi]).float().to(device))
            predicted.retain_grad()
            loss = self._loss.compute(
                predicted,
                torch.as_tensor(target[target.roi]).float().to(device),
                torch.as_tensor(weight[weight.roi]).float().to(device),
            )
            loss.backward()
            optimizer.step()

            if (
                self.snapshot_iteration is not None
                and iteration % self.snapshot_iteration == 0
            ):
                snapshot_zarr = zarr.open(self.snapshot_container.container, "a")
                # remove batch dim from all snapshot arrays
                snapshot_arrays = {
                    "volumes/raw": np_to_funlib_array(
                        raw[0], offset=raw.offset, voxel_size=raw.voxel_size
                    ),
                    "volumes/gt": np_to_funlib_array(
                        gt[0], offset=gt.offset, voxel_size=gt.voxel_size
                    ),
                    "volumes/target": np_to_funlib_array(
                        target[0], offset=target.offset, voxel_size=target.voxel_size
                    ),
                    "volumes/weight": np_to_funlib_array(
                        weight[0], offset=weight.offset, voxel_size=weight.voxel_size
                    ),
                    "volumes/prediction": np_to_funlib_array(
                        predicted.detach().cpu().numpy()[0],
                        offset=target.roi.offset,
                        voxel_size=target.voxel_size,
                    ),
                    "volumes/gradients": np_to_funlib_array(
                        predicted.grad.detach().cpu().numpy()[0],
                        offset=target.roi.offset,
                        voxel_size=target.voxel_size,
                    ),
                }
                if mask is not None:
                    snapshot_arrays["volumes/mask"] = mask
                logger.warning(
                    f"Saving Snapshot. Iteration: {iteration}, "
                    f"Loss: {loss.detach().cpu().numpy().item()}!"
                )
                for k, v in snapshot_arrays.items():
                    k = f"{iteration}/{k}"
                    snapshot_array_identifier = (
                        self.snapshot_container.array_identifier(k)
                    )
                    if k not in snapshot_zarr:
                        array = create_from_identifier(
                            snapshot_array_identifier,
                            v.axis_names,
                            v.roi,
                            (
                                v.shape[0]
                                if (v.channel_dims == 1 and v.shape[0] > 1)
                                else None
                            ),
                            v.voxel_size,
                            v.dtype if not v.dtype == bool else np.float32,
                            model.output_shape * v.voxel_size,
                            overwrite=True,
                        )
                    else:
                        array = open_from_identifier(
                            snapshot_array_identifier, mode="a"
                        )

                    # neuroglancer doesn't allow bools
                    if not v.dtype == bool:
                        data = v[:]
                    else:
                        data = v[:].astype(np.float32)

                    # remove channel dim if there is only 1 channel
                    if v.channel_dims == 1 and v.shape[0] == 1:
                        data = data[0]

                    array[:] = data

            logger.debug(
                f"Trainer step took {time.time() - t_start_prediction} seconds"
            )
            self.iteration += 1
            self.scheduler.step()
            yield TrainingIterationStats(
                loss=loss.item(),
                iteration=iteration,
                time=time.time() - t_start_prediction,
            )
            t_start_fetch = time.time()

    def __iter__(self):
        """
        Initializes the training pipeline.

        Returns:
            Iterator[None]: An iterator of None.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> for _ in trainer:
            >>>     pass
        """
        with gp.build(self._pipeline):
            teardown = False
            while not teardown:
                batch = self._pipeline.request_batch(self._request)
                yield batch
                teardown = yield
        yield None

    def next(self):
        """
        Fetches the next batch of data.

        Returns:
            Tuple[Array, Array, Array, Array, Array]: A tuple containing the raw data, ground truth data, target data, weight data, and mask data.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> raw, gt, target, weight, mask = trainer.next()

        """
        batch = next(self._iter)
        self._iter.send(False)
        return (
            gp_to_funlib_array(
                batch[self._raw_key],
            ),
            gp_to_funlib_array(batch[self._gt_key]),
            gp_to_funlib_array(batch[self._target_key]),
            gp_to_funlib_array(batch[self._weight_key]),
            (
                gp_to_funlib_array(batch[self._mask_key])
                if self._mask_key is not None
                else None
            ),
        )

    def __enter__(self):
        """
        Enters the context manager.

        Returns:
            GunpowderTrainer: The GunpowderTrainer object.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> with trainer:
            >>>     pass
        """
        self._iter = iter(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context manager.

        Args:
            exc_type: The exception type.
            exc_val: The exception value.
            exc_tb: The exception traceback.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> with trainer:
            >>>     pass
        """
        try:
            self._iter.send(True)
        except TypeError:
            self._iter.send(None)
        pass

    def can_train(self, datasets) -> bool:
        """
        Checks if the trainer can train with a specific set of datasets.

        Args:
            datasets (List[Dataset]): The list of datasets.
        Returns:
            bool: True if the trainer can train with the datasets, False otherwise.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> can_train = trainer.can_train(datasets)

        """
        return all([dataset.gt is not None for dataset in datasets])

    def visualize_pipeline(self, bind_address="0.0.0.0", bind_port=0):
        """
        Visualizes the pipeline for the run, including all produced arrays.

        Args:
            bind_address : str
                Bind address for Neuroglancer webserver
            bind_port : int
                Bind port for Neuroglancer webserver
        """

        if self._pipeline is None:
            raise ValueError("Pipeline not initialized!")

        import neuroglancer

        # self.iteration = 0

        pipeline = self._pipeline.children[0].children[0].copy()
        if self.num_data_fetchers > 1:
            pipeline = pipeline.children[0]

        pipeline += gp.Stack(1)

        request = self._request
        # raise Exception(request)

        def batch_generator():
            with gp.build(pipeline):
                while True:
                    yield pipeline.request_batch(request)

        batch_gen = batch_generator()

        def load_batch(event):
            print("fetching_batch")
            batch = next(batch_gen)

            with viewer.txn() as s:
                while len(s.layers) > 0:
                    del s.layers[0]

                # reverse order for raw so we can set opacity to 1, this
                # way higher res raw replaces low res when available
                for name, array in batch.arrays.items():
                    print(name)
                    data = array.data[0]

                    channel_dims = len(data.shape) - len(array.spec.voxel_size)
                    assert channel_dims <= 1

                    dims = neuroglancer.CoordinateSpace(
                        names=["c^", "z", "y", "x"][-len(data.shape) :],
                        units="nm",
                        scales=tuple([1] * channel_dims) + tuple(array.spec.voxel_size),
                    )

                    local_vol = neuroglancer.LocalVolume(
                        data=data,
                        voxel_offset=tuple([0] * channel_dims)
                        + tuple((-array.spec.roi.shape / 2) / array.spec.voxel_size),
                        dimensions=dims,
                    )

                    if name == self._gt_key:
                        s.layers[str(name)] = neuroglancer.SegmentationLayer(
                            source=local_vol
                        )
                    else:
                        s.layers[str(name)] = neuroglancer.ImageLayer(source=local_vol)

                s.layout = neuroglancer.row_layout(
                    [
                        neuroglancer.column_layout(
                            [
                                neuroglancer.LayerGroupViewer(
                                    layers=[str(k) for k, v in batch.items()]
                                ),
                            ]
                        )
                    ]
                )

        neuroglancer.set_server_bind_address(
            bind_address=bind_address, bind_port=bind_port
        )

        viewer = neuroglancer.Viewer()

        viewer.actions.add("load_batch", load_batch)

        with viewer.config_state.txn() as s:
            s.input_event_bindings.data_view["keyt"] = "load_batch"

        print(viewer)
        load_batch(None)

        input("Enter to quit!")

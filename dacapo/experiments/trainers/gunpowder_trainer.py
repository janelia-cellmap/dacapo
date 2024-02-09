from ..training_iteration_stats import TrainingIterationStats
from .trainer import Trainer

from dacapo.gp import (
    DaCapoArraySource,
    GraphSource,
    DaCapoTargetFilter,
    CopyMask,
    Product,
)
from dacapo.experiments.datasplits.datasets.arrays import (
    NumpyArray,
    ZarrArray,
    OnesArray,
)

from funlib.geometry import Coordinate
import gunpowder as gp

import zarr
import torch
import numpy as np

import time
import logging

logger = logging.getLogger(__name__)


class GunpowderTrainer(Trainer):
    iteration = 0

    def __init__(self, trainer_config):
        self.learning_rate = trainer_config.learning_rate
        self.batch_size = trainer_config.batch_size
        self.num_data_fetchers = trainer_config.num_data_fetchers
        self.print_profiling = 100
        self.snapshot_iteration = trainer_config.snapshot_interval
        self.min_masked = trainer_config.min_masked

        self.augments = trainer_config.augments
        self.mask_integral_downsample_factor = 4
        self.clip_raw = trainer_config.clip_raw

        # Testing out if calculating multiple times and multiplying is necessary
        self.add_predictor_nodes_to_dataset = trainer_config.add_predictor_nodes_to_dataset

        self.scheduler = None

    def create_optimizer(self, model):
        optimizer = torch.optim.RAdam(lr=self.learning_rate, params=model.parameters())
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=1000,
            last_epoch=-1,
        )
        return optimizer

    def build_batch_provider(self, datasets, model, task, snapshot_container=None):
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
        dataset_weight_key = gp.ArrayKey("DATASET_WEIGHT")
        datasets_weight_key = gp.ArrayKey("DATASETS_WEIGHT")
        weight_key = gp.ArrayKey("WEIGHT")
        sample_points_key = gp.GraphKey("SAMPLE_POINTS")

        # Get source nodes
        dataset_sources = []
        weights = []
        for dataset in datasets:
            weights.append(dataset.weight)
            assert isinstance(dataset.weight, int), dataset

            raw_source = DaCapoArraySource(dataset.raw, raw_key)
            if self.clip_raw:
                raw_source += gp.Crop(
                    raw_key, dataset.gt.roi.snap_to_grid(dataset.raw.voxel_size)
                )
            gt_source = DaCapoArraySource(dataset.gt, gt_key)
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
                mask_source = DaCapoArraySource(dataset.mask, mask_key)
            else:
                # Always provide a mask. By default it is simply an array
                # of ones with the same shape/roi as gt. Avoids making us
                # specially handle no mask case and allows padding of the
                # ground truth without worrying about training on incorrect
                # data.
                mask_source = DaCapoArraySource(OnesArray.like(dataset.gt), mask_key)
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
                + gp.Pad(raw_key, None, 0)
                + gp.Pad(gt_key, None, 0)
                + gp.Pad(mask_key, None, 0)
                + gp.RandomLocation(
                    ensure_nonempty=sample_points_key
                    if points_source is not None
                    else None,
                    ensure_centered=sample_points_key
                    if points_source is not None
                    else None,
                )
            )

            dataset_source += gp.Reject(mask_placeholder, 1e-6)

            for augment in self.augments:
                dataset_source += augment.node(raw_key, gt_key, mask_key)

            if self.add_predictor_nodes_to_dataset:
                # Add predictor nodes to dataset_source
                dataset_source += DaCapoTargetFilter(
                    task.predictor,
                    gt_key=gt_key,
                    weights_key=dataset_weight_key,
                    mask_key=mask_key,
                )

            dataset_sources.append(dataset_source)
        pipeline = tuple(dataset_sources) + gp.RandomProvider(weights)

        # Add predictor nodes to pipeline
        pipeline += DaCapoTargetFilter(
            task.predictor,
            gt_key=gt_key,
            target_key=target_key,
            weights_key=datasets_weight_key if self.add_predictor_nodes_to_dataset else weight_key,
            mask_key=mask_key,
        )

        if self.add_predictor_nodes_to_dataset:
            pipeline += Product(dataset_weight_key, datasets_weight_key, weight_key)

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
        t_start_fetch = time.time()

        logger.info("Starting iteration!")

        for iteration in range(self.iteration, self.iteration + num_iterations):
            raw, gt, target, weight, mask = self.next()
            logger.debug(
                f"Trainer fetch batch took {time.time() - t_start_fetch} seconds"
            )

            for param in model.parameters():
                param.grad = None

            t_start_prediction = time.time()
            predicted = model.forward(torch.as_tensor(raw[raw.roi]).to(device).float())
            predicted.retain_grad()
            loss = self._loss.compute(
                predicted,
                torch.as_tensor(target[target.roi]).to(device).float(),
                torch.as_tensor(weight[weight.roi]).to(device).float(),
            )
            loss.backward()
            optimizer.step()

            if (
                self.snapshot_iteration is not None
                and iteration % self.snapshot_iteration == 0
            ):
                snapshot_zarr = zarr.open(self.snapshot_container.container, "a")
                snapshot_arrays = {
                    "volumes/raw": raw,
                    "volumes/gt": gt,
                    "volumes/target": target,
                    "volumes/weight": weight,
                    "volumes/prediction": NumpyArray.from_np_array(
                        predicted.detach().cpu().numpy(),
                        target.roi,
                        target.voxel_size,
                        target.axes,
                    ),
                    "volumes/gradients": NumpyArray.from_np_array(
                        predicted.grad.detach().cpu().numpy(),
                        target.roi,
                        target.voxel_size,
                        target.axes,
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
                    if k not in snapshot_zarr:
                        snapshot_array_identifier = (
                            self.snapshot_container.array_identifier(k)
                        )
                        ZarrArray.create_from_array_identifier(
                            snapshot_array_identifier,
                            v.axes,
                            v.roi,
                            v.num_channels,
                            v.voxel_size,
                            v.dtype if not v.dtype == bool else np.float32,
                        )
                        dataset = snapshot_zarr[k]
                    else:
                        dataset = snapshot_zarr[k]
                    # remove batch dimension. Everything has a batch
                    # and channel dim because of torch.
                    if not v.dtype == bool:
                        data = v[v.roi][0]
                    else:
                        data = v[v.roi][0].astype(np.float32)
                    if v.num_channels is None:
                        # remove channel dimension
                        assert data.shape[0] == 1, (
                            f"Data for array {k} should not have channels but has shape: "
                            f"{v.shape}. The first dimension is channels"
                        )
                        data = data[0]
                    dataset[:] = data
                    dataset.attrs["offset"] = v.roi.offset
                    dataset.attrs["resolution"] = v.voxel_size
                    dataset.attrs["axes"] = v.axes

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
        with gp.build(self._pipeline):
            teardown = False
            while not teardown:
                batch = self._pipeline.request_batch(self._request)
                yield batch
                teardown = yield
        yield None

    def next(self):
        batch = next(self._iter)
        self._iter.send(False)
        return (
            NumpyArray.from_gp_array(batch[self._raw_key]),
            NumpyArray.from_gp_array(batch[self._gt_key]),
            NumpyArray.from_gp_array(batch[self._target_key]),
            NumpyArray.from_gp_array(batch[self._weight_key]),
            NumpyArray.from_gp_array(batch[self._mask_key])
            if self._mask_key is not None
            else None,
        )

    def __enter__(self):
        self._iter = iter(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._iter.send(True)
        pass

    def can_train(self, datasets) -> bool:
        return all([dataset.gt is not None for dataset in datasets])

from ..training_iteration_stats import TrainingIterationStats
from .trainer import Trainer

from dacapo.gp import (
    DaCapoArraySource,
    DaCapoTargetFilter,
    GammaAugment,
    ElasticAugment,
)
from dacapo.experiments.datasplits.datasets.arrays import NumpyArray, ZarrArray

from funlib.geometry import Coordinate
import gunpowder as gp

import zarr
import torch

import time
import logging

logger = logging.getLogger(__name__)


class GunpowderTrainer(Trainer):
    learning_rate = None
    batch_size = None
    iteration = 0

    def __init__(self, trainer_config):
        self.learning_rate = trainer_config.learning_rate
        self.batch_size = trainer_config.batch_size
        self.num_data_fetchers = trainer_config.num_data_fetchers
        self.print_profiling = 100

        self.simple_augment = trainer_config.simple_augment
        self.elastic_augment = trainer_config.elastic_augment
        self.intensity_augment = trainer_config.intensity_augment
        self.gamma_augment = trainer_config.gamma_augment
        self.intensity_scale_shift = trainer_config.intensity_scale_shift
        self.snapshot_iteration = trainer_config.snapshot_interval

    def create_optimizer(self, model):
        return torch.optim.Adam(lr=self.learning_rate, params=model.parameters())

    def build_batch_provider(self, datasets, model, task, snapshot_container=None):
        input_shape = Coordinate(model.input_shape)
        output_shape = Coordinate(model.output_shape)

        # define keys:
        raw_key = gp.ArrayKey("RAW")
        gt_key = gp.ArrayKey("GT")
        mask_key = gp.ArrayKey("MASK")

        target_key = gp.ArrayKey("TARGET")
        weight_key = gp.ArrayKey("WEIGHT")

        # Get source nodes
        dataset_sources = []
        raw_voxel_size = datasets[0].raw.voxel_size
        prediction_voxel_size = model.scale(raw_voxel_size)
        for dataset in datasets:

            raw_source = DaCapoArraySource(dataset.raw, raw_key)
            gt_source = DaCapoArraySource(dataset.gt, gt_key)
            array_sources = [raw_source, gt_source]
            if mask_key is not None and dataset.mask is not None:
                mask_source = DaCapoArraySource(dataset.mask, mask_key)
                array_sources.append(mask_source)
            else:
                # if any of the training datasets do not have a mask available,
                # we cannot use it during training
                mask_key = None

            dataset_source = (
                tuple(array_sources) + gp.MergeProvider() + gp.RandomLocation()
            )

            dataset_sources.append(dataset_source)
        pipeline = tuple(dataset_sources) + gp.RandomProvider()

        if self.simple_augment is not None:
            pipeline += gp.SimpleAugment(**self.simple_augment)
        if self.elastic_augment is not None:
            pipeline += ElasticAugment(**self.elastic_augment)
        if self.intensity_augment is not None:
            pipeline += gp.IntensityAugment(raw_key, **self.intensity_augment)
        if self.gamma_augment is not None:
            pipeline += GammaAugment(raw_key, **self.gamma_augment)
        if self.intensity_scale_shift is not None:
            pipeline += gp.IntensityScaleShift(raw_key, **self.intensity_scale_shift)

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

        # define input and output size:
        # switch to world units
        input_size = raw_voxel_size * input_shape
        output_size = prediction_voxel_size * output_shape

        # generate request for all necessary inputs to training
        request = gp.BatchRequest()
        request.add(raw_key, input_size)
        request.add(target_key, output_size)
        request.add(weight_key, output_size)
        # request additional keys for snapshots
        request.add(gt_key, output_size)
        if mask_key is not None:
            request.add(mask_key, output_size)

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
                logger.info("Saving Snapshot!")
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
                            v.dtype,
                        )
                        dataset = snapshot_zarr[k]
                    else:
                        dataset = snapshot_zarr[k]
                    # remove batch dimension. Everything has a batch
                    # and channel dim because of torch.
                    data = v[v.roi][0]
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

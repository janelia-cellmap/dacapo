from ..training_iteration_stats import TrainingIterationStats
from .trainer import Trainer

from dacapo.gp import DaCapoArraySource, DaCapoTargetFilter
from dacapo.experiments.datasplits.datasets.arrays import NumpyArray

from funlib.geometry import Coordinate
import gunpowder as gp

import torch


class GunpowderTrainer(Trainer):
    learning_rate = None
    batch_size = None
    iteration = 0

    def __init__(self, trainer_config):
        self.learning_rate = trainer_config.learning_rate
        self.batch_size = trainer_config.batch_size
        self.num_data_fetchers = trainer_config.num_data_fetchers
        self.augments = []
        self.print_profiling = 100

    def create_optimizer(self, model):
        return torch.optim.Adam(lr=self.learning_rate, params=model.parameters())

    def build_batch_provider(self, datasets, model, task):
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
            if dataset.mask is not None:
                mask_source = DaCapoArraySource(dataset.mask, mask_key)
                array_sources.append(mask_source)

            dataset_source = (
                tuple(array_sources) + gp.MergeProvider() + gp.RandomLocation()
            )

            if dataset.mask is not None:
                dataset_source += gp.Reject(mask_key, self.trainer.min_masked)

            dataset_sources.append(dataset_source)
        pipeline = tuple(dataset_sources) + gp.RandomProvider()

        for augmentation in self.augments:
            # TODO: Should each augmentation be output into a new key?
            # Could be helpful to show users the affects of applying an
            # augmentation with specific parameters
            # TODO: Can we remove the need for augmentations to provide a node?
            # Some augmentations can get quite involved (elastic augment)
            pipeline += self.get_augment_node(augmentation, raw_key)

        # Add predictor nodes to pipeline
        pipeline += DaCapoTargetFilter(
            task.predictor, gt_key=gt_key, target_key=target_key, weights_key=weight_key
        )

        # Trainer attributes:
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

        self._request = request
        self._pipeline = pipeline
        self._raw_key = raw_key
        self._gt_key = gt_key
        self._weight_key = weight_key
        self._target_key = target_key

    def iterate(self, num_iterations, model, optimizer):
        for self.iteration in range(self.iteration, self.iteration + num_iterations):
            raw, target, weight = self.next()

            for param in model.parameters():
                param.grad = None

            predicted = model.forward(torch.as_tensor(raw[raw.roi]))
            loss = self.loss(
                predicted,
                torch.as_tensor(target[target.roi]),
                torch.as_tensor(weight[weight.roi]),
            )
            loss.backward()
            optimizer.step()

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
            NumpyArray(batch[self._raw_key]),
            NumpyArray.from_gp_array(batch[self._target_key]),
            NumpyArray.from_gp_array(batch[self._weight_key]),
        )

    def __enter__(self):
        self._iter = iter(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._iter.send(True)
        pass

    def can_train(self, datasets) -> bool:
        return all([dataset.gt is not None for dataset in datasets])
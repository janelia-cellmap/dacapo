from dacapo.experiments.datasplits.datasets.arrays import Array, NumpyArray
from dacapo.gp import DaCapoArraySource, DaCapoTargetProvider

import gunpowder as gp
from funlib.geometry import Coordinate

from typing import Tuple, Dict


class GunpowderPipeline:
    def __init__(
        self, datasets, architecture, task, trainer, min_masked=0.1, print_profiling=100
    ):
        self.datasets = datasets
        self.architecture = architecture
        self.task = task
        self.trainer = trainer

        self.min_masked = min_masked
        self.print_profiling = print_profiling

        # define keys:
        self.raw_key = gp.ArrayKey("RAW")
        self.gt_key = gp.ArrayKey("GT")
        self.mask_key = gp.ArrayKey("MASK")
        self.predictor_keys = {
            predictor.name: (
                gp.ArrayKey(f"{predictor.name.upper()}_TARGET"),
                gp.ArrayKey(f"{predictor.name.upper()}_WEIGHT"),
            )
            for predictor in self.task.predictors
        }

        # internal variables
        self._pipeline, self._request = self._build_pipeline()

    def _build_pipeline(self):

        input_shape = self.architecture.input_shape
        output_shape = self.architecture.output_shape

        # Get source nodes
        dataset_sources = []
        raw_voxel_size = None
        prediction_voxel_size = None
        """
        upsampled_gt = gp.ArrayKey("UPSAMPLED_GT")
        upsampled_mask = gp.ArrayKey("UPSAMPLED_MASK")
        scaled_gt = gp.ArrayKey("SCALED_GT")
        scaled_mask = gp.ArrayKey("SCALED_MASK")
        """
        for dataset in self.datasets:

            raw_source = DaCapoArraySource(dataset.raw, self.raw_key)
            gt_source = DaCapoArraySource(dataset.gt, self.gt_key)
            if dataset.mask is not None:
                mask_source = DaCapoArraySource(dataset.mask, self.mask_key)
                dataset_sources = (raw_source, gt_source, mask_source)
            else:
                dataset_sources = (raw_source, gt_source)

            dataset_source = dataset_sources + gp.MergeProvider() + gp.RandomLocation()

            if dataset.mask is not None:
                dataset_source += gp.Reject(self.mask_key, self.min_masked)

            dataset_sources.append(dataset_source)
        pipeline = tuple(dataset_sources) + gp.RandomProvider()

        for augmentation in self.trainer.augments:
            # TODO: Should each augmentation be output into a new key?
            # Could be helpful to show users the affects of applying an
            # augmentation with specific parameters
            pipeline += augmentation.node(self.raw_key)

        # Add predictor nodes to pipeline
        for predictor in self.task.predictors:
            name = predictor.name
            predictor_target, predictor_weights = self.predictor_keys[name]
            pipeline += DaCapoTargetProvider(predictor, self.gt_key, predictor_target)
            # TODO: Handle weights/masks?

        # Trainer attributes:
        pipeline += gp.PreCache(num_workers=self.trainer.num_data_fetchers)

        # stack to create a batch dimension
        pipeline += gp.Stack(self.trainer.batch_size)

        # print profiling stats
        pipeline += gp.PrintProfilingStats(every=self.print_profiling)

        # define input and output size:
        # switch to world units
        input_size = raw_voxel_size * input_shape
        output_size = prediction_voxel_size * output_shape

        # generate request for all necessary inputs to training
        request = gp.BatchRequest()
        request.add(self.raw_key, input_size)
        request.add(self.gt_key, output_size)
        for predictor in self.task.predictors:
            name = predictor.name
            predictor_target, predictor_weight = self.predictor_keys[name]
            request.add(predictor_target, output_size)
            if predictor_weight is not None:
                request.add(predictor_weight, output_size)

        return pipeline, request

    def __iter__(self):
        with gp.build(self.pipeline):
            while True:
                batch = self.pipeline.request_batch(self.request)
                raw = NumpyArray.from_gp_array(batch[self.raw_key])
                gt = NumpyArray.from_gp_array(batch[self.gt_key])
                targets = {
                    name: NumpyArray.from_gp_array(batch[target_key])
                    for name, (target_key, _) in self._predictor_keys.items()
                }
                weights = {
                    name: NumpyArray.from_gp_array(batch[weight_key])
                    for name, (_, weight_key) in self._predictor_keys.items()
                    if weight_key is not None
                }
                yield (
                    raw,
                    gt,
                    targets,
                    weights,
                )

    def next(self) -> Tuple[Array, Array, Dict[str, Array], Dict[str, Array]]:
        return next(self._iterator)

import attr
import random
import logging

from .gp_augments import AugmentConfig
from .trainer_config import TrainerConfig
from dacapo.gp import GraphSource, CopyMask, DaCapoTargetFilter

from typing import Optional, List
import gunpowder as gp
import numpy as np

from funlib.geometry import Coordinate
from funlib.persistence import Array
from dacapo.experiments.datasplits.datasets import Dataset
from dacapo.experiments.tasks.predictors import Predictor

import torch

logger = logging.getLogger(__name__)


def pipeline_generator(
    pipeline: gp.Pipeline,
    request: gp.BatchRequest,
    raw_key: gp.ArrayKey,
    gt_key: gp.ArrayKey,
    mask_key: gp.ArrayKey,
    target_key: gp.ArrayKey,
    weight_key: gp.ArrayKey,
    predictor: Predictor | None = None,
):
    while True:
        with gp.build(pipeline) as pipeline:
            batch_request = request.copy()
            batch_request._random_seed = random.randint(0, 2**32 - 1)
            batch = pipeline.request_batch(batch_request)
            yield (
                {
                    "raw": torch.from_numpy(batch[raw_key].data),
                    "gt": torch.from_numpy(batch[gt_key].data),
                    "mask": torch.from_numpy(batch[mask_key].data),
                    **(
                        {
                            "target": torch.from_numpy(batch[target_key].data),
                            "weight": torch.from_numpy(batch[weight_key].data),
                        }
                        if predictor is not None
                        else {}
                    ),
                }
            )


class GeneratorDataset(torch.utils.data.IterableDataset):
    """
    Helper class to return a torch IterableDataset from a generator
    """

    def __init__(self, generator, *args, **kwargs):
        self.generator = generator
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self.generator(*self.args, **self.kwargs)


@attr.s
class GunpowderTrainerConfig(TrainerConfig):
    """
    This class is used to configure a Gunpowder Trainer. It contains attributes related to
    augmentations to apply and sampling strategies to use during training.
    """

    augments: List[AugmentConfig] = attr.ib(
        factory=lambda: list(),
        metadata={"help_text": "The augments to apply during training."},
    )
    min_masked: Optional[float] = attr.ib(default=0.15)
    clip_raw: bool = attr.ib(default=False)

    sample_strategy: str | None = attr.ib(
        default=None,
        metadata={
            "help_text": "The strategy to use for sampling (default None). Options are 'integral_mask', "
            "'reject', and 'sample_points'. 'integral_mask' will sample every possible crop "
            "as long as the center voxel is masked in. 'reject' will continuously sample "
            "until a crop is found where the center voxel is masked in. `sample_points` will "
            "simply sample randomly from a list of center points.\n"
            "'integral_mask': Fast, but may use a  lot of memory.\n"
            "'reject': Slow, but uses less memory.\n"
            "'sample_points': Fast, but requires a list of sample points.\n"
            "`None`: Randomly sample anywhere as long as the center voxel is contained in the mask."
        },
    )

    def iterable_dataset(
        self,
        datasets: List[Dataset],
        input_shape: Coordinate,
        output_shape: Coordinate,
        predictor: Predictor | None = None,
    ) -> torch.utils.data.IterableDataset:
        """
        Returns an pytorch compatible IterableDataset.
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for more info
        """
        assert len(datasets) >= 1, "Expected at least one dataset, got an empty list"

        # get voxel sizes
        raw_voxel_size = datasets[0].raw.voxel_size
        target_voxel_size = datasets[0].gt.voxel_size

        # define input and output size:
        # switch to world units
        input_size = raw_voxel_size * input_shape
        output_size = target_voxel_size * output_shape

        # define keys:
        raw_key = gp.ArrayKey("RAW")
        gt_key = gp.ArrayKey("GT")
        mask_key = gp.ArrayKey("MASK")

        # Make requests such that the mask placeholder is not empty.
        # We request a single pixel from the placeholder.
        # This means we can pad gt and mask as much as we want and not worry retrieving
        # empty masks.
        # as long as the gt is large enough to accommodate one voxel we shouldn't have
        # empty samples.
        mask_placeholder = gp.ArrayKey("MASK_PLACEHOLDER")

        # generated from the task if given
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
            if dataset.raw.channel_dims == 0:
                raw_source += gp.Unsqueeze([raw_key], axis=0)
            if self.clip_raw:
                raw_source += gp.Crop(
                    raw_key, dataset.gt.roi.snap_to_grid(dataset.raw.voxel_size)
                )
            gt_source = gp.ArraySource(gt_key, dataset.gt)

            points_source = None
            if self.sample_strategy == "sample_points":
                sample_points = dataset.sample_points
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

            dataset_source = tuple(array_sources) + gp.MergeProvider()
            dataset_source += CopyMask(
                mask_key,
                mask_placeholder,
                drop_channels=True,
            )

            # Pad the data we care about infinitely
            dataset_source += gp.Pad(raw_key, None)
            dataset_source += gp.Pad(gt_key, None)
            dataset_source += gp.Pad(mask_key, None)

            if self.sample_strategy == "sample_points":
                dataset_source += gp.RandomLocation(
                    ensure_nonempty=sample_points_key,
                    ensure_centered=sample_points_key,
                )
            elif self.sample_strategy == "integral_mask":
                dataset_source += gp.RandomLocation(
                    min_masked=1,
                    mask=mask_placeholder,
                )
            elif self.sample_strategy == "reject":
                dataset_source += gp.RandomLocation()
                dataset_source += gp.Reject(mask_placeholder, 1.0)
            elif self.sample_strategy is None:
                dataset_source += gp.RandomLocation()

            for augment in self.augments:
                dataset_source += augment.node(raw_key, gt_key, mask_key)

            dataset_sources.append(dataset_source)
        pipeline = tuple(dataset_sources) + gp.RandomProvider(weights)

        if predictor is not None:
            # Add predictor nodes to pipeline
            pipeline += DaCapoTargetFilter(
                predictor,
                gt_key=gt_key,
                target_key=target_key,
                weights_key=weight_key,
                mask_key=mask_key,
            )

        # generate request for all necessary inputs to training
        request = gp.BatchRequest()
        request.add(raw_key, input_size)

        # request additional keys for snapshots
        request.add(gt_key, output_size)
        request.add(mask_key, output_size)

        # Add keys necessary to train this task
        if predictor is not None:
            request.add(target_key, output_size)
            request.add(weight_key, output_size)

        # Add mask placeholder to guarantee center voxel is contained in
        # the mask, and to be used for some sampling strategies.
        request.add(
            mask_placeholder,
            target_voxel_size,
        )
        request.array_specs[mask_placeholder].roi = request.array_specs[
            mask_placeholder
        ].roi.snap_to_grid(target_voxel_size)

        # Function to generate batches
        # This allows us to pass the gunpowder pipeline into a torch `IterableDataset`
        # We also use random seeds to ensure that each worker is fetching different
        # data.

        return GeneratorDataset(
            pipeline_generator,
            pipeline,
            request,
            raw_key,
            gt_key,
            mask_key,
            target_key,
            weight_key,
            predictor,
        )

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
            logger.info("fetching_batch")
            batch = next(batch_gen)

            with viewer.txn() as s:
                while len(s.layers) > 0:
                    del s.layers[0]

                # reverse order for raw so we can set opacity to 1, this
                # way higher res raw replaces low res when available
                for name, array in batch.arrays.items():
                    logger.info(name)
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

        logger.info(viewer)
        load_batch(None)

        input("Enter to quit!")



# ## Example
# config = GunpowderTrainerConfig(name="gunpowder")

# data = SimpleDataSplitConfig(path="my_data.zarr", name="sample_data")

# pytorch_dataset = config.iterable_dataset(data.train, input_shape, output_shape)


# # desired
# pytorch_dataset = dacapo.VolumeDataset(
#     "my_data.zarr", "sample_data", input_shape, output_shape
# )

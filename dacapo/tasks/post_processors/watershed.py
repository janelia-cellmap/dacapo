import attr

from .steps import Fragment, Agglomerate, CreateLUTS, Segment
from .steps.waterz_merge_functions import MergeFunction
from .post_processor_abc import PostProcessorABC

from pathlib import Path
from typing import List
import logging


logger = logging.getLogger(__name__)


@attr.s
class Watershed(PostProcessorABC):
    output: str = attr.ib(default="instance_ids")

    # parameter ranges to explore:
    # creating fragments
    filter_fragments: List[float] = attr.ib(factory=lambda: list([0]))
    fragments_in_xy: List[bool] = attr.ib(factory=lambda: list([False]))
    epsilon_agglomerate: List[float] = attr.ib(factory=lambda: list([0]))
    min_seed_distance: List[float] = attr.ib(factory=lambda: list([1]))
    compactness: List[float] = attr.ib(factory=lambda: list([0]))
    # agglomerating fragments
    merge_function: List[MergeFunction] = attr.ib(
        factory=lambda: list([MergeFunction.MEAN])
    )
    # create_luts
    threshold: List[float] = attr.ib(factory=lambda: list([0.5]))

    def tasks(
        self,
        input_id: str,
        input_zarr: Path,
        affs_dataset: str,
    ):
        """
        input_id should be the unique id of the predictions you are post processing.
        i.e. run.id + iteration or prediction.id if run during prediction.
        This will mostly be used to store block processing statuses in mongodb

        common inputs:
            input_id
            store
            output_file
            affs_dataset
            mask_file
            mask_dataset

            # step inputs that we can generate
            fragments_dataset
            lookup
            roi

        output:
            segmentation_dataset

        """

        # what about read/write block sizes? different for fragment/agglomerate/segment?
        # what about num_workers per step? different for fragment/agglomerate/segment?
        # what about input/output rois? input/output_roi = affs.roi

        fragments_dataset = f"{affs_dataset}_fragments"
        segmentation_dataset = f"{affs_dataset}_ids"
        lookup = f"{input_id}_watershed_lut"
        # create a seperate dataset: f"{fragments_dataset}_{i}" for each parameter group
        # store parameters as attributes on the zarr

        fragment_tasks, parameters = Fragment(
            filter_fragments=self.filter_fragments,
            fragments_in_xy=self.fragments_in_xy,
            epsilon_agglomerate=self.epsilon_agglomerate,
            min_seed_distance=self.min_seed_distance,
            compactness=self.compactness,
        ).tasks(
            input_id=input_id,
            input_zarr=input_zarr,
            affs_dataset=affs_dataset,
            fragments_dataset=fragments_dataset,
            mask_file=None,
            mask_dataset=None,
        )
        agglomerate_tasks, parameters = Agglomerate(
            merge_function=self.merge_function,
        ).tasks(
            input_id=input_id,
            input_zarr=input_zarr,
            affs_dataset=affs_dataset,
            fragments_dataset=fragments_dataset,
            upstream_tasks=(fragment_tasks, parameters),
        )
        create_luts_tasks, parameters = CreateLUTS(threshold=self.threshold).tasks(
            input_id=input_id,
            lookup=lookup,
            upstream_task=(agglomerate_tasks, parameters),
        )
        segment_tasks, parameters = Segment().tasks(
            input_id=input_id,
            input_zarr=input_zarr,
            fragments_dataset=fragments_dataset,
            segmentation_dataset=segmentation_dataset,
            lookup=lookup,
            upstream_task=(create_luts_tasks, parameters),
        )

        return segment_tasks, parameters
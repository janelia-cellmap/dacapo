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
    filter_fragments: List[float] = attr.ib(
        factory=lambda: list([0]),
        metadata={
            "help_text": "A float between 0 and 1 determining a threshold for ignoring "
            "fragments with an average affinity below this threshold."
        },
    )
    fragments_in_xy: List[bool] = attr.ib(
        factory=lambda: list([False]),
        metadata={
            "help_text": "Whether to generate fragments in xy only or to include the z dimension."
        },
    )
    epsilon_agglomerate: List[float] = attr.ib(
        factory=lambda: list([0]),
        metadata={
            "help_text": "Whether to automatically merge fragments with scores below epsilon. "
            "Can help reduce the number of fragments and keep the agglomeration graph small."
        },
    )
    min_seed_distance: List[float] = attr.ib(
        factory=lambda: list([1]),
        metadata={"help_text": "The minimum distance between seeds in watershed."},
    )
    compactness: List[float] = attr.ib(
        factory=lambda: list([0]),
        metadata={"help_text": "The compactness of watershed fragments."},
    )
    # agglomerating fragments
    merge_function: List[MergeFunction] = attr.ib(
        factory=lambda: list([MergeFunction.MEAN]),
        metadata={"help_text": "The waterz merge function to use for agglomeration."},
    )
    # create_luts
    threshold: List[float] = attr.ib(
        factory=lambda: list([0.5]),
        metadata={"help_text": "The threshold to use for merging."},
    )

    def tasks(
        self,
        pred_id: str,
        container,
        input_dataset,
        output_dataset,
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

        fragments_dataset = f"{input_dataset}_fragments"
        segmentation_dataset = output_dataset
        lookup = f"{pred_id}_watershed_lut"
        # create a seperate dataset: f"{fragments_dataset}_{i}" for each parameter group
        # store parameters as attributes on the zarr

        fragment_tasks, parameters = Fragment(
            filter_fragments=self.filter_fragments,
            fragments_in_xy=self.fragments_in_xy,
            epsilon_agglomerate=self.epsilon_agglomerate,
            min_seed_distance=self.min_seed_distance,
            compactness=self.compactness,
        ).tasks(
            input_id=pred_id,
            input_zarr=container,
            affs_dataset=input_dataset,
            fragments_dataset=fragments_dataset,
            mask_file=None,
            mask_dataset=None,
        )
        agglomerate_tasks, parameters = Agglomerate(
            merge_function=self.merge_function,
        ).tasks(
            input_id=pred_id,
            input_zarr=container,
            affs_dataset=input_dataset,
            fragments_dataset=fragments_dataset,
            upstream_tasks=(fragment_tasks, parameters),
        )
        create_luts_tasks, parameters = CreateLUTS(threshold=self.threshold).tasks(
            input_id=pred_id,
            lookup=lookup,
            upstream_task=(agglomerate_tasks, parameters),
        )
        segment_tasks, parameters = Segment().tasks(
            input_id=pred_id,
            input_zarr=container,
            fragments_dataset=fragments_dataset,
            segmentation_dataset=segmentation_dataset,
            lookup=lookup,
            upstream_task=(create_luts_tasks, parameters),
        )

        return segment_tasks, parameters
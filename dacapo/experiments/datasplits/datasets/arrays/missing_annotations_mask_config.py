import attr

from .array_config import ArrayConfig

from typing import List, Tuple
from funlib.persistence import Array
from fibsem_tools.metadata.groundtruth import LabelList

import dask.array as da


@attr.s
class MissingAnnotationsMaskConfig(ArrayConfig):
    """
    This config class provides the necessary configuration for turning an Annotated dataset into a
    multi class binary classification problem

    Attributes:
        source_array_config : ArrayConfig
            The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension
        groupings : List[Tuple[str, List[int]]]
            List of id groups with a symantic name. Each id group is a List of ids.
            Group i found in groupings[i] will be binarized and placed in channel i.
    Note:
        The output array will have a channel dimension equal to the number of groups.
        Each channel will be a binary mask of the ids in the groupings list.
    """

    source_array_config: ArrayConfig = attr.ib(
        metadata={
            "help_text": "The Array from which to pull annotated data. Is expected to contain a volume with uint64 voxels and no channel dimension"
        }
    )

    groupings: List[Tuple[str, List[int]]] = attr.ib(
        metadata={
            "help_text": "List of id groups with a symantic name. Each id group is a List of ids. "
            "Group i found in groupings[i] will be binarized and placed in channel i."
        }
    )

    def array(self, mode: str = "r") -> Array:
        labels = self.source_array_config.array(mode)
        grouped = da.ones((len(self.groupings), *labels.shape), dtype=bool)
        grouped[:] = labels.data > 0
        labels_list = LabelList.parse_obj(
            {"labels": labels._source_data.attrs["labels"]}
        ).labels
        present_not_annotated = set(
            [
                label.value
                for label in labels_list
                if label.annotationState.present and not label.annotationState.annotated
            ]
        )
        for i, (_, ids) in enumerate(self.groupings):
            if any([id in present_not_annotated for id in ids]):
                grouped[i] = 0

        return Array(
            grouped, labels.offset, labels.voxel_size, labels.axis_names, labels.units
        )

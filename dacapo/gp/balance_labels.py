import logging
import copy

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
from scipy.ndimage import generate_binary_structure

from gunpowder.array import Array
from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.batch import Batch
from gunpowder.batch_request import BatchRequest
from gunpowder.array_spec import ArraySpec

logger = logging.getLogger(__name__)


class AddDistance(BatchFilter):
    """Compute array with signed distances from specific labels
    Args:
        label_array_key(:class:``ArrayKey``):

            The :class:``ArrayKey`` to read the labels from.

        distance_array_key(:class:``ArrayKey``):

            The :class:``ArrayKey`` to generate containing the values of the
            distance transform.

        mask_array_key(:class:``ArrayKey``, optional):

            The :class:``ArrayKey`` to update or provide in order to compensate
            for windowing artifacts after distance transformation.

        fg_values (int, tuple, optional):

            ids from which to compute distance transform on foreground objects
            (defaults to None). fg_values can be given as None, an int, or an
            iterable of int, None, or Iterable of int.

            Consider a volume with the following labels:
            0: intracellular space
            1: neuron
            2: neuron membrane
            3: mito
            4: vesicles
            Assuming bg_values = 0 there are a couple options for fg_values.

            If fg_values = None. A binary mask will be made where data != 0
            and distances will be calculated only to the nearest intracellular
            space.

            If fg_values = 1. Error will be thrown since it is not clear what
            to do with 2,3,4. If fg_values = 1, and bg_values = None. Then
            0,2,3,4 will all be considered a single background object while
            calculating distances.

            If fg_values = [1, 4, None]. 3 fg distance transforms will be calculated.
            Each with binary mask data == 1, data == 4, data != (0, 1, 4).

            If fg_values = [[1,2],None]. 2 fg distance transforms will be calculated.
            Each with binary mask data == (1, 2), data != (0, 1, 2)


        bg_values (int, tuple, optional):

            ids from which to compute distance transform as background

        max_distance(scalar, tuple, optional):

            maximal distance that computed distances will be clipped to. For a
            single value this is the absolute value of the minimal and maximal distance.
            A tuple should be given as (minimal_distance, maximal_distance)
    """

    def __init__(
        self,
        label_array_key,
        distance_array_key,
        mask_array_key=None,
        label_id=None,
        bg_value=0,
        max_distance=None,
    ):

        self.label_array_key = label_array_key
        self.distance_array_key = distance_array_key
        self.mask_array_key = mask_array_key

        self.all_labels = set()
        self.rest = None
        self.fg_objects = []
        self.bg_objects = []
        if label_id is None:
            assert self.rest is None
            self.rest = "fg"
        elif isinstance(label_id, int):
            self.all_labels.add(label_id)
            self.fg_objects.append([label_id])
        else:
            for object_labels in label_id:
                if object_labels is None:
                    assert self.rest is None
                    self.rest = "fg"
                elif isinstance(object_labels, int):
                    self.all_labels.add(object_labels)
                    self.fg_objects.append([object_labels])
                else:
                    group = []
                    for object_label in object_labels:
                        assert isinstance(object_label, int)
                        self.all_labels.add(object_label)
                        group.append(object_label)
                    self.fg_objects.append(group)

        if bg_value is None:
            assert self.rest is None
            self.rest = "bg"
        elif isinstance(bg_value, int):
            self.all_labels.add(bg_value)
            self.bg_objects.append([bg_value])
        else:
            for object_labels in bg_value:
                if object_labels is None:
                    assert self.rest is None
                    self.rest = "bg"
                elif isinstance(object_labels, int):
                    self.all_labels.add(object_labels)
                    self.bg_objects.append([object_labels])
                else:
                    group = []
                    for object_label in object_labels:
                        assert isinstance(object_label, int)
                        self.all_labels.add(object_label)
                        group.append(object_label)
                    self.bg_objects.append(group)

        self.max_distance = max_distance

    def setup(self):

        spec = self.spec[self.label_array_key].copy()
        spec.dtype = np.float32
        spec.interpolatable = True
        self.provides(self.distance_array_key, spec)
        if self.mask_array_key is not None:
            mask_spec = spec.copy()
            mask_spec.dtype = np.bool
            mask_spec.interpolatable = False
            if self.mask_array_key in self.spec:
                self.updates(self.mask_array_key, mask_spec)
            else:
                self.provides(self.mask_array_key, mask_spec)
        self.enable_autoskip()

    def prepare(self, request):

        deps = BatchRequest()
        request_roi = request[self.distance_array_key].roi
        if self.mask_array_key is not None:
            if self.mask_array_key in self.spec:
                deps[self.mask_array_key] = ArraySpec(roi=request_roi)
        deps[self.label_array_key] = ArraySpec(roi=request_roi)
        return deps

    def process(self, batch, request):

        voxel_size = self.spec[self.label_array_key].voxel_size
        data = batch.arrays[self.label_array_key].data

        # get mask data. Let the mask be optional.
        if self.mask_array_key is not None and self.mask_array_key in self.spec:
            mask = batch.arrays[self.mask_array_key].data
        elif self.mask_array_key is not None:
            mask = np.ones(data.shape, dtype=np.bool)
        logging.debug("labels contained in batch {0:}".format(np.unique(data)))

        sampling = tuple(float(v) for v in voxel_size)
        if self.rest is not None:
            unique_values = np.unique(data)
            remaining_labels = []
            for v in unique_values:
                if v not in self.all_labels:
                    remaining_labels.append(v)
            fg_objects = copy.deepcopy(self.fg_objects)
            bg_objects = copy.deepcopy(self.bg_objects)
            if len(remaining_labels) > 0:
                if self.rest == "fg":
                    fg_objects.append(remaining_labels)
                if self.rest == "bg":
                    bg_objects.append(remaining_labels)
        else:
            fg_objects = self.fg_objects
            bg_objects = self.bg_objects

        distances = self.__signed_distance(
            data, fg_objects, bg_objects, sampling=sampling
        )
        spec = self.spec[self.distance_array_key].copy()
        spec.roi = request[self.distance_array_key].roi

        outputs = Batch()

        outputs.arrays[self.distance_array_key] = Array(distances, spec)

        if self.mask_array_key is not None:
            mask_voxel_size = tuple(
                float(v) for v in self.spec[self.mask_array_key].voxel_size
            )
            mask = self.__constrain_distances(mask, distances, mask_voxel_size)
            outputs.arrays[self.mask_array_key] = Array(mask, spec)

    @staticmethod
    def __signed_distance(label_array, foreground_ids, background_ids, **kwargs):
        # calculate signed distance transform relative to a binary label. Positive distance
        # inside the object, negative distance outside the object. This function estimates
        # signed distance by taking the difference between the distance transform of the
        # label ("inner distances") and the distance transform of the complement of the label
        # ("outer distances"). To compensate for an edge effect, .5 (half a pixel's distance)
        # is added to the positive distances and subtracted from the negative distances.
        constant_label = label_array.std() == 0
        dims = label_array.ndim

        if constant_label:
            if label_array.max == 0:
                return np.zeros(label_array.shape) - 1
            else:
                tmp = np.pad(label_array, 1)
                distances = distance_transform_edt(
                    binary_erosion(
                        tmp,
                        border_value=1,
                        structure=generate_binary_structure(dims, dims),
                    ),
                    **kwargs,
                )
                return distances[(slice(1, -1),) * distances.ndim]

        else:
            dims = label_array.ndim
            inner_distances = []
            for foreground_obj in foreground_ids:
                inner_distances.append(
                    distance_transform_edt(
                        binary_erosion(
                            np.isin(label_array, foreground_obj),
                            border_value=1,
                            structure=generate_binary_structure(dims, dims),
                        ),
                        **kwargs,
                    )
                )
            outer_distances = []
            for background_obj in background_ids:
                outer_distances.append(
                    distance_transform_edt(
                        np.isin(label_array, background_obj),
                        **kwargs,
                    )
                )
            return np.sum(inner_distances, axis=0) - np.sum(outer_distances, axis=0)

    def __constrain_distances(self, mask, distances, mask_sampling):
        # remove elements from the mask where the label distances exceed the
        # distance from the boundary

        tmp = np.pad(mask, 1)
        slices = tmp.ndim * (slice(1, -1),)
        boundary_distance = distance_transform_edt(
            binary_erosion(
                tmp,
                border_value=1,
                structure=generate_binary_structure(tmp.ndim, tmp.ndim),
            ),
            sampling=mask_sampling,
        )
        boundary_distance = boundary_distance[slices]
        if self.max_distance is not None:
            if self.add_constant is None:
                add = 0
            else:
                add = self.add_constant
            boundary_distance = self.__clip_distance(
                boundary_distance, (-self.max_distance - add, self.max_distance - add)
            )

        mask_output = mask.copy()
        if self.max_distance is not None:
            logging.debug(
                "Total number of masked in voxels before distance masking {0:}".format(
                    np.sum(mask_output)
                )
            )
            mask_output[
                (abs(distances) >= boundary_distance)
                * (distances >= 0)
                * (boundary_distance < self.max_distance - add)
            ] = 0
            logging.debug(
                "Total number of masked in voxels after postive distance masking {0:}".format(
                    np.sum(mask_output)
                )
            )
            mask_output[
                (abs(distances) >= boundary_distance + 1)
                * (distances < 0)
                * (boundary_distance + 1 < self.max_distance - add)
            ] = 0
            logging.debug(
                "Total number of masked in voxels after negative distance masking {0:}".format(
                    np.sum(mask_output)
                )
            )
        else:
            logging.debug(
                "Total number of masked in voxels before distance masking {0:}".format(
                    np.sum(mask_output)
                )
            )
            mask_output[
                np.logical_and(abs(distances) >= boundary_distance, distances >= 0)
            ] = 0
            logging.debug(
                "Total number of masked in voxels after postive distance masking {0:}".format(
                    np.sum(mask_output)
                )
            )
            mask_output[
                np.logical_and(abs(distances) >= boundary_distance + 1, distances < 0)
            ] = 0
            logging.debug(
                "Total number of masked in voxels after negative distance masking {0:}".format(
                    np.sum(mask_output)
                )
            )
        return mask_output
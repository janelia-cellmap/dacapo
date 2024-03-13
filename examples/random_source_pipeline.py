import gunpowder as gp
import logging
import numpy as np
import random
import torch
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
)
from skimage.measure import label as relabel

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True


def calc_max_padding(output_size, voxel_size, sigma, mode="shrink"):

    method_padding = gp.Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]]) + method_padding),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


class CreatePoints(gp.BatchFilter):
    def __init__(
        self,
        labels,
    ):

        self.labels = labels

    def process(self, batch, request):

        labels = batch[self.labels].data

        num_points = random.randint(20, 150)

        for n in range(num_points):
            z = random.randint(1, labels.shape[0] - 1)
            y = random.randint(1, labels.shape[1] - 1)
            x = random.randint(1, labels.shape[2] - 1)

            labels[z, y, x] = 1

        batch[self.labels].data = labels


class DilatePoints(gp.BatchFilter):
    def __init__(self, labels, dilations=2):

        self.labels = labels

    def process(self, batch, request):

        labels = batch[self.labels].data

        struct = generate_binary_structure(2, 2)

        dilations = random.randint(2, 8)

        for z in range(labels.shape[0]):

            dilated = binary_dilation(labels[z], structure=struct, iterations=dilations)

            labels[z] = dilated.astype(labels.dtype)

        batch[self.labels].data = labels


class Relabel(gp.BatchFilter):
    def __init__(self, labels):

        self.labels = labels

    def process(self, batch, request):

        labels = batch[self.labels].data

        relabeled = relabel(labels, connectivity=1).astype(labels.dtype)  # type: ignore

        batch[self.labels].data = relabeled


class ExpandLabels(gp.BatchFilter):
    def __init__(self, labels, background=0):
        self.labels = labels
        self.background = background

    def process(self, batch, request):

        labels_data = batch[self.labels].data
        distance = labels_data.shape[0]

        distances, indices = distance_transform_edt(
            labels_data == self.background, return_indices=True
        )  # type: ignore

        expanded_labels = np.zeros_like(labels_data)

        dilate_mask = distances <= distance

        masked_indices = [
            dimension_indices[dilate_mask] for dimension_indices in indices
        ]

        nearest_labels = labels_data[tuple(masked_indices)]

        expanded_labels[dilate_mask] = nearest_labels

        batch[self.labels].data = expanded_labels


class ChangeBackground(gp.BatchFilter):
    def __init__(self, labels):

        self.labels = labels

    def process(self, batch, request):

        labels = batch[self.labels].data

        labels[labels == 0] = np.max(labels) + 1

        batch[self.labels].data = labels


class ZerosSource(gp.BatchProvider):
    def __init__(self, key, spec):
        self.key = key
        self._spec = {key: spec}

    def setup(self):
        pass

    def provide(self, request):
        batch = gp.Batch()

        roi = request[self.key].roi
        shape = (roi / self._spec[self.key].voxel_size).get_shape()
        spec = self._spec[self.key].copy()
        spec.roi = roi

        batch.arrays[self.key] = gp.Array(np.zeros(shape), spec)

        return batch


def random_source_pipeline():
    """Create a random source pipeline and batch request for example training.


    Returns:
        gunpowder.Pipeline: The batch generating Gunpowder pipeline.
        gunpowder.BatchRequest: The batch request for the pipeline.
    """

    labels = gp.ArrayKey("LABELS")

    voxel_size = gp.Coordinate((8, 8, 8))

    input_shape = gp.Coordinate((148, 148, 148))

    input_size = input_shape * voxel_size

    request = gp.BatchRequest()

    request.add(labels, input_size)

    source_spec = gp.ArraySpec(roi=gp.Roi((0, 0, 0), input_size), voxel_size=voxel_size)
    source = ZerosSource(labels, source_spec)

    pipeline = source

    # randomly sample some points and write them into our zeros array as ones
    pipeline += CreatePoints(labels)

    # grow the boundaries
    pipeline += DilatePoints(labels)

    # relabel connected components
    pipeline += Relabel(labels)

    # expand the labels outwards into the background
    pipeline += ExpandLabels(labels)

    # there will still be some background, change this to max id + 1
    pipeline += ChangeBackground(labels)

    # relabel ccs again to deal with incorrectly connected background
    pipeline += Relabel(labels)

    return pipeline, request

from typing import Iterable
import gunpowder as gp
import logging
import numpy as np
import random
import torch
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
    gaussian_filter,
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
        num_points=(20, 150),
    ):

        self.labels = labels
        self.num_points = num_points

    def process(self, batch, request):

        labels = batch[self.labels].data

        num_points = random.randint(*self.num_points)

        for n in range(num_points):
            z = random.randint(1, labels.shape[0] - 1)
            y = random.randint(1, labels.shape[1] - 1)
            x = random.randint(1, labels.shape[2] - 1)

            labels[z, y, x] = 1

        batch[self.labels].data = labels


class MakeRaw(gp.BatchFilter):
    def __init__(
        self,
        raw,
        labels,
        gaussian_noise_args: Iterable = (0, 0.1),
        gaussian_blur_args: Iterable = (0.5, 1.5),
        membrane_like=True,
        membrane_size=3,
        inside_value=0.5,
    ):
        self.raw = raw
        self.labels = labels
        self.gaussian_noise_args = gaussian_noise_args
        self.gaussian_blur_args = gaussian_blur_args
        self.membrane_like = membrane_like
        self.membrane_size = membrane_size
        self.inside_value = inside_value

    def setup(self):
        spec = self.spec[self.labels].copy()
        spec.dtype = np.float32
        self.provides(self.raw, spec)

    def process(self, batch, request):
        labels = batch[self.labels].data
        raw = np.zeros_like(labels, dtype=np.float32)
        raw[labels > 0] = 1

        # generate membrane-like structure
        if self.membrane_like:
            distance = distance_transform_edt(raw)
            inside_mask = distance > self.membrane_size  # type: ignore
            raw[inside_mask] = self.inside_value

        # now add blur
        raw = gaussian_filter(raw, random.uniform(*self.gaussian_blur_args))

        # now add noise
        raw += np.random.normal(*self.gaussian_noise_args, raw.shape)  # type: ignore

        # normalize to [0, 1]
        raw -= raw.min()
        raw /= raw.max()

        batch[self.raw].data = raw


class DilatePoints(gp.BatchFilter):
    def __init__(self, labels, dilations=[2, 8], connectivity=2):

        self.labels = labels
        self.dilations = dilations
        self.connectivity = connectivity

    def process(self, batch, request):

        labels = batch[self.labels].data

        struct = generate_binary_structure(labels.ndim, connectivity=self.connectivity)

        dilations = random.randint(*self.dilations)

        dilated = binary_dilation(labels, structure=struct, iterations=dilations)

        labels = dilated.astype(labels.dtype)

        batch[self.labels].data = labels


class RandomDilateLabels(gp.BatchFilter):
    def __init__(self, labels, dilations=[2, 8], connectivity=2):

        self.labels = labels
        self.dilations = dilations
        self.connectivity = connectivity

    def process(self, batch, request):

        labels = batch[self.labels].data

        struct = generate_binary_structure(labels.ndim, connectivity=self.connectivity)

        new_labels = np.zeros_like(labels)
        for id in np.unique(labels):
            if id == 0:
                continue
            mask = labels == id
            dilations = random.randint(*self.dilations)
            dilated = binary_dilation(mask, structure=struct, iterations=dilations)

            # make sure we don't overlap existing labels
            dilated[labels > 0] = False
            new_labels[dilated] = id

        batch[self.labels].data = new_labels


class Relabel(gp.BatchFilter):
    def __init__(self, labels, connectivity=1):

        self.labels = labels
        self.connectivity = connectivity

    def process(self, batch, request):

        labels = batch[self.labels].data

        relabeled = relabel(labels, connectivity=self.connectivity).astype(labels.dtype)  # type: ignore

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

        batch.arrays[self.key] = gp.Array(np.zeros(shape, dtype=spec.dtype), spec)

        return batch


def random_source_pipeline(
    voxel_size=(8, 8, 8),
    input_shape=(148, 148, 148),
    dtype=np.uint8,
    expand_labels=False,
    relabel_connectivity=1,
    dilate_connectivity=2,
    random_dilate=True,
    random_dilate_connectivity=2,
    num_points=(20, 150),
    gaussian_noise_args=(0, 0.1),
    gaussian_blur_args=(0.5, 1.5),
    membrane_like=True,
    membrane_size=3,
    inside_value=0.5,
):
    """Create a random source pipeline and batch request for example training.

    Args:

        voxel_size (tuple of int): The size of a voxel in world units.
        input_shape (tuple of int): The shape of the input arrays.
        dtype (numpy.dtype): The dtype of the label arrays.
        expand_labels (bool): Whether to expand the labels into the background.
        relabel_connectivity (int): The connectivity used for for relabeling.
        dilate_connectivity (int): The connectivity of the binary structure used for dilation.
        random_dilate (bool): Whether to randomly dilate the individual labels.
        random_dilate_connectivity (int): The connectivity of the binary structure used for random dilation.
        num_points (tuple of int): The range of the number of points to add to the labels.
        gaussian_noise_args (tuple of float): The mean and standard deviation of the Gaussian noise to add to the raw array.
        gaussian_blur_args (tuple of float): The mean and standard deviation of the Gaussian blur to apply to the raw array.
        membrane_like (bool): Whether to generate a membrane-like structure in the raw array.
        membrane_size (int): The width of the membrane-like structure on the outside of the objects.
        inside_value (float): The value to set inside the membranes of objects.

    Returns:

        gunpowder.Pipeline: The batch generating Gunpowder pipeline.
        gunpowder.BatchRequest: The batch request for the pipeline.
    """

    voxel_size = gp.Coordinate(voxel_size)
    input_shape = gp.Coordinate(input_shape)

    labels = gp.ArrayKey("LABELS")
    raw = gp.ArrayKey("RAW")

    input_size = input_shape * voxel_size

    request = gp.BatchRequest()

    request.add(labels, input_size)
    request.add(raw, input_size)

    source_spec = gp.ArraySpec(
        roi=gp.Roi((0, 0, 0), input_size), voxel_size=voxel_size, dtype=dtype
    )
    source = ZerosSource(labels, source_spec)

    pipeline = source

    # randomly sample some points and write them into our zeros array as ones
    pipeline += CreatePoints(labels, num_points=num_points)

    # grow the boundaries
    pipeline += DilatePoints(labels, connectivity=dilate_connectivity)

    # relabel connected components
    pipeline += Relabel(labels, connectivity=relabel_connectivity)

    if expand_labels:
        # expand the labels outwards into the background
        pipeline += ExpandLabels(labels)

    # relabel ccs again to deal with incorrectly connected background
    pipeline += Relabel(labels, connectivity=relabel_connectivity)

    # randomly dilate labels
    if random_dilate:
        pipeline += RandomDilateLabels(labels, connectivity=random_dilate_connectivity)

    # make a raw array
    pipeline += MakeRaw(
        raw,
        labels,
        gaussian_noise_args=gaussian_noise_args,
        gaussian_blur_args=gaussian_blur_args,
        membrane_like=membrane_like,
        membrane_size=membrane_size,
        inside_value=inside_value,
    )

    return pipeline, request

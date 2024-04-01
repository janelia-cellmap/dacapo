from typing import Iterable
import gunpowder as gp
import logging
import numpy as np
import random
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
)
from skimage.measure import label as relabel

logging.basicConfig(level=logging.INFO)


class CreatePoints(gp.BatchFilter):
    """
    A batch filter that creates random points in a 3D label volume.

    Attributes:
        labels (str): The key of the label data in the batch.
        num_points (tuple): A tuple specifying the range of the number of points to create.
    Methods:
        process: Create random points in the label volume.
    """

    def __init__(
        self,
        labels,
        num_points=(20, 150),
    ):
        """
        Initialize the Pipeline object.

        Args:
            labels (list): A list of labels.
            num_points (tuple, optional): A tuple representing the range of number of points. Defaults to (20, 150).
        Examples:
            >>> CreatePoints(labels="LABELS", num_points=(20, 150))
        """
        self.labels = labels
        self.num_points = num_points

    def process(self, batch, request):
        """
        Process the batch by creating random points in the label volume.

        Args:
            batch (dict): The input batch containing label data.
            request (gp.BatchRequest): The batch request.
        Raises:
            ValueError: If the number of points is not an integer.
        Examples:
            >>> CreatePoints.process(batch, request)
        """
        labels = batch[self.labels].data

        num_points = random.randint(*self.num_points)

        z = np.random.randint(1, labels.shape[0] - 1, num_points)
        y = np.random.randint(1, labels.shape[1] - 1, num_points)
        x = np.random.randint(1, labels.shape[2] - 1, num_points)

        labels[z, y, x] = 1

        batch[self.labels].data = labels


class MakeRaw(gp.BatchFilter):
    """
    A batch filter that generates a raw image from labels.

    Attributes:
        raw (str): The key of the raw data in the batch.
        labels (str): The key of the label data in the batch.
        gaussian_noise_args (tuple): A tuple specifying the mean and standard deviation of the Gaussian noise.
        gaussian_noise_lim (float): The limit of the Gaussian noise.
        gaussian_blur_args (tuple): A tuple specifying the mean and standard deviation of the Gaussian blur.
        membrane_like (bool): A boolean indicating whether to generate a membrane-like structure.
        membrane_size (int): The size of the membrane-like structure.
        inside_value (float): The value to set inside the membranes of objects.
    Methods:
        setup: Set up the batch filter by defining the specification of the raw image.
        process: Generate the raw image from the labels.
    """

    class Pipeline:
        def __init__(
            self,
            raw,
            labels,
            gaussian_noise_args: Iterable = (0.5, 0.1),
            gaussian_noise_lim: float = 0.3,
            gaussian_blur_args: Iterable = (0.5, 1.5),
            membrane_like=True,
            membrane_size=3,
            inside_value=0.5,
        ):
            """
            Initialize the Pipeline object.

            Args:
                raw: The raw data.
                labels: The labels data.
                gaussian_noise_args: Tuple of two floats representing the mean and standard deviation
                    of the Gaussian noise to be added to the data. Default is (0.5, 0.1).
                gaussian_noise_lim: The limit of the Gaussian noise. Default is 0.3.
                gaussian_blur_args: Tuple of two floats representing the mean and standard deviation
                    of the Gaussian blur to be applied to the data. Default is (0.5, 1.5).
                membrane_like: Boolean indicating whether to apply membrane-like transformation to the data.
                    Default is True.
                membrane_size: The size of the membrane. Default is 3.
                inside_value: The value to be assigned to the inside of the membrane. Default is 0.5.
            Examples:
                >>> Pipeline(raw="RAW", labels="LABELS", gaussian_noise_args=(0.5, 0.1), gaussian_noise_lim=0.3,
                >>>          gaussian_blur_args=(0.5, 1.5), membrane_like=True, membrane_size=3, inside_value=0.5)
            """
            self.raw = raw
            self.labels = labels
            self.gaussian_noise_args = gaussian_noise_args
            self.gaussian_noise_lim = gaussian_noise_lim
            self.gaussian_blur_args = gaussian_blur_args
            self.membrane_like = membrane_like
            self.membrane_size = membrane_size
            self.inside_value = inside_value

    def setup(self):
        """
        Set up the batch filter by defining the specification of the raw image.

        Raises:
            ValueError: If the data type is not float32.
        Examples:
            >>> MakeRaw.setup()

        """
        spec = self.spec[self.labels].copy()  # type: ignore
        spec.dtype = np.float32
        self.provides(self.raw, spec)

    def process(self, batch, request):
        """
        Process the batch by generating the raw image from the labels.

        Args:
            batch (gp.Batch): The input batch.
            request (gp.BatchRequest): The request for the output batch.
        Returns:
            gp.Batch: The output batch.
        Raises:
            ValueError: If the data type is not float32.
        Examples:
            >>> MakeRaw.process(batch, request)
        """
        labels = batch[self.labels].data
        raw: np.ndarray = np.zeros_like(labels, dtype=np.float32)
        raw[labels > 0] = 1

        # generate membrane-like structure
        if self.membrane_like:
            for id in np.unique(labels):
                if id == 0:
                    continue
                raw[distance_transform_edt(labels == id) > self.membrane_size] = self.inside_value  # type: ignore

        # now add blur
        raw = gaussian_filter(raw, random.uniform(*self.gaussian_blur_args))

        # now add noise
        noise = np.random.normal(*self.gaussian_noise_args, raw.shape)  # type: ignore
        # normalize to [0, gaussian_noise_lim]
        noise -= noise.min()
        noise /= noise.max()
        noise *= self.gaussian_noise_lim

        raw += noise
        raw /= 1 + self.gaussian_noise_lim
        raw = 1 - raw  # invert
        raw.clip(0, 1, out=raw)

        # add to batch
        spec = self._spec[self.raw].copy()  # type: ignore
        spec.roi = request[self.raw].roi
        batch[self.raw] = gp.Array(raw, spec)


class DilatePoints(gp.BatchFilter):
    """
    A batch filter that performs dilation on labeled points.

    Attributes:
        labels (str): The key of the labels data in the batch.
        dilations (list[int]): A list of two integers representing the range of dilations to apply.
    Methods:
        process: Perform dilation on the labeled points.
    """

    def __init__(self, labels, dilations=[2, 8]):
        """
        Initialize the DilatePoints batch filter.

        Args:
            labels (str): The key of the labels data in the batch.
            dilations (list[int]): A list of two integers representing the range of dilations to apply.
        Raises:
            ValueError: If the dilations are not integers.
        Examples:
            >>> DilatePoints(labels="LABELS", dilations=[2, 8])

        """
        self.labels = labels
        self.dilations = dilations

    def process(self, batch, request):
        """
        Process the batch by performing dilation on the labeled points.

        Args:
            batch (Batch): The input batch.
            request (Request): The request object.
        Raises:
            ValueError: If the dilations are not integers.
        Examples:
            >>> DilatePoints.process(batch, request)

        """
        labels = batch[self.labels].data

        dilations = random.randint(*self.dilations)
        labels = (distance_transform_edt(labels == 0) <= dilations).astype(labels.dtype)  # type: ignore

        batch[self.labels].data = labels


class RandomDilateLabels(gp.BatchFilter):
    """
    A batch filter that randomly dilates labels in a batch.

    Attributes:
        labels (str): The key of the labels in the batch.
        dilations (list[int]): A list of two integers representing the range of dilations.
    Methods:
        process: Randomly dilate the labels in the batch.

    """

    def __init__(self, labels, dilations=[2, 8]):
        self.labels = labels
        self.dilations = dilations

    def process(self, batch, request):
        """
        Process the batch by randomly dilating labels.

        Args:
            batch (dict): The input batch.
            request: The request object.
        Raises:
            ValueError: If the dilations are not integers.
        Examples:
            >>> RandomDilateLabels.process(batch, request)

        """
        labels = batch[self.labels].data

        new_labels = np.zeros_like(labels)
        for id in np.unique(labels):
            if id == 0:
                continue
            dilations = np.random.randint(*self.dilations)

            # make sure we don't overlap existing labels
            new_labels[
                np.logical_or(
                    labels == id,
                    np.logical_and(
                        distance_transform_edt(labels != id) <= dilations, labels == 0
                    ),
                )
            ] = id  # type: ignore

        batch[self.labels].data = new_labels


class Relabel(gp.BatchFilter):
    """
    A batch filter that relabels the labels in a batch.

    Args:
        labels (str): The name of the labels data in the batch.
        connectivity (int, optional): The connectivity used for relabeling. Defaults to 1.
    Methods:
        process: Process the batch and relabel the labels.

    """

    def __init__(self, labels, connectivity=1):
        """
        Initialize the Pipeline object.

        Args:
            labels (str): The name of the labels data in the batch.
            connectivity (int, optional): The connectivity used for relabeling. Defaults to 1.
        Raises:
            ValueError: If the connectivity is not an integer.
        Examples:
            >>> Relabel(labels="LABELS", connectivity=1)
        """
        self.labels = labels
        self.connectivity = connectivity

    def process(self, batch, request):
        """
        Process the batch and relabel the labels.

        Args:
            batch (Batch): The input batch.
            request (Request): The request for processing.
        Returns:
            Batch: The output batch.
        Raises:
            ValueError: If the connectivity is not an integer.
        Examples:
            >>> Relabel.process(batch, request)


        """
        labels = batch[self.labels].data

        relabeled = relabel(labels, connectivity=self.connectivity).astype(labels.dtype)  # type: ignore

        batch[self.labels].data = relabeled


class ExpandLabels(gp.BatchFilter):
    """
    A batch filter that expands labels by assigning the nearest label to each pixel within a specified distance.

    Attributes:
        labels (str): The name of the labels data in the batch.
        background (int): The label value representing the background.
    Methods:
        process: Process the batch and expand the labels.

    """

    def __init__(self, labels, background=0):
        """
        Initialize the Pipeline object.

        Args:
            labels (list): A list of labels.
            background (int, optional): The background value. Defaults to 0.
        Raises:
            ValueError: If the background is not an integer.
        Examples:
            >>> ExpandLabels(labels="LABELS", background=0)

        """
        self.labels = labels
        self.background = background

    def process(self, batch, request):
        """
        Process the batch by expanding the labels.

        Args:
            batch (gp.Batch): The input batch.
            request (gp.BatchRequest): The batch request.
        Raises:
            ValueError: If the background is not an integer.
        Examples:
            >>> ExpandLabels.process(batch, request)

        """
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
    """
    A batch provider that generates arrays filled with zeros.

    Attributes:
        key (str): The key to use for the generated array.
        _spec (dict): A dictionary containing the specification of the array.
    Methods:
        setup: Perform any necessary setup before providing batches.
        provide: Provide a batch containing an array filled with zeros.

    """

    def __init__(self, key, spec):
        """
        Initialize a Pipeline object.

        Args:
            key (str): The key to use for the generated array.
            spec (ArraySpec): The specification of the array.
        Raises:
            ValueError: If the key is not a string.
        Examples:
            >>> ZerosSource(key="LABELS", spec=ArraySpec(roi=gp.Roi((0, 0, 0), (100, 100, 100)), voxel_size=(8, 8, 8), dtype=np.uint8))

        """
        self.key = key
        self._spec = {key: spec}

    def setup(self):
        """
        Perform any necessary setup before providing batches.

        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> ZerosSource.setup()

        """
        pass

    def provide(self, request):
        """
        Provide a batch containing an array filled with zeros.

        Args:
            request (gp.BatchRequest): The request for the batch.
        Returns:
            gp.Batch: The provided batch.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> ZerosSource.provide(request)
        """
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
    random_dilate=True,
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
        random_dilate (bool): Whether to randomly dilate the individual labels.
        num_points (tuple of int): The range of the number of points to add to the labels.
        gaussian_noise_args (tuple of float): The mean and standard deviation of the Gaussian noise to add to the raw array.
        gaussian_blur_args (tuple of float): The mean and standard deviation of the Gaussian blur to apply to the raw array.
        membrane_like (bool): Whether to generate a membrane-like structure in the raw array.
        membrane_size (int): The width of the membrane-like structure on the outside of the objects.
        inside_value (float): The value to set inside the membranes of objects.
    Returns:
        gunpowder.Pipeline: The batch generating Gunpowder pipeline.
        gunpowder.BatchRequest: The batch request for the pipeline.
    Raises:
        ValueError: If the input_shape is not an integer.
    Examples:
        >>> random_source_pipeline(voxel_size=(8, 8, 8), input_shape=(148, 148, 148), dtype=np.uint8, expand_labels=False,
        >>>                        relabel_connectivity=1, random_dilate=True, num_points=(20, 150), gaussian_noise_args=(0, 0.1),
        >>>                        gaussian_blur_args=(0.5, 1.5), membrane_like=True, membrane_size=3, inside_value=0.5)
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
    pipeline += DilatePoints(labels)

    # relabel connected components
    pipeline += Relabel(labels, connectivity=relabel_connectivity)

    if expand_labels:
        # expand the labels outwards into the background
        pipeline += ExpandLabels(labels)

        # relabel ccs again to deal with incorrectly connected background
        pipeline += Relabel(labels, connectivity=relabel_connectivity)

    # randomly dilate labels
    if random_dilate:
        pipeline += RandomDilateLabels(labels)

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

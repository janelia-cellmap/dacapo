# TODO This file should be deleted after we megrate to the new version of funlib.persistence

import numpy as np
from funlib.persistence import Array


def to_ndarray(result_data, roi, fill_value=0):
    """An alternative implementation of `__getitem__` that supports
    using fill values to request data that may extend outside the
    roi covered by result_data.

    Args:

        roi (`class:Roi`, optional):

            If given, copy only the data represented by this ROI. This is
            equivalent to::

                array[roi].to_ndarray()

        fill_value (scalar, optional):

            The value to use to fill in values that are outside the ROI
            provided by this data. Defaults to 0.
    """

    shape = roi.shape / result_data.voxel_size
    data = np.zeros(
        result_data[result_data.roi].shape[: result_data.n_channel_dims] + shape,
        dtype=result_data.data.dtype,
    )
    if fill_value != 0:
        data[:] = fill_value

    array = Array(data, roi, result_data.voxel_size)

    shared_roi = result_data.roi.intersect(roi)

    if not shared_roi.empty:
        array[shared_roi] = result_data[shared_roi]

    return data


def save_ndarray(data, roi, array):
    """An alternative implementation of `__setitem__` that supports
    using fill values to request data that may extend outside the
    roi covered by result_data.

    Args:

        roi (`class:Roi`, optional):

            If given, copy only the data represented by this ROI. This is
            equivalent to::

                array[roi] = data

        fill_value (scalar, optional):

            The value to use to fill in values that are outside the ROI
            provided by this data. Defaults to 0.
    """
    intersection_roi = roi.intersect(array.roi)
    if not intersection_roi.empty:
        result_array = Array(data, roi, array.voxel_size)
        array[intersection_roi] = result_array[intersection_roi]

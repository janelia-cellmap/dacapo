# TODO This file should be deleted after we megrate to the new version of funlib.persistence

import numpy as np
from funlib.persistence import Array


def to_ndarray(result_data, roi, fill_value=0):
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
    intersection_roi = roi.intersect(array.roi)
    if not intersection_roi.empty:
        result_array = Array(data, roi, array.voxel_size)
        array[intersection_roi] = result_array[intersection_roi]

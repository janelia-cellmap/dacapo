from typing import Optional
import neuroglancer
from IPython.display import IFrame
import numpy as np
import gunpowder as gp
from funlib.persistence import Array
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray


def get_viewer(
    raw_array: gp.Array | Array | ZarrArray,
    labels_array: gp.Array | Array | ZarrArray,
    pred_array: Optional[gp.Array | Array | ZarrArray] = None,
    pred_labels_array: Optional[gp.Array | Array | ZarrArray] = None,
    width: int = 1500,
    height: int = 600,
) -> IFrame:
    arrays = {
        "raw": raw_array,
        "labels": labels_array,
    }
    if pred_array is not None:
        arrays["pred"] = pred_array
    if pred_labels_array is not None:
        arrays["pred_labels"] = pred_labels_array

    data = {}
    voxel_sizes = {}
    for name, array in arrays.items():
        if hasattr(array, "to_ndarray"):
            data[name] = array.to_ndarray()
        else:
            data[name] = array.data
        if hasattr(array, "voxel_size"):
            voxel_sizes[name] = array.voxel_size
        else:
            voxel_sizes[name] = array.spec.voxel_size

    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()
    with viewer.txn() as state:
        state.showSlices = False
        add_seg_layer(state, "labels", data["labels"], voxel_sizes["labels"])

        add_scalar_layer(state, "raw", data["raw"], voxel_sizes["raw"])

        if "pred" in data:
            add_scalar_layer(state, "pred", data["pred"], voxel_sizes["pred"])

        if "pred_labels" in data:
            add_seg_layer(
                state, "pred_labels", data["pred_labels"], voxel_sizes["pred_labels"]
            )

    return IFrame(src=viewer, width=width, height=height)


def add_seg_layer(state, name, data, voxel_size):
    state.layers[name] = neuroglancer.SegmentationLayer(
        # segments=[str(i) for i in np.unique(data[data > 0])], # this line will cause all objects to be selected and thus all meshes to be generated...will be slow if lots of high res meshes
        source=neuroglancer.LocalVolume(
            data=data,
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units=["nm", "nm", "nm"],
                scales=voxel_size,
            ),
        ),
        segments=np.unique(data[data > 0]),
    )


def add_scalar_layer(state, name, data, voxel_size):
    state.layers[name] = neuroglancer.ImageLayer(
        source=neuroglancer.LocalVolume(
            data=data,
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units=["nm", "nm", "nm"],
                scales=voxel_size,
            ),
        ),
    )

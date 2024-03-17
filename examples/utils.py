import neuroglancer
from IPython.display import IFrame
import numpy as np
import gunpowder as gp
from funlib.persistence import Array
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray


def get_viewer(
    raw_array: gp.Array | Array | ZarrArray,
    labels_array: gp.Array | Array | ZarrArray,
    width=1500,
    height=600,
) -> IFrame:
    if hasattr(labels_array, "to_ndarray"):
        labels_data = labels_array.to_ndarray()
    else:
        labels_data = labels_array.data
    if hasattr(raw_array, "to_ndarray"):
        raw_data = raw_array.to_ndarray()
    else:
        raw_data = raw_array.data

    if hasattr(labels_array, "voxel_size"):
        labels_voxel_size = labels_array.voxel_size
    else:
        labels_voxel_size = labels_array.spec.voxel_size
    if hasattr(raw_array, "voxel_size"):
        raw_voxel_size = raw_array.voxel_size
    else:
        raw_voxel_size = raw_array.spec.voxel_size
    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()
    with viewer.txn() as state:
        state.showSlices = False
        state.layers["segs"] = neuroglancer.SegmentationLayer(
            # segments=[str(i) for i in np.unique(data[data > 0])], # this line will cause all objects to be selected and thus all meshes to be generated...will be slow if lots of high res meshes
            source=neuroglancer.LocalVolume(
                data=labels_data,
                dimensions=neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units=["nm", "nm", "nm"],
                    scales=labels_voxel_size,
                ),
                # voxel_offset=ds.roi.begin / ds.voxel_size,
            ),
            segments=np.unique(labels_data[labels_data > 0]),
        )

        state.layers["raw"] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=raw_data,
                dimensions=neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units=["nm", "nm", "nm"],
                    scales=raw_voxel_size,
                ),
            ),
        )

    return IFrame(src=viewer, width=width, height=height)

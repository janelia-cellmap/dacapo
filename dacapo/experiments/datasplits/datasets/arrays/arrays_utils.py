import numpy as np
import zarr
import neuroglancer
from funlib.show.neuroglancer import ScalePyramid
from funlib.persistence import open_ds

def open_dataset(f, ds):
        original_ds = ds
        slices_str = original_ds[len(ds):]

        try:
            dataset_as = []
            if all(key.startswith("s") for key in zarr.open(f)[ds].keys()):
                raise AttributeError("This group is a multiscale array!")
            for key in zarr.open(f)[ds].keys():
                dataset_as.extend(open_dataset(f, f"{ds}/{key}{slices_str}"))
            return dataset_as
        except AttributeError as e:
            # dataset is an array, not a group
            pass

        print("ds    :", ds)
        try:
            zarr.open(f)[ds].keys()
            is_multiscale = True
        except:
            is_multiscale = False

        if not is_multiscale:
            a = open_ds(f, ds)

            if a.data.dtype == np.int64 or a.data.dtype == np.int16:
                print("Converting dtype in memory...")
                a.data = a.data[:].astype(np.uint64)

            return [(a, ds)]
        else:
            return [([open_ds(f, f"{ds}/{key}") for key in zarr.open(f)[ds].keys()], ds)]
    
def parse_dims(array):

    if type(array) == list:
        array = array[0]

    if type(array) == tuple:
        array = array[0]

    dims = len(array.data.shape)
    spatial_dims = array.roi.dims
    channel_dims = dims - spatial_dims

    print("dims        :", dims)
    print("spatial dims:", spatial_dims)
    print("channel dims:", channel_dims)

    return dims, spatial_dims, channel_dims
        
def create_coordinate_space(array, spatial_dim_names, channel_dim_names, unit):
    if type(array) == list:
        array = array[0]

    if type(array) == tuple:
        array = array[0]

    dims, spatial_dims, channel_dims = parse_dims(array)
    assert spatial_dims > 0

    if channel_dims > 0:
        channel_names = channel_dim_names[-channel_dims:]
    else:
        channel_names = []
    spatial_names = spatial_dim_names[-spatial_dims:]
    names = channel_names + spatial_names
    units = [""] * channel_dims + [unit] * spatial_dims
    scales = [1] * channel_dims + list(array.voxel_size)

    print("Names    :", names)
    print("Units    :", units)
    print("Scales   :", scales)

    return neuroglancer.CoordinateSpace(
        names=names,
        units=units,
        scales=scales)

def get_neuroglancer_layer(
        array,
        spatial_dim_names=None,
        channel_dim_names=None,
        rgb_channels=None,
        units='nm'):

    """Add a layer to a neuroglancer context.

    Args:

        array:

            A ``daisy``-like array, containing attributes ``roi``,
            ``voxel_size``, and ``data``. If a list of arrays is given, a
            ``ScalePyramid`` layer is generated.

        name:

            The name of the layer.

        spatial_dim_names:

            The names of the spatial dimensions. Defaults to ``['t', 'z', 'y',
            'x']``. The last elements of this list will be used (e.g., if your
            data is 2D, the channels will be ``['y', 'x']``).

        channel_dim_names:

            The names of the non-spatial (channel) dimensions. Defaults to
            ``['b^', 'c^']``.  The last elements of this list will be used
            (e.g., if your data is 2D but the shape of the array is 3D, the
            channels will be ``['c^']``).

        opacity:

            A float to define the layer opacity between 0 and 1.

        shader:

            A string to be used as the shader. Possible values are:

                None     :  neuroglancer's default shader
                'rgb'    :  An RGB shader on dimension `'c^'`. See argument
                            ``rgb_channels``.
                'color'  :  Shows intensities as a constant color. See argument
                            ``color``.
                'binary' :  Shows a binary image as black/white.
                'heatmap':  Shows an intensity image as a jet color map.

        rgb_channels:

            Which channels to use for RGB (default is ``[0, 1, 2]``).

        color:

            A list of floats representing the RGB values for the constant color
            shader.

        visible:

            A bool which defines the initial layer visibility.

        value_scale_factor:

            A float to scale array values with for visualization.

        units:

            The units used for resolution and offset.
    """

    if channel_dim_names is None:
        channel_dim_names = ["b", "c^"]
    if spatial_dim_names is None:
        spatial_dim_names = ["t", "z", "y", "x"]

    if rgb_channels is None:
        rgb_channels = [0, 1, 2]
    
    if type(array) == list and len(array) == 1:
        array = array[0]

    is_multiscale = type(array) == list

    dims, spatial_dims, channel_dims = parse_dims(array)


    if is_multiscale:

        dimensions = []
        for a in array:
            print("got ",a)
            dimensions.append(
                create_coordinate_space(
                    a,
                    spatial_dim_names,
                    channel_dim_names,
                    units))

        if type(array[0]) == tuple:
            voxel_offset = [0] * channel_dims + \
                list(array[0][0].roi.offset / array[0][0].voxel_size)
        else:
            voxel_offset = [0] * channel_dims + \
                list(array[0].roi.offset / array[0].voxel_size)
        
        if type(a) == tuple:
            layer = ScalePyramid(
                [
                    neuroglancer.LocalVolume(
                        data=a[0].data,
                        voxel_offset=voxel_offset,
                        dimensions=array_dims
                    )
                    for a, array_dims in zip(array, dimensions)
                ]
            )
        else:
            layer = ScalePyramid(
                [
                    neuroglancer.LocalVolume(
                        data=a.data,
                        voxel_offset=voxel_offset,
                        dimensions=array_dims
                    )
                    for a, array_dims in zip(array, dimensions)
                ]
            )

    else:

        if type(array) == tuple:
            array = array[0]

        voxel_offset = [0] * channel_dims + \
                list(array.roi.offset / array.voxel_size)

        dimensions = create_coordinate_space(
            array,
            spatial_dim_names,
            channel_dim_names,
            units)

        layer = neuroglancer.LocalVolume(
            data=array.data,
            voxel_offset=voxel_offset,
            dimensions=dimensions,
        )
        return layer
import gunpowder as gp
import zarr


def get_voxel_size(filename, ds_name):

    ds = gp.ArrayKey('DS')
    source = gp.ZarrSource(
        str(filename),
        {
            ds: ds_name
        })
    with gp.build(source):
        return source.spec[ds].voxel_size

def get_dataset_roi(filename, ds_name):

    ds = gp.ArrayKey('DS')
    source = gp.ZarrSource(
        str(filename),
        {
            ds: ds_name
        })
    with gp.build(source):
        return source.spec[ds].roi

def get_dataset_shape(filename, ds_name):

    return zarr.open(str(filename))[ds_name].shape

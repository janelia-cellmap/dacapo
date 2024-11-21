# Fix ome-zarr metadata reading
import os
import zarr
import logging
from funlib.geometry import Coordinate

logger = logging.getLogger(__name__)


def access_parent(node):
    parent_path = os.path.split(node.path)[0]
    return zarr.hierarchy.group(store=node.store, path=parent_path)


def regularize_offset(voxel_size_float, offset_float):
    """
        offset is not a multiple of voxel_size. This is often due to someone defining
        offset to the point source of each array element i.e. the center of the rendered
        voxel, vs the offset to the corner of the voxel.
        apparently this can be a heated discussion. See here for arguments against
        the convention we are using: http://alvyray.com/Memos/CG/Microsoft/6_pixel.pdf

    Args:
        voxel_size_float ([float]): float voxel size list
        offset_float ([float]): float offset list
    Returns:
        (Coordinate, Coordinate)): returned offset size that is multiple of voxel size
    """
    voxel_size, offset = Coordinate(voxel_size_float), Coordinate(offset_float)

    if voxel_size is not None and (offset / voxel_size) * voxel_size != offset:

        logger.debug(
            f"Offset: {offset} being rounded to nearest voxel size: {voxel_size}"
        )
        offset = (
            (Coordinate(offset) + (Coordinate(voxel_size) / 2)) / Coordinate(voxel_size)
        ) * Coordinate(voxel_size)
        logger.debug(f"Rounded offset: {offset}")

    return Coordinate(voxel_size), Coordinate(offset)


def check_for_offset(array, order):
    """checks specific attributes(offset, transform["translate"]) for offset
        value in the parent directory of the input array

    Args:
        array (zarr.core.Array): array to check
        order (string): colexicographical/lexicographical order
    Raises:
        ValueError: raises value error if no offset value is found

    Returns:
       [float] : returns offset of the voxel (unitless) in respect to
                the center of the coordinate system
    """
    offset = None
    parent_group = access_parent(array)
    for item in [array, parent_group]:

        if "offset" in item.attrs:
            offset = item.attrs["offset"]
            return offset

        elif "transform" in item.attrs:
            transform_order = item.attrs["transform"].get("ordering", "C")
            offset = item.attrs["transform"]["translate"]
            if transform_order != order:
                offset = offset[::-1]
            return offset

    return offset


def check_for_multiscale(group):
    """check if multiscale attribute exists in the input group and for any parent level group

    Args:
        group (zarr.hierarchy.Group): group to check

    Returns:
        tuple({}, zarr.hierarchy.Group): (multiscales attribute body, zarr group where multiscales was found)
    """
    multiscales = group.attrs.get("multiscales", None)

    if multiscales:
        return (multiscales, group)

    if group.path == "":
        return (multiscales, group)

    return check_for_multiscale(access_parent(group))


def check_for_units(array, order):
    """checks specific attributes(units, pixelResolution["unit"] transform["units"])
        for units(nm, cm, etc.) value in the parent directory of the input array

    Args:
        array (zarr.core.Array): array to check
        order (string): colexicographical/lexicographical order
    Raises:
        ValueError: raises value error if no units value is found

    Returns:
       [string] : returns units for the voxel_size
    """

    units = None
    parent_group = access_parent(array)
    for item in [array, parent_group]:

        if "units" in item.attrs:
            return item.attrs["units"]
        elif "pixelResolution" in item.attrs:
            unit = item.attrs["pixelResolution"]["unit"]
            return [unit for _ in range(len(array.shape))]
        elif "transform" in item.attrs:
            # Davis saves transforms in C order regardless of underlying
            # memory format (i.e. n5 or zarr). May be explicitly provided
            # as transform.ordering
            transform_order = item.attrs["transform"].get("ordering", "C")
            units = item.attrs["transform"]["units"]
            if transform_order != order:
                units = units[::-1]
            return units

    if units is None:
        Warning(
            f"No units attribute was found for {type(array.store)} store. Using pixels."
        )
        return "pixels"


def check_for_attrs_multiscale(ds, multiscale_group, multiscales):
    """checks multiscale attribute of the .zarr or .n5 group
        for voxel_size(scale), offset(translation) and units values

    Args:
        ds (zarr.core.Array): input zarr Array
        multiscale_group (zarr.hierarchy.Group): the group attrs
                                                that contains multiscale
        multiscales ({}): dictionary that contains all the info necessary
                            to create multiscale resolution pyramid

    Returns:
        ([float],[float],[string]): returns (voxel_size, offset, physical units)
    """

    voxel_size = None
    offset = None
    units = None

    if multiscales is not None:
        logger.info("Found multiscales attributes")
        scale = os.path.split(ds.path)[1]
        if isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore)):
            for level in multiscales[0]["datasets"]:
                if level["path"] == scale:

                    voxel_size = level["transform"]["scale"]
                    offset = level["transform"]["translate"]
                    units = level["transform"]["units"]
                    return voxel_size, offset, units
        # for zarr store
        else:
            units = [item["unit"] for item in multiscales[0]["axes"]]
            for level in multiscales[0]["datasets"]:
                if level["path"].lstrip("/") == scale:
                    for attr in level["coordinateTransformations"]:
                        if attr["type"] == "scale":
                            voxel_size = attr["scale"]
                        elif attr["type"] == "translation":
                            offset = attr["translation"]
                    return voxel_size, offset, units

    return voxel_size, offset, units


def _read_attrs(ds, order="C"):
    """check n5/zarr metadata and returns voxel_size, offset, physical units,
        for the input zarr array(ds)

    Args:
        ds (zarr.core.Array): input zarr array
        order (str, optional): _description_. Defaults to "C".

    Raises:
        TypeError: incorrect data type of the input(ds) array.
        ValueError: returns value error if no multiscale attribute was found
    Returns:
        _type_: _description_
    """
    voxel_size = None
    offset = None
    units = None
    multiscales = None

    if not isinstance(ds, zarr.core.Array):
        raise TypeError(
            f"{os.path.join(ds.store.path, ds.path)} is not zarr.core.Array"
        )

    # check recursively for multiscales attribute in the zarr store tree
    multiscales, multiscale_group = check_for_multiscale(group=access_parent(ds))

    # check for attributes in .zarr group multiscale
    if not isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore)):
        if multiscales:
            voxel_size, offset, units = check_for_attrs_multiscale(
                ds, multiscale_group, multiscales
            )

    # if multiscale attribute is missing
    if voxel_size is None:
        voxel_size = check_for_voxel_size(ds, order)
    if offset is None:
        offset = check_for_offset(ds, order)
    if units is None:
        units = check_for_units(ds, order)

    dims = len(ds.shape)
    dims = dims if dims <= 3 else 3

    if voxel_size is not None and offset is not None and units is not None:
        if order == "F" or isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore)):
            return voxel_size[::-1], offset[::-1], units[::-1]
        else:
            return voxel_size, offset, units

    # if no voxel offset are found in transform, offset or scale, check in n5 multiscale attribute:
    if (
        isinstance(ds.store, (zarr.n5.N5Store, zarr.n5.N5FSStore))
        and multiscales != None
    ):

        voxel_size, offset, units = check_for_attrs_multiscale(
            ds, multiscale_group, multiscales
        )

    # return default value if an attribute was not found
    if voxel_size is None:
        voxel_size = (1,) * dims
        Warning(f"No voxel_size attribute was found. Using {voxel_size} as default.")
    if offset is None:
        offset = (0,) * dims
        Warning(f"No offset attribute was found. Using {offset} as default.")
    if units is None:
        units = "pixels"
        Warning(f"No units attribute was found. Using {units} as default.")

    if order == "F":
        return voxel_size[::-1], offset[::-1], units[::-1]
    else:
        return voxel_size, offset, units


def _read_voxel_size_offset(path, order="C"):
    ds = zarr.open(path, mode="r")
    try:
        order = ds.attrs["order"]
    except KeyError:
        order = ds.order

    voxel_size, offset, units = _read_attrs(ds, order)

    return regularize_offset(voxel_size, offset)

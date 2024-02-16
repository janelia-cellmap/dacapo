"""
This module manages the DVID array which contains the main 3D imaging and annotation data types of the DVID API.

Classes:
    DVIDArray
"""

class DVIDArray(Array):
    """This is a DVID array

    Attributes:
        name (str): Name of the array.
        source (tuple[str, str, str]): The source of the array.
        attrs: properties of the DVID array
    """

    def __init__(self, array_config):
        """ Create DVID array with the provided array configurations."""
        super().__init__()
        self.name: str = array_config.name
        self.source: tuple[str, str, str] = array_config.source

    def __str__(self):
        """Convert the DVIDArray instance to string."""
        return f"DVIDArray({self.source})"

    def __repr__(self):
        """Representation of the DVIDArray instance."""
        return f"DVIDArray({self.source})"

    @lazy_property.LazyProperty
    def attrs(self):
        """Fetches attributes of DVID array."""
        return fetch_info(*self.source)

    @property
    def axes(self):
        """Returns all the axes of array."""
        return ["c", "z", "y", "x"][-self.dims :]

    @property
    def dims(self) -> int:
        """Returns the number of dimensions of voxel."""
        return self.voxel_size.dims

    @lazy_property.LazyProperty
    def _daisy_array(self) -> funlib.persistence.Array:
        """Does not return anything, need to be implemented in child class"""
        raise NotImplementedError()

    @lazy_property.LazyProperty
    def voxel_size(self) -> Coordinate:
        """Returns voxel size as coordinates"""
        return Coordinate(self.attrs["Extended"]["VoxelSize"])

    @lazy_property.LazyProperty
    def roi(self) -> Roi:
        """Returns Roi (Region of Interest) of DVID array."""
        return Roi(
            Coordinate(self.attrs["Extents"]["MinPoint"]) * self.voxel_size,
            Coordinate(self.attrs["Extents"]["MaxPoint"]) * self.voxel_size,
        )

    @property
    def writable(self) -> bool:
        """Returns False by default, DVID array should be read-only."""
        return False

    @property
    def dtype(self) -> Any:
        """Returns type of the array data"""
        return np.dtype(self.attrs["Extended"]["Values"][0]["DataType"])

    @property
    def num_channels(self) -> Optional[int]:
        """Returns none by default. Has to be implemented in child class, if supported."""
        return None

    @property
    def spatial_axes(self) -> List[str]:
        """Returns the axis which are not ['c', 'b']."""
        return [ax for ax in self.axes if ax not in set(["c", "b"])]

    @property
    def data(self) -> Any:
        """Not implemented. Needs to be implemented in child class"""
        raise NotImplementedError()

    def __getitem__(self, roi: Roi) -> np.ndarray[Any, Any]:
        """Returns the content of DVID array."""
        box = np.array(
            (roi.offset / self.voxel_size, (roi.offset + roi.shape) / self.voxel_size)
        )
        if self.source[2] == "grayscale":
            data = fetch_raw(*self.source, box)
        elif self.source[2] == "segmentation":
            data = fetch_labelmap_voxels(*self.source, box)
        else:
            raise Exception(self.source)
        return data

    def _can_neuroglance(self) -> bool:
        """Check if the data can be viewed with Neuroglancer browser"""
        return True

    def _neuroglancer_source(self):
        """Needs to be implemented in child class."""
        raise NotImplementedError()

    def _neuroglancer_layer(self) -> Tuple[neuroglancer.ImageLayer, Dict[str, Any]]:
        """Returns the Neuroglancer layer and its properties as a dict"""
        raise NotImplementedError()

    def _transform_matrix(self):
        """Provides transformation matrix. Not implemented yet."""
        raise NotImplementedError()

    def _output_dimensions(self) -> Dict[str, Tuple[float, str]]:
        """Provides dimensions of the output. Not implemented yet."""
        raise NotImplementedError()

    def _source_name(self) -> str:
        """Provides name of the source. Not implemented yet."""
        raise NotImplementedError()

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Method to add metadata to DVIDArray. Not implemented yet."""
        raise NotImplementedError()
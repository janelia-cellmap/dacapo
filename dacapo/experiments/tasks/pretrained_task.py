from dacapo.io import PbConfig, h5py_like

class ConduitFidiskRegular(h5py_like.Dataset):
    """
    A 'ConduitFidiskRegular' is a dataset class in dacapo's file system. 

    It's an interface for reading and writing regular h5 files. In constructor, 
    it attempts to automatically determine whether the file is read or write mode.

    Attributes:
        file (h5py.File): The read/write file object.
    """
    def __init__(self, config: PbConfig):
        """
        Initializes the 'ConduitFidiskRegular' with the specified configuration.

        The constructor opens file, read or write mode is determined based on 
        the provided configuration state ( config.open ).

        Args:
            config (PbConFig): A configuration object containing path file and open state.
                              It includes the path file and the open state (reading or writing).
        """
        super().__init__(omode=config.open)
        self.file = h5py.File(config.path, self.omode)
    
    def close(self):
        """
        Closes the file if it is open.

        This method directly calls the `close` method of h5py.File object.
        """
        if self.file is not None:
            self.file.close()
        super().close()

    def slice_datasets(self, names):
        """
        Creates a generator from given names and returns a dict of datasets.

        This method iterates over the names and yields datasets as dictionary.

        Args:
            names (iter): An iterable of dataset names to be sliced.

        Returns:
            dict: A dictionary where each key-value pair represents a dataset name and its content.
        """
        return {
            name: self[name] for name in names
        } if names is not None else {name: self[name] for name in self.keys()}
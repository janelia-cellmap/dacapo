```python
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray

import zarr
import neuroglancer
import attr

from abc import ABC, abstractmethod
import itertools
import json
from pathlib import Path
from typing import Optional, Tuple

@attr.s
class LocalArrayIdentifier:
    """
    A class used to identify local arrays.

    Attributes
    ----------
    container : Path
        The path to the container
    dataset : str
        The dataset name
    """
    container: Path = attr.ib()
    dataset: str = attr.ib()


@attr.s
class LocalContainerIdentifier:
    """
    A class used to identify local containers.

    Attributes
    ----------
    container : Path
        The path to the container
    """

    container: Path = attr.ib()

    def array_identifier(self, dataset) -> LocalArrayIdentifier:
        """
        Returns a LocalArrayIdentifier object for specified dataset.
        
        Parameters
        ----------
        dataset: str
            The name of the dataset.

        Returns
        -------
        LocalArrayIdentifier
            A LocalArrayIdentifier object.
        """
        return LocalArrayIdentifier(self.container, dataset)


class ArrayStore(ABC):
    """Base class for array stores.
    Provides functions to create, write, display and remove arrays.

    This class is designed to support I/O on local arrays. 
    It generates identifiers for the caller to create and write arrays.
    """
    # methods are omitted for brevity.

    def _visualize_training(self, run):
        """
        Returns a neuroglancer link to visualize snapshots and validations.

        The method creates an interactive viewer for visualizing data in 3D. 
        The viewer supports real-time sharing of data with multiple
        collaborators and powerful segmentation and image annotation tools.

        Parameters
        ----------
        run: str
            The name of the run.

        Returns
        -------
        str
            A URL string that points to the neuroglancer viewer.
        """
        # code omitted for brevity.
```
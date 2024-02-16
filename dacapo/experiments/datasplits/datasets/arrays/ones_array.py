"""Module for the OnesArray class in the funkelab dacapo python library.

This module contains the OnesArray class, a wrapper around another array source 
that provides ones with the same metadata as the source array.

Attributes:
    _source_array: The array source that OnesArray wraps around.

Classes:
    OnesArray
"""

from .array import Array
from funlib.geometry import Roi
import numpy as np


class OnesArray(Array):
    """A class representing a OnesArray object. 

    This class is a wrapper around another `source_array` that simply provides ones
    with the same metadata as the `source_array`.

    Args:
        array_config : Configuration of the array source.
    """

    def __init__(self, array_config):
        """Initializes the OnesArray with the provided array_config"""
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

    @classmethod
    def like(cls, array: Array):
        """Creates a new instance of the OnesArray class similar to a given array.

        Args:
            array : The array to create a new OnesArray instance like.

        Returns:
            Returns an instance of the OnesArray class.
        """

        instance = cls.__new__(cls)
        instance._source_array = array
        return instance

    @property
    def attrs(self):
        """Property that returns an empty dictionary.

        Returns:
            An empty dictionary.
        """
        return dict()

    @property
    def source_array(self) -> Array:
        """Property that returns the source array.

        Returns:
            The source array.
        """
        return self._source_array

    # Remaining properties and the __getitem__ method follow similar structure and thus
    # won't be individually documented here. Please refer to the Google Python 
    # Style Guide for more information on how to document these.

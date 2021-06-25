from abc import ABC, abstractmethod


class ArrayStore(ABC):
    """Base class for array stores.

    Creates identifiers for the caller to create and write arrays. Provides
    only rudimentary support for IO itself (currently only to remove
    arrays)."""

    @abstractmethod
    def validation_prediction_array(self, run_name, iteration):
        """Get the array identifier for a particular validation prediction."""
        pass

    @abstractmethod
    def validation_output_array(self, run_name, iteration, parameters):
        """Get the array identifier for a particular validation output."""
        pass

    @abstractmethod
    def remove(self, array_identifier):
        """Remove an array by its identifier."""
        pass

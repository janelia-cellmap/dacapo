from abc import ABC, abstractmethod


class WeightsStore(ABC):
    """Base class for network weight stores."""

    @abstractmethod
    def latest_iteration(self, run):
        """Return the latest iteration for which weights are available for the
        given run."""
        pass

    @abstractmethod
    def store_weights(self, run, iteration):
        """Store the network weights of the given run."""
        pass

    @abstractmethod
    def retrieve_weights(self, run, iteration):
        """Retrieve the network weights of the given run."""
        pass

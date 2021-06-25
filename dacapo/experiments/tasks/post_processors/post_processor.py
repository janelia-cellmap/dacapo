from abc import ABC, abstractmethod


class PostProcessor(ABC):
    """Base class of all post-processors.

    A post-processor takes a model's prediction and converts it into the final
    output (e.g., per-voxel class probabilities into a semantic segmentation).
    """

    @abstractmethod
    def enumerate_parameters(self):
        """Enumerate all possible parameters of this post-processor. Should
        yield instances of ``PostProcessorParameters``."""
        pass

    @abstractmethod
    def set_prediction(self, prediction_array):
        pass

    @abstractmethod
    def process(
            self,
            parameters,
            output_array):
        """Convert predictions into the final output."""
        pass

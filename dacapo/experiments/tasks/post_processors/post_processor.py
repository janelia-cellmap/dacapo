from abc import ABC, abstractmethod


class PostProcessor(ABC):
    """Base class of all post-processors.

    A post-processor takes a model's prediction and converts it into the final
    output (e.g., per-voxel class probabilities into a semantic segmentation).
    """

    @abstractmethod
    def process(self, container, prediction_dataset, output_dataset):
        """Convert predictions into the final output.

        Since the predictions (and therefore also the output) are in general
        too large to be held in memory, this method receives the path to a zarr
        container and the dataset names of the prediction and the output
        dataset to be generated.
        """
        pass

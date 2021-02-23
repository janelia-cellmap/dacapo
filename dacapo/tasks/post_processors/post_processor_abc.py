from abc import ABC, abstractmethod
import time


class PostProcessorABC(ABC):
    """
    This class handles post processing.

    The majority of the complexity of the PostProcessor comes from
    the fact that it must be daisy processing friendly so that it can
    be applied to arbitrarily large volumes.

    Since many common postprocessing pipelines cannot be handled as a
    single daisy task, a PostProcessor is considered to be a series
    of `PostProcessingStep`s

    Each `PostProcessor` must have a zarr container defined, along with
    a name, which will be used along with the run name to generate a
    mongodb collection to keep track of block statuses. This will
    also be used to save any data that an individual step needs to save.
    """

    @abstractmethod
    def task(self):
        """
        return a daisy task with the appropriate upstream tasks. When run
        blockwise through daisy, this task should result in the dataset
        defined by the "prediction" arg being populated.
        """
        pass

    @abstractmethod
    def daisy_steps(self):
        pass

    def set_prediction(self, prediction):
        """To be implemented in subclasses. This function will be called before
        repeated calls to ``process`` and allows the post-processor to carry
        out general post-processing that does not depend on parameters."""
        pass

    def enumerate(self, prediction):
        """Enumerate all parameter combinations and process the predictions.
        Yields tuples of ``(parameter, post_processed)``."""

        self.set_prediction(prediction)

        for parameters in self.parameter_range:
            if not self.reject_parameters(parameters):
                print(f"Post-processing prediction with {parameters}...")
                start = time.time()
                post_processed = self.process(prediction, parameters)
                print(f"...done ({time.time() - start}s)")
                yield parameters, post_processed

    def reject_parameters(self, parameters):
        """To be implemented in subclasses to reject parameter configurations
        that should be skipped."""
        return False
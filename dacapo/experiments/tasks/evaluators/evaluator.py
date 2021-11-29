from abc import ABC, abstractmethod


class Evaluator(ABC):
    """Base class of all evaluators.

    An evaluator takes a post-processor's output and compares it against
    ground-truth.
    """

    @abstractmethod
    def evaluate(
            self,
            output_array,
            evaluation_dataset):
        """Compare an output dataset against ground-truth from an evaluation
        dataset.

        The evaluation dataset is a dictionary mapping from ``DataKey`` to
        ``ArraySource`` or ``GraphSource``.

        Should return an instance of ``EvaluationScores``.
        """
        pass

    @abstractmethod
    def is_best(self, score, criterion):
        """
        Check if the provided score is the best according to some criterion
        """
        pass

    @abstractmethod
    def set_best(self, iteration_scores):
        """
        Store a mapping from criterion to the best model according to that criterion
        """
        pass

    @property
    @abstractmethod
    def criteria(self):
        """
        A list of all criteria for which a model might be "best". i.e. your
        criteria might be "precision", "recall", and "jaccard". It is unlikely
        that the best iteration/post processing parameters will be the same
        for all 3 of these criteria
        """
        pass
from typing import List


from .evaluator import Evaluator
from .instance_evaluation_scores import InstanceEvaluationScores
from dacapo.utils.voi import voi as _voi
from dacapo.tmp import open_from_identifier

import numpy as np
import numpy_indexed as npi

import logging

logger = logging.getLogger(__name__)


def relabel(array, return_backwards_map=False, inplace=False):
    """
    Relabel array, such that IDs are consecutive. Excludes 0.

    Args:
        array (ndarray):
                The array to relabel.
        return_backwards_map (``bool``, optional):
                If ``True``, return an ndarray that maps new labels (indices in
                the array) to old labels.
        inplace (``bool``, optional):
                Perform the replacement in-place on ``array``.
    Returns:
        A tuple ``(relabelled, n)``, where ``relabelled`` is the relabelled
        array and ``n`` the number of unique labels found.
        If ``return_backwards_map`` is ``True``, returns ``(relabelled, n,
        backwards_map)``.
    Raises:
        ValueError:
                If ``array`` is not of type ``np.ndarray``.
    Examples:
        >>> array = np.array([[1, 2, 0], [0, 2, 1]])
        >>> relabel(array)
        (array([[1, 2, 0], [0, 2, 1]]), 2)
        >>> relabel(array, return_backwards_map=True)
        (array([[1, 2, 0], [0, 2, 1]]), 2, [0, 1, 2])
    Note:
        This function is used to relabel an array, such that IDs are consecutive. Excludes 0.

    """

    if array.size == 0:
        if return_backwards_map:
            return array, 0, []
        else:
            return array, 0

    # get all labels except 0
    old_labels = np.unique(array)
    old_labels = old_labels[old_labels != 0]

    if old_labels.size == 0:
        if return_backwards_map:
            return array, 0, [0]
        else:
            return array, 0

    n = len(old_labels)
    new_labels = np.arange(1, n + 1, dtype=array.dtype)

    replaced = npi.remap(
        array.flatten(), old_labels, new_labels, inplace=inplace
    ).reshape(array.shape)

    if return_backwards_map:
        backwards_map = np.insert(old_labels, 0, 0)
        return replaced, n, backwards_map

    return replaced, n


class InstanceEvaluator(Evaluator):
    """
    A class representing an evaluator for instance segmentation tasks.

    Attributes:
        criteria : List[str]
            the evaluation criteria
    Methods:
        evaluate(output_array_identifier, evaluation_array)
            Evaluate the output array against the evaluation array.
        score
            Return the evaluation scores.
    Note:
        The InstanceEvaluator class is used to evaluate the performance of an instance segmentation task.

    """

    criteria: List[str] = ["voi_merge", "voi_split", "voi"]

    def evaluate(self, output_array_identifier, evaluation_array):
        """
        Evaluate the output array against the evaluation array.

        Args:
            output_array_identifier : str
                the identifier of the output array
            evaluation_array : Zarr Array
                the evaluation array
        Returns:
            InstanceEvaluationScores
                the evaluation scores
        Raises:
            ValueError: if the output array identifier is not valid
        Examples:
            >>> instance_evaluator = InstanceEvaluator()
            >>> output_array_identifier = "output_array"
            >>> evaluation_array = open_from_identifier("evaluation_array")
            >>> instance_evaluator.evaluate(output_array_identifier, evaluation_array)
            InstanceEvaluationScores(voi_merge=0.0, voi_split=0.0)
        Note:
            This function is used to evaluate the output array against the evaluation array.

        """
        output_array = open_from_identifier(output_array_identifier)
        evaluation_data = evaluation_array[evaluation_array.roi].astype(np.uint64)
        output_data = output_array[output_array.roi].astype(np.uint64)
        results = voi(evaluation_data, output_data)

        return InstanceEvaluationScores(
            voi_merge=results["voi_merge"],
            voi_split=results["voi_split"],
        )

    @property
    def score(self) -> InstanceEvaluationScores:
        """
        Return the evaluation scores.

        Returns:
            InstanceEvaluationScores
                the evaluation scores
        Raises:
            NotImplementedError: if the function is not implemented
        Examples:
            >>> instance_evaluator = InstanceEvaluator()
            >>> instance_evaluator.score
            InstanceEvaluationScores(voi_merge=0.0, voi_split=0.0)
        Note:
            This function is used to return the evaluation scores.

        """
        return InstanceEvaluationScores()


def voi(truth, test):
    """
    Calculate the variation of information (VOI) between two segmentations.

    Args:
        truth : ndarray
            the ground truth segmentation
        test : ndarray
            the test segmentation
    Returns:
        dict
            the variation of information (VOI) scores
    Raises:
        ValueError: if the truth and test arrays are not of type np.ndarray
    Examples:
        >>> truth = np.array([[1, 1, 0], [0, 2, 2]])
        >>> test = np.array([[1, 1, 0], [0, 2, 2]])
        >>> voi(truth, test)
        {'voi_split': 0.0, 'voi_merge': 0.0}
    Note:
        This function is used to calculate the variation of information (VOI) between two segmentations.

    """
    voi_split, voi_merge = _voi(test + 1, truth + 1, ignore_groundtruth=[])
    return {"voi_split": voi_split, "voi_merge": voi_merge}

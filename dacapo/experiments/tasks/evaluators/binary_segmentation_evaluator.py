"""
This module contains classes for evaluating binary segmentation provided by 
`dacapo` library: 

1. BinarySegmentationEvaluator: class to compute similarity metrics for binary 
   segmentation.
2. ArrayEvaluator: the class that calculates evaluation metrics.
3. CremiEvaluator: the class that provides Cremi score for segmentation evaluation.

Classes:
-------
`BinarySegmentationEvaluator`: class to compute similarity metrics for binary 
segmentation.

`ArrayEvaluator`: Class that calculates various evaluation metrics such as Dice 
coefficient, Jaccard Coefficient, Hausdorff distance, false discovery rate and VOI.

`CremiEvaluator`: The class provides Cremi score for segmentation evaluation.
"""

class BinarySegmentationEvaluator(Evaluator):
    """
    This class serves to evaluate binary segmentations.

    Attributes:
    -----------
    `clip_distance` (float): Maximum distance till where evaluation will be 
      considered.
    `tol_distance` (float): Tolerance in distance while considering segmentation.
    `channels` (list): List of channels involved in the segmentation.
    """

    def evaluate(self, output_array_identifier, evaluation_array):
        """
        Method to evaluate the segmentation by calculation evaluation data and calling  
        ArrayEvaluator to calculate metrics.

        Returns:
        --------
        `score_dict`: Dictionary of evaluation metrics.
        """

    @property
    def score(self):
        """
        Method to compute evaluation scores.

        Returns:
        --------
        `channel_scores` : List of tuple containing channel and respective evaluation
        scores.
        """

class ArrayEvaluator:
    """
    Class that calculates various evaluation metrics.

    Attributes:
    -----------
    `truth_binary` : Ground truth binary mask.
    `test_binary` : Predicted binary mask.
    `truth_empty` : Boolean indicating if the ground truth mask is empty.
    `test_empty` : Boolean indicating if the test mask is empty.
    `metric_params` : Parameters for metric calculation.
    `resolution` : Voxel size in the array.
    """

    def jaccard(self):
        """
        Computes the jaccard coefficient.

        Returns:
        --------
        Jaccard Coefficient. If truth or test is empty , returns Not a Number.
        """

class CremiEvaluator:
    """
    The class provides Cremi score for segmentation evaluation.

    Attributes:
    -----------
    `truth` : Ground truth binary mask.
    `test` : Predicted binary mask.
    `sampling` : A tuple representing x, y, z resolution of the voxel.
    `clip_distance` : Maximum distance till where evaluation will be considered.
    `tol_distance` : Tolerance in distance while considering segmentation.
    """

    def f1_score_with_tolerance(self):
        """
        Computes F1 score with tolerance.

        Returns:
        --------
        F1 score . If truth or test is empty , returns Not a Number.
        """
    pass

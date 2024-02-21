"""
This script imports important classes from individual sub-modules into the package's root namespace which includes DummyEvaluationScores, DummyEvaluator, EvaluationScores,
Evaluator, MultiChannelBinarySegmentationEvaluationScores, BinarySegmentationEvaluationScores, BinarySegmentationEvaluator, InstanceEvaluationScores, and InstanceEvaluator.

These classes are used for different types of evaluation and scoring in the DACapo python library.

Modules:
    - dummy_evaluation_scores: Contains the definition for DummyEvaluationScores Class.
    - dummy_evaluator: Contains the definition for DummyEvaluator Class.
    - evaluation_scores: Contains the definition for EvaluationScores Class.
    - evaluator: Contains the definition for Evaluator Class.
    - binary_segmentation_evaluation_scores: Contains the definition for MultiChannelBinarySegmentationEvaluationScores and BinarySegmentationEvaluationScores Classes.
    - binary_segmentation_evaluator: Contains the definition for BinarySegmentationEvaluator Class.
    - instance_evaluation_scores: Contains the definition for InstanceEvaluationScores Class.
    - instance_evaluator: Contains the definition for InstanceEvaluator Class.

Note:
    - Import errors are ignored with `noqa` flag.
"""
from .dummy_evaluation_scores import DummyEvaluationScores  # noqa
from .dummy_evaluator import DummyEvaluator  # noqa
from .evaluation_scores import EvaluationScores  # noqa
from .evaluator import Evaluator  # noqa
from .binary_segmentation_evaluation_scores import (
    MultiChannelBinarySegmentationEvaluationScores,
    BinarySegmentationEvaluationScores,
)  # noqa
from .binary_segmentation_evaluator import BinarySegmentationEvaluator  # noqa
from .instance_evaluation_scores import InstanceEvaluationScores  # noqa
from .instance_evaluator import InstanceEvaluator  # noqa
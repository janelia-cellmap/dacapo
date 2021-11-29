from dacapo.experiments.tasks.evaluators import BinarySegmentationEvaluator

binary_segmentation_evaluator = BinarySegmentationEvaluator(clip_distance=5, tol_distance=10, channels=["a", "b", "c"])
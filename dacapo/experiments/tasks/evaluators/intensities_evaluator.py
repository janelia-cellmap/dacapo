import xarray as xr
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_root_mse

from .evaluator import Evaluator
from .intensities_evaluation_scores import IntensitiesEvaluationScores


class IntensitiesEvaluator(Evaluator):
    """IntensitiesEvaluator Class

    An evaluator takes a post-processor's output and compares it against
    ground-truth.
    """
    criteria = ["ssim", "psnr", "nrmse"]
    
    def evaluate(self, output_array_identifier, evaluation_array) -> IntensitiesEvaluationScores:
        output_array = ZarrArray.open_from_array_identifier(output_array_identifier)
        evaluation_data = evaluation_array[evaluation_array.roi].astype(np.uint64)
        output_data = output_array[output_array.roi].astype(np.uint64)
        return IntensitiesEvaluationScores(ssim=structural_similarity(evaluation_data, output_data), 
                                           psnr=peak_signal_noise_ratio(evaluation_data, output_data),
                                           nrmse=normalized_root_mse(evaluation_data, output_data))

    @property
    def score(self) -> IntensitiesEvaluationScores:
        return IntensitiesEvaluationScores()


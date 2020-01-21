from .affinities import evaluate_affs
from again.prediction_types import Affinities
from again.task_types import InstanceSegmentation

def evaluate(task_config, prediction, gt, store_results=None):

    have_labels = task_config.type == InstanceSegmentation
    predict_affs = task_config.predict == Affinities
    dims = task_config.data.dims

    if have_labels and predict_affs:
        return evaluate_affs(prediction, gt, dims, store_results=store_results)
    else:
        raise RuntimeError(
            "Evaluation other than affs versus labels not yet implemented")

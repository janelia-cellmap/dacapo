from .evaluators import InstanceEvaluator
from .losses import AffinitiesLoss
from .post_processors import WatershedPostProcessor
from .predictors import AffinitiesPredictor
from .task import Task


class AffinitiesTask(Task):
    

    def __init__(self, task_config):
        

        self.predictor = AffinitiesPredictor(
            neighborhood=task_config.neighborhood,
            lsds=task_config.lsds,
            num_voxels=task_config.num_lsd_voxels,
            downsample_lsds=task_config.downsample_lsds,
            affs_weight_clipmin=task_config.affs_weight_clipmin,
            affs_weight_clipmax=task_config.affs_weight_clipmax,
            lsd_weight_clipmin=task_config.lsd_weight_clipmin,
            lsd_weight_clipmax=task_config.lsd_weight_clipmax,
            background_as_object=task_config.background_as_object,
        )
        self.loss = AffinitiesLoss(
            len(task_config.neighborhood), task_config.lsds_to_affs_weight_ratio
        )
        self.post_processor = WatershedPostProcessor(offsets=task_config.neighborhood)
        self.evaluator = InstanceEvaluator()

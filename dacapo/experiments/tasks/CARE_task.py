#%%
import sys
sys.path.append('/Users/brianreicher/Documents/GitHub/dacapo/')

# from .evaluators import IntensitiesEvaluator
# from .losses import MSELoss
# from .post_processors import CAREPostProcessor
# from .predictors import CAREPredictor
# from .task import Task

from dacapo.experiments.tasks.predictors import CAREPredictor
from dacapo.experiments.tasks.losses import MSELoss
from dacapo.experiments.tasks.post_processors import CAREPostProcessor
from dacapo.experiments.tasks.evaluators import IntensitiesEvaluator
from dacapo.experiments.tasks import Task



class CARETask(Task):
    """CAREPredictor."""

    def __init__(self, task_config) -> None:
        """Create a `CARETask`."""

        self.predictor = CAREPredictor(num_channels=task_config.num_channels)
        self.loss = MSELoss()
        self.post_processor = CAREPostProcessor()
        self.evaluator = IntensitiesEvaluator()
#%%
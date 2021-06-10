from abc import ABC, abstractmethod

from dacapo.store import MongoDbStore


class PostProcessingStepABC(ABC):
    @abstractmethod
    def tasks(self, **kwargs):
        # Must return a list of Tasks, and a list of their respective parameters
        pass

    @property
    @abstractmethod
    def step_id(self):
        # All PostProcessingSteps must have a step_id property
        pass

    @abstractmethod
    def get_process_function(self):
        # All PostProcessingSteps must define a process function
        # see daisy process function documentation.
        pass

    def get_check_function(self, pred_id):
        # default check function is provided.
        store = MongoDbStore()
        return lambda b: store.check_block(pred_id, self.step_id, b.block_id)

from .task import Task

import torch


class PretrainedTask(Task):
    

    def __init__(self, task_config):
        
        sub_task = task_config.sub_task_config.task_type(task_config.sub_task_config)
        self.weights = task_config.weights

        self.predictor = sub_task.predictor
        self.loss = sub_task.loss
        self.post_processor = sub_task.post_processor
        self.evaluator = sub_task.evaluator

    def create_model(self, architecture):
        
        model = self.predictor.create_model(architecture)

        saved_state_dict = torch.load(str(self.weights))
        model.chain.load_state_dict(saved_state_dict["model"])
        return model

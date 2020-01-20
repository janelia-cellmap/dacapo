from .radam import RAdam
from torch.optim import *

def create_optimizer(optimizer_config, model):

    return optimizer_config.type(model.parameters(), optimizer_config.lr)

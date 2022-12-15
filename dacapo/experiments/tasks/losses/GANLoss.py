from .loss import Loss
import torch


class GANLoss(Loss):
    def __init__(self, gan_mode):
        self.gan_mode = gan_mode

    def compute(self, prediction, target, weight):            
        # match self.gan_mode: 
        #     case 'lsgan':
        #         return torch.nn.MSELoss().forward(prediction * weight, target * weight)
        #     case 'vanilla':
        #         return torch.nn.BCEWithLogitsLoss().forward(prediction * weight, target * weight)

        # if self.gan_mode in ['wgangp']:
        #     return None
        pass
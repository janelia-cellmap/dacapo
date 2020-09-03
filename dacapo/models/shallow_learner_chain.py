import torch
import funlib.learn.torch as ft
from .model import Model


class ShallowLearnerChain(Model):

    def __init__(self, task, model_config):

        # TODO: rewrite using 'task' and 'model_config'
        super(ShallowLearnerChain, self).__init__(num_latents)

        # first learner from fmaps_in to latent
        learners = [create_shallow_learner(1, num_latents)]

        for i in range(num_learners - 1):
            learners.append(
                create_shallow_learner(
                    num_latents,
                    num_latents))
        self.learners = torch.nn.ModuleList(learners)
        self.num_learners = num_learners

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, x):
        x = self.learners[0](x)
        for i in range(1, self.num_learners):
            y = self.learners[i](x)
            x = self.crop(x, y.size()) + y
        return x

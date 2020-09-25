from dacapo.evaluate import evaluate_affs
from dacapo.models import Model
from dacapo.tasks.post_processors import Watershed
import gunpowder as gp
import torch

import time


class Affinities(Model):
    def __init__(self, data, model, post_processor=None):

        assert data.gt.num_classes == 0, (
            f"Your GT has {data.gt.num_classes} classes, don't know how "
            "to get affinities out of that."
        )

        self.dims = data.raw.spatial_dims

        super(Affinities, self).__init__(model.input_shape, model.fmaps_in, self.dims)

        if self.dims == 2:
            self.neighborhood = [(0, 1, 0), (0, 0, 1)]
        elif self.dims == 3:
            self.neighborhood = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        else:
            raise RuntimeError("Affinities other than 2D/3D not implemented")

        conv = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}[self.dims]
        affs = [
            model,
            conv(model.fmaps_out, self.dims, (1,) * self.dims),
            torch.nn.Sigmoid(),
        ]

        self.affs = torch.nn.Sequential(*affs)
        self.prediction_channels = self.dims
        self.target_channels = self.dims
        if post_processor is None:
            self.post_processor = Watershed()
        else:
            self.post_processor = post_processor

    def add_target(self, gt, target):

        return (
            gp.AddAffinities(
                affinity_neighborhood=self.neighborhood, labels=gt, affinities=target
            )
            # +
            # ensure affs are float
            # gp.Normalize(target, factor=1.0)
        )

    def forward(self, x):
        affs = self.affs(x)
        return affs

    def evaluate(self, predictions, gt, targets, return_results):
        reconstructions = self.post_processor.enumerate(predictions)

        for parameters, reconstruction in reconstructions:

            print(f"Evaluation post-processing with {parameters}...")
            start = time.time()
            # This could be factored out.
            # keep evaulate as a super class method
            # over-write evaluate_reconstruction
            ret = evaluate_affs(reconstruction, gt, return_results=return_results)

            print(f"...done ({time.time() - start}s)")

        if return_results:
            scores, results = ret
            yield parameters, scores, results
        else:
            yield parameters, ret

from dacapo.evaluate import evaluate_affs
from dacapo.models import Model
import gunpowder as gp
import torch


class Affinities(Model):

    def __init__(self, data, model):

        super(Affinities, self).__init__()

        assert data.gt.num_classes == 0, (
            f"Your GT has {data.gt.num_classes} classes, don't know how "
            "to get affinities out of that.")

        self.dims = data.raw.spatial_dims

        if self.dims == 2:
            self.neighborhood = [(0, 1, 0), (0, 0, 1)]
        elif self.dims == 3:
            self.neighborhood = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        else:
            raise RuntimeError("Affinities other than 2D/3D not implemented")

        conv = {
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d
        }[self.dims]
        affs = [
            model,
            conv(model.fmaps_out, self.dims, (1,)*self.dims),
            torch.nn.Sigmoid()
        ]

        self.affs = torch.nn.Sequential(*affs)
        self.prediction_channels = self.dims
        self.target_channels = self.dims

    def add_target(self, gt, target):

        return (
            gp.AddAffinities(
                affinity_neighborhood=self.neighborhood,
                labels=gt,
                affinities=target) +
            # ensure affs are float
            gp.Normalize(target, factor=1.0)
        )

    def forward(self, x):
        return self.affs(x)

    def evaluate(self, prediction, gt, targets, store_results):
        return evaluate_affs(
            prediction,
            gt,
            self.dims,
            store_results=store_results)

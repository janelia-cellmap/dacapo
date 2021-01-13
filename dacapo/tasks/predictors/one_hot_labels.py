from dacapo.evaluate import evaluate_labels
from dacapo.models import Model
from dacapo.tasks.post_processors import ArgMax
import gunpowder as gp
import numpy as np
import time
import torch


class OneHotLabels(Model):

    def __init__(
            self,
            data,
            model,
            matching_score,
            matching_threshold,
            num_classes, 
            post_processor=None):

        dims = data.raw.spatial_dims

        super(OneHotLabels, self).__init__(
            model.output_shape,
            model.fmaps_out,            
            num_classes)

        assert num_classes > 0, (
            "Your GT has no classes, don't know how to get one-hot encoding "
            "out of that.")

        conv = {
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d
        }[dims]
        logits = [
            conv(model.fmaps_out, num_classes, (1,)*dims)
        ]

        self.logits = torch.nn.Sequential(*logits)
        self.probs = torch.nn.LogSoftmax()
        self.prediction_channels = data.gt.num_classes
        self.background_label = data.gt.background_label
        self.target_channels = 0
        self.matching_score = matching_score
        self.matching_threshold = matching_threshold
        if post_processor is None:
            self.post_processor = ArgMax()
        else:
            self.post_processor = post_processor

        self.output_channels = num_classes

    def add_target(self, gt, target):

        # target is gt, just ensure proper type
        class AddClassLabels(gp.BatchFilter):

            def __init__(self, gt, target):
                self.gt = gt
                self.target = target

            def setup(self):
                self.provides(target, self.spec[gt])
                self.enable_autoskip()

            def process(self, batch, request):
                spec = batch[self.gt].spec.copy()
                spec.dtype = np.int64
                batch[self.target] = gp.Array(
                    batch[self.gt].data.astype(np.int64),
                    spec)

        return AddClassLabels(gt, target), None

    def forward(self, x):
        logits = self.logits(x)
        if not self.training:
            return self.probs(logits)
        return logits

    def evaluate(
            self,
            predictions,
            gt,
            target,
            return_results):

        reconstructions = self.post_processor.enumerate(predictions)

        for parameters, reconstruction in reconstructions:

            # This could be factored out.
            # keep evaulate as a super class method
            # over-write evaluate_reconstruction
            ret = evaluate_labels(
                reconstruction,
                gt,
                return_results=return_results,
                background_label=self.background_label,
                matching_score=self.matching_score,
                matching_threshold=self.matching_threshold)

            if return_results:
                scores, results = ret
                yield parameters, scores, results
            else:
                yield parameters, ret

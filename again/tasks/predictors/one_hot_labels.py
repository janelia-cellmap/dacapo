from again.evaluate import evaluate_labels
from again.models import Model
from again.tasks.post_processors import ArgMax
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
            post_processor=None):

        dims = data.raw.spatial_dims
        num_classes = data.gt.num_classes

        super(OneHotLabels, self).__init__(
            model.input_shape,
            model.fmaps_in,
            num_classes)

        assert num_classes > 0, (
            "Your GT has no classes, don't know how to get one-hot encoding "
            "out of that.")

        conv = {
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d
        }[dims]
        logits = [
            model,
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
                batch[self.target] = gp.Array(
                    batch[self.gt].data.astype(np.int64),
                    batch[self.gt].spec)

        return AddClassLabels(gt, target)

    def forward(self, x):
        logits = self.logits(x)
        if not self.training:
            return self.probs(logits)
        return logits

    def evaluate(
            self,
            logits,
            gt,
            target,
            return_results=None):

        predictions = self.post_processor.enumerate(logits)

        for parameters, prediction in predictions:

            print(f"Evaluating post-processing with {parameters}...")
            start = time.time()
            ret = evaluate_labels(
                prediction,
                gt,
                return_results=return_results,
                background_label=self.background_label,
                matching_score=self.matching_score,
                matching_threshold=self.matching_threshold)
            print(f"...done ({time.time() - start}s)")

            if return_results:
                scores, results = ret
                yield parameters, scores, results
            else:
                yield parameters, ret

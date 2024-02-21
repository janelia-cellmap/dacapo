import logging
import random

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.profiling import Timing

logger = logging.getLogger(__name__)

class RejectIfEmpty(BatchFilter):
    """
    Node to reject batches based on the mask's filled vs empty ratio.

    Args:
        gt (ArrayKey, optional): The ground truth array key that will be used. 
            Default is None.
        p (float, optional): The probability threshold for rejecting batches until 
            a non-empty ground truth is found. Default is 0.5.
        background (int, optional): The value representing the background in 
            the ground truth data. Default is 0.

    This class inherits from :class: `BatchFilter`.

    In the setup() method, it asserts that only one provider is in the upstream.
    In the provide() method, it makes sure that the gt ArrayKey is in the request 
    provided. It then keeps requesting batches from the upstream until it finds 
    a batch where the ground truth is not empty, or the random number generated 
    is greater than the threshold p.
    """

    def __init__(self, gt=None, p=0.5, background=0):
        self.gt = gt
        self.p = p
        self.background = 0

    def setup(self):
        """Asserts that only one upstream provider is supported."""
        upstream_providers = self.get_upstream_providers()
        assert len(upstream_providers) == 1, "Only 1 upstream provider supported"
        self.upstream_provider = upstream_providers[0]

    def provide(self, request):
        """
        Provide the processed batch.

        Args:
            request: The batch request.

        Returns:
            Batch: The processed batch.

        Random seed is initialized based on the request's random seed. Setup the 
        timer. If there is no gt in the request, it will assert error. Continue 
        requesting batch from the upstream provider until the data's min and max 
        value is not same as background value or the random number generated is 
        less than p (the probability threshold). It returns the accepted batch 
        after stopping the timer.
        """
        random.seed(request.random_seed)

        report_next_timeout = 10
        num_rejected = 0

        timing = Timing(self)
        timing.start()

        assert self.gt in request, f"Cannot reject on {self.gt} if its not requested"

        have_good_batch = random.random() < self.p
        while True:
            batch = self.upstream_provider.request_batch(request)

            gt_data = batch.arrays[self.gt].data
            empty = (gt_data.min() == self.background) and (
                gt_data.max() == self.background
            )
            if empty and have_good_batch:
                num_rejected += 1
                logger.debug(
                    "reject empty gt at %s",
                    batch.arrays[self.gt].spec.roi,
                )
                if timing.elapsed() > report_next_timeout:
                    logger.warning(
                        "rejected %d batches, been waiting for a good one " "since %ds",
                        num_rejected,
                        report_next_timeout,
                    )
                    report_next_timeout *= 2
                if report_next_timeout > 20:
                    break
                continue
            else:
                break

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
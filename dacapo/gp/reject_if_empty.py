import logging
import random

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.profiling import Timing

logger = logging.getLogger(__name__)


class RejectIfEmpty(BatchFilter):
    

    def __init__(self, gt=None, p=0.5, background=0):
        
        self.gt = gt
        self.p = p
        self.background = 0

    def setup(self):
        
        upstream_providers = self.get_upstream_providers()
        assert len(upstream_providers) == 1, "Only 1 upstream provider supported"
        self.upstream_provider = upstream_providers[0]

    def provide(self, request):
        
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
                    f"reject empty gt at {batch.arrays[self.gt].spec.roi}",
                )
                if timing.elapsed() > report_next_timeout:
                    logger.warning(
                        f"rejected {num_rejected} batches, been waiting for a good one "
                        f"since {report_next_timeout}s",
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

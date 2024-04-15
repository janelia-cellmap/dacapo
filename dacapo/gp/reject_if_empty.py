import logging
import random

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.profiling import Timing

logger = logging.getLogger(__name__)


class RejectIfEmpty(BatchFilter):
    """
    Reject batches based on the masked-in vs. masked-out ratio.
    Attributes:
        gt (:class:`ArrayKey`, optional):
            The gt array to use
        p (``float``, optional):
            The probability that we reject until gt is nonempty
    Method:
        setup: Set up the provider.
        provide: Provide a batch.

    """

    def __init__(self, gt=None, p=0.5, background=0):
        """
        Initialize the RejectIfEmpty filter.

        Args:
            gt (:class:`ArrayKey`, optional): The gt array to use.
            p (float, optional): The probability that we reject until gt is nonempty.
            background (int, optional): The background value to consider as empty.
        Raises:
            AssertionError: If only 1 upstream provider is supported.
        Examples:
            >>> RejectIfEmpty(gt=gt, p=0.5, background=0)
            RejectIfEmpty(gt=gt, p=0.5, background=0)
        """
        self.gt = gt
        self.p = p
        self.background = 0

    def setup(self):
        """
        Set up the provider.

        Raises:
            AssertionError: If only 1 upstream provider is supported.
        Examples:
            >>> setup()
            setup()
        """
        upstream_providers = self.get_upstream_providers()
        assert len(upstream_providers) == 1, "Only 1 upstream provider supported"
        self.upstream_provider = upstream_providers[0]

    def provide(self, request):
        """
        Provides a batch of data, rejecting empty ground truth (gt) if requested.

        Args:
            request: The request object containing the necessary information.
        Returns:
            The batch of data.
        Raises:
            AssertionError: If the requested gt is not present in the request.
        Examples:
            >>> provide(request)
            provide(request)

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

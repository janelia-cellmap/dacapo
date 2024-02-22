import attr


@attr.s
class TrainingIterationStats:
    iteration: int = attr.ib(
        metadata={"help_text": "The iteration that produced these stats."}
    )
    loss: float = attr.ib(metadata={"help_text": "The loss of this iteration."})
    time: float = attr.ib(
        metadata={"help_text": "The time it took to process this iteration."}
    )

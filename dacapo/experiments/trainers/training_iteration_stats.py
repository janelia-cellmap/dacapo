import attr


@attr.s
class TrainingIterationStats:

    iteration: int = attr.ib()
    loss: float = attr.ib()
    time: float = attr.ib()

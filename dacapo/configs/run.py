import attr


@attr.s
class RunComponents:
    task: str = attr.ib()
    dataset: str = attr.ib()
    model: str = attr.ib()
    optimizer: str = attr.ib()
    name: str = attr.ib()


@attr.s
class RunExecution:
    repetitions: int = attr.ib()
    num_iterations: int = attr.ib()
    keep_best_validation: str = attr.ib()
    validation_interval: int = attr.ib(default=1000)
    snapshot_interval: int = attr.ib(default=0)
    bsub_flags: str = attr.ib(default="")
    batch: bool = attr.ib(default=True)

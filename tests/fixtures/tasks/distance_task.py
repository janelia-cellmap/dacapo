from dacapo.experiments.tasks import DistanceTaskConfig

distance_task_config = DistanceTaskConfig(
    name="distance_task",
    channels=[
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    ],
    clip_distance=5,
    tol_distance=10,
)

from dacapo.experiments.tasks import DistanceTaskConfig

distance_task_config = DistanceTaskConfig(
    name="distance_task",
    channels=[
        ("0", [0]),
        ("1", [1, 2]),
        ("2", [3, 4, 5]),
        ("3", [6, 7, 8, 9]),
        ("4", [10, 11, 0, 1, 2]),
        ("5", [3, 4, 5, 6, 7, 8]),
    ],
)

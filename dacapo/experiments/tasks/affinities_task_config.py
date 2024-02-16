@attr.s
class AffinitiesTaskConfig(TaskConfig):
    """
    Defines parameters required for affinity task configuration in the funkelab dacapo library.
    Contains parameters for handling voxel affinities for instance segmentations.

    Attributes:
        task_type: a task type object from the AffinitiesTask class.
        neighborhood (List[Coordinate]): A list of offsets to calculate affinities.
        lsds (bool): Flag to determine if to train lsds along with affinities.
        lsds_to_affs_weight_ratio (float): Weightage value for lsds compared with affs.
        affs_weight_clipmin (float): Minimum clipping point for affinity weights.
        affs_weight_clipmax (float): Maximum clipping point for affinity weights.
        lsd_weight_clipmin (float): Minimum clipping point for lsd weights.
        lsd_weight_clipmax (float): Maximum clipping point for lsd weights.
        background_as_object (bool): Flag that determines whether the background is treated as a separate object.
    """

    task_type = AffinitiesTask

    neighborhood: List[Coordinate] = attr.ib(
        metadata={
            "help_text": "The neighborhood upon which to calculate affinities."
        }
    )
    lsds: bool = attr.ib(
        default=False,
        metadata={
            "help_text": "Whether to train lsds with affinities."
        },
    )
    lsds_to_affs_weight_ratio: float = attr.ib(
        default=1,
        metadata={
            "help_text": "The weightage for lsds to affinities."
        },
    )
    affs_weight_clipmin: float = attr.ib(
        default=0.05,
        metadata={"help_text": "The minimum value for affinities weights."},
    )
    affs_weight_clipmax: float = attr.ib(
        default=0.95,
        metadata={"help_text": "The maximum value for affinities weights."},
    )
    lsd_weight_clipmin: float = attr.ib(
        default=0.05,
        metadata={"help_text": "The minimum value for lsds weights."},
    )
    lsd_weight_clipmax: float = attr.ib(
        default=0.95,
        metadata={"help_text": "The maximum value for lsds weights."},
    )
    background_as_object: bool = attr.ib(
        default=False,
        metadata={
            "help_text": (
                "Whether to treat the background as a distinct object."
            )
        },
    )
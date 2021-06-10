from gunpowder.array_spec import ArraySpec
from funlib.geometry import Coordinate, Roi
import attr
from intern import array

from dacapo.gp.bossdb import BossDB

from typing import List, Optional


@attr.s
class BossDBSource:
    """
    The class representing a single zarr dataset
    """

    db_name: str = attr.ib()

    axes: Optional[List[str]] = attr.ib(default=None)
    voxel_size: Optional[Coordinate] = attr.ib(default=None)
    offset: Optional[Coordinate] = attr.ib(default=None)
    # roi: Optional[Roi] = attr.ib(default=None)

    def __attrs_post_init__(self):
        db = array(self.db_name)
        self.shape = Coordinate(db.shape)
        assert self.voxel_size is None or self.voxel_size == Coordinate(
            db.voxel_size[0]
        )
        self.voxel_size = Coordinate(db.voxel_size[0])
        self.units = db.voxel_size[1]  # bossdb provides tuple (voxel_size, units: str)
        if self.offset is None:
            self.offset = Coordinate((0,) * self.voxel_size.dims())
        if len(self.shape) == len(self.voxel_size):
            self.num_channels = 1
        elif len(self.shape) == len(self.voxel_size):
            self.num_channels = self.shape[0]

    @property
    def dims(self):
        return self.voxel_size.dims()

    @property
    def roi(self):
        return Roi(self.offset, self.voxel_size * self.shape)

    def get_source(self, array, overwrite_spec=None):
        spec = ArraySpec(
            roi=Roi(self.offset, self.shape * self.voxel_size),
            voxel_size=self.voxel_size,
        )
        return BossDB(
            array,
            self.db_name,
            spec,
        )

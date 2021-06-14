import gunpowder as gp
import numpy as np

from intern import array


class BossDB(gp.BatchProvider):
    def __init__(self, out_array, db_name, override_spec=None):
        self.out_array = out_array
        self.array_spec = override_spec
        self.db_name = db_name

    def setup(self):
        self.db = array(self.db_name)
        self.update_spec()
        self.provides(self.out_array, self.array_spec)

    def update_spec(self):

        voxel_size = gp.Coordinate(self.db.voxel_size[0])
        shape = gp.Coordinate(self.db.shape[-voxel_size.dims :])

        if self.array_spec is None:
            self.array_spec = gp.ArraySpec(
                gp.Roi((0,) * voxel_size.dims, shape * voxel_size),
                voxel_size=voxel_size,
            )
        else:
            self.array_spec.roi.shape = shape * voxel_size
            self.array_spec.voxel_size = voxel_size

    def provide(self, request):
        output = gp.Batch()
        roi = request[self.out_array].roi
        voxel_roi = roi / self.array_spec.voxel_size

        out_data = self.db[voxel_roi.to_slices()]
        out_spec = self.array_spec.copy()
        out_spec.roi = roi
        output[self.out_array] = gp.Array(out_data, out_spec)
        return output

import gunpowder as gp

class Squash(gp.BatchFilter):
    '''Remove a dimension of arrays in batches passing through this node.
    Assumes that the shape of this dimension is 1.

    Args:

        dim (int, optional):

              The dimension to remove (defaults to 0, i.e., the first
              dimension).
    '''

    def __init__(self, dim=0):
        self.dim = dim

    def setup(self):

        # remove selected dim from provided specs
        for key, upstream_spec in self.get_upstream_provider().spec.items():
            spec = upstream_spec.copy()
            spec.roi = gp.Roi(
                self.__remove_dim(spec.roi.get_begin()),
                self.__remove_dim(spec.roi.get_shape()))
            spec.voxel_size = self.__remove_dim(spec.voxel_size)
            self.spec[key] = spec

    def prepare(self, request):

        upstream_spec = self.get_upstream_provider().spec

        # add a new dim
        for key, spec in request.items():
            upstream_voxel_size = upstream_spec[key].voxel_size
            v = upstream_voxel_size[self.dim]
            spec.roi = gp.Roi(
                self.__insert_dim(spec.roi.get_begin(), 0),
                self.__insert_dim(spec.roi.get_shape(), v))
            if spec.voxel_size is not None:
                spec.voxel_size = self.__insert_dim(spec.voxel_size, v)

    def process(self, batch, request):

        for key, array in batch.arrays.items():

            # remove first dim
            array.spec.roi = gp.Roi(
                self.__remove_dim(array.spec.roi.get_begin()),
                self.__remove_dim(array.spec.roi.get_shape()))
            array.spec.voxel_size = self.__remove_dim(array.spec.voxel_size)
            assert array.data.shape[self.dim] == 1, \
                "Squash for dim %d requires that the array %s has size 1 in " \
                "that dim, but array shape is %s" % (
                    self.dim,
                    key,
                    array.data.shape)
            array.data = array.data.reshape(
                self.__remove_dim(array.data.shape))

    def __remove_dim(self, a):
        return a[:self.dim] + a[self.dim + 1:]

    def __insert_dim(self, a, s):
        return a[:self.dim] + (s,) + a[self.dim:]

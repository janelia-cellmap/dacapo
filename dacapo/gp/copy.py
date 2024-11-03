import gunpowder as gp


class CopyMask(gp.BatchFilter):
    

    def __init__(
        self, array_key: gp.ArrayKey, copy_key: gp.ArrayKey, drop_channels: bool = False
    ):
        
        self.array_key = array_key
        self.copy_key = copy_key
        self.drop_channels = drop_channels

    def setup(self):
        
        self.enable_autoskip()
        self.provides(self.copy_key, self.spec[self.array_key].copy())

    def prepare(self, request):
        
        deps = gp.BatchRequest()
        deps[self.array_key] = request[self.copy_key].copy()
        return deps

    def process(self, batch, request):
        
        outputs = gp.Batch()

        outputs[self.copy_key] = batch[self.array_key]
        if self.drop_channels:
            while (
                outputs[self.copy_key].data.ndim
                > outputs[self.copy_key].spec.voxel_size.dims
            ):
                outputs[self.copy_key].data = outputs[self.copy_key].data.max(axis=0)

        return outputs

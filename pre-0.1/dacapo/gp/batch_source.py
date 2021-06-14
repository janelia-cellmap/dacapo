import gunpowder as gp


class BatchSource(gp.BatchProvider):
    def __init__(self, batch):
        self.batch = batch

    def setup(self):
        for k, v in self.batch.items():
            self.provides(k, v.spec.copy())

    def provide(self, request):
        output = gp.Batch()

        for key, spec in request.items():
            output[key] = self.batch[key].crop(spec.roi)

        return output

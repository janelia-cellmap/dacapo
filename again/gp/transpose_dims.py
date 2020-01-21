import gunpowder as gp

class TransposeDims(gp.BatchFilter):

    def __init__(self, array, permutation):
        self.array = array
        self.permutation = permutation

    def process(self, batch, request):

        batch.arrays[self.array].data = \
            batch.arrays[self.array].data.transpose(self.permutation)

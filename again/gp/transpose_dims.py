import gunpowder as gp

class TransposeDims(gp.BatchFilter):

    def __init__(self, permutation):
        self.permutation = permutation

    def process(self, batch, request):

        for key, array in batch.arrays.items():
            array.data = array.data.transpose(self.permutation)

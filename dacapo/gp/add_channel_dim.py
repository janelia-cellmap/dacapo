import gunpowder as gp
import numpy as np

class AddChannelDim(gp.BatchFilter):

    def __init__(self, array):
        self.array = array

    def process(self, batch, request):

        if self.array not in batch:
            return

        batch[self.array].data = batch[self.array].data[np.newaxis]

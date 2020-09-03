import gunpowder as gp
import numpy as np


class ArgMax:

    def process(self, prediction, parameters):
        return gp.Array(
            np.argmax(prediction.data, axis=0),
            spec=prediction.spec)

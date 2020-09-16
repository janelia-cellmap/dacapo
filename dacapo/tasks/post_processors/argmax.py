import gunpowder as gp
import numpy as np

from .post_processor import PostProcessor


class ArgMax(PostProcessor):

    def process(self, prediction, parameters):
        return gp.Array(
            np.argmax(prediction.data, axis=0),
            spec=prediction.spec)

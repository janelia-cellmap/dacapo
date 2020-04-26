import numpy as np


def sanatize(d):
    '''Ensure numpy datatypes in a dictionary are converted to float/int for
    JSON serialization.'''

    sanatized = {}
    for k, v in d.items():
        try:
            if isinstance(v, dict):
                sanatized[k] = sanatize(v)
            elif isinstance(v, np.ndarray):
                assert len(v.shape) == 1, "Can only store 1d arrays"
                sanatized[k] = list([x.item() for x in v])
            else:
                sanatized[k] = v.item()
        except AttributeError:
            # neither a dict nor a numpy type, leave as is
            sanatized[k] = v
    return sanatized

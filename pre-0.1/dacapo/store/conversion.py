import numpy as np

# TODO: delete this if it is really unnecessary
def sanatize(d):
    """Ensure numpy datatypes in a dictionary are converted to float/int for
    JSON serialization."""

    sanatized = {}
    for k, v in d.items():
        try:
            if isinstance(v, dict):
                sanatized[k] = sanatize(v)
            else:
                # what is this for?
                sanatized[k] = v.item()
        except AttributeError:
            # neither a dict nor a numpy type, leave as is
            sanatized[k] = v
    return sanatized

__version__ = "0.3.5"
__version_info__ = tuple(int(i) for i in __version__.split("."))

from .options import Options  # noqa

# Import modules conditionally to handle missing dependencies gracefully
try:
    from . import experiments, utils  # noqa
    from .apply import apply  # noqa
    from .train import train  # noqa
    from .validate import validate, validate_run  # noqa
    from .predict import predict  # noqa
    from .blockwise import run_blockwise, segment_blockwise  # noqa
    from . import predict_local
except ImportError as e:
    import warnings
    warnings.warn(f"Some dacapo modules unavailable due to missing dependencies: {e}")
    # Make core functionality available even if full features are not
    experiments = None
    utils = None

from .run_config import RunConfig
import warnings


class Run:
    def __new__(cls, run_config: RunConfig):
        warnings.warn(
            "Run is depricated and doesn't need to be used", DeprecationWarning
        )
        return run_config

"""
This module imports the components of the funkelab dacapo python library which are required
to build models and run configurations. It also includes functionalities to perform training,
validation and retrieving statistics or scores from these processes.

This includes:

    - Definition and structure of the Model.
    - Configuration and execution of a Run.
    - Settings and preferences for a Run through RunConfig.
    - Extraction of statistics from each iteration in a training through TrainingIterationStats.
    - Overall statistics from a full training session through TrainingStats.
    - Scores from each iteration in validation through ValidationIterationScores.
    - Overall scores from a full validation session through ValidationScores.
"""

from .model import Model  # noqa
"""
Defining the structure and methods for Model in the library
"""

from .run import Run  # noqa
"""
Defining the structure and methods for Run in the library. This includes setting up a run, execution and returning results.
"""

from .run_config import RunConfig  # noqa
"""
Defining the settings and configurations available for use during a run.
"""

from .training_iteration_stats import TrainingIterationStats  # noqa
"""
Provides functionalities to extract and present statistics from each training iteration during a run.
"""

from .training_stats import TrainingStats  # noqa
"""
Provides functionalities to extract and present overall training statistics from a complete run.
"""

from .validation_iteration_scores import ValidationIterationScores  # noqa
"""
Provides functionalities to extract and present scores from each validation iteration during a run.
"""

from .validation_scores import ValidationScores  # noqa
"""
Provides functionalities to extract and present overall validation scores from a complete run.
"""

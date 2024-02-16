```python
import json
from bokeh.embed.standalone import json_item
from dacapo.store.create_store import create_config_store, create_stats_store
from dacapo.experiments.run import Run

from bokeh.palettes import Category20 as palette
import bokeh.layouts
import bokeh.plotting
import numpy as np

from collections import namedtuple
import itertools
from typing import List

RunInfo = namedtuple(
    "RunInfo",
    [
        "name",
        "task",
        "architecture",
        "trainer",
        "datasplit",
        "training_stats",
        "validation_scores",
        "validation_score_name",
        "plot_loss",
    ],
)

def smooth_values(a, n, stride=1):
    """
    Function to smooth the given values using standard deviation.
    
    Args:
        a (np.array): Array of values to smooth.
        n (int): The window size for the moving average smoothing.
        stride (int, optional): The stride length to use. Defaults to 1.

    Returns:
        Tuple: Contains the smoothed values.
    """ 

def get_runs_info(
    run_config_names: List[str],
    validation_score_names: List[str],
    plot_losses: List[bool],
) -> List[RunInfo]:
    """
    Function to get the information of runs.

    Args:
        run_config_names (List[str]): List of run configuration names.
        validation_score_names (List[str]): List of validation score names.
        plot_losses (List[bool]): List of boolean values indicating whether to plot loss or not. 

    Returns:
        List[RunInfo]: List containing RunInfo for each run.
    """

def plot_runs(
    run_config_base_names,
    smooth=100,
    validation_scores=None,
    higher_is_betters=None,
    plot_losses=None,
    return_json=False,
):
    """
    Function to plot runs.

    Args:
        run_config_base_names (List[str]): List of run configuration base names.
        smooth (int, optional): Smoothing factor. Defaults to 100.
        validation_scores (List[str], optional): List of validation scores. Defaults to None.
        higher_is_betters (bool, optional): Boolean indicating higher value is better. Defaults to None.
        plot_losses (bool, optional): Boolean indicating whether to plot losses. Defaults to None.
        return_json (bool, optional): Boolean indicating whether to return the plot as JSON. Defaults to False.

    Returns:
        JSON or Plot: Returns JSON or Plots based on the return_json flag.
    """ 
```

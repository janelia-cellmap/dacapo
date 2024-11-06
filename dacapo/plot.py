import json
from bokeh.embed.standalone import json_item
from dacapo.store.create_store import create_config_store, create_stats_store
from dacapo.experiments.run import Run

from bokeh.palettes import Category20 as palette
import bokeh.layouts
import bokeh.plotting
import numpy as np
from tqdm import tqdm

from collections import namedtuple
import itertools
from typing import List
import matplotlib.pyplot as plt


import os


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
    Smooth values with a moving average.

    Args:
        a: values to smooth
        n: number of values to average
        stride: stride of the smoothing
    Returns:
        m: smoothed values
        s: standard deviation of the smoothed values
    Raises:
        ValueError: If run_name is not found in config store
    Examples:
        >>> smooth_values([1,2,3,4,5], 3)
    """
    a = np.array(a)

    # mean
    m = np.cumsum(a)
    m[n:] = m[n:] - m[:-n]
    m = m[n - 1 :] / n

    # mean of squared values
    m2 = np.cumsum(a**2)
    m2[n:] = m2[n:] - m2[:-n]
    m2 = m2[n - 1 :] / n

    # stddev
    s = m2 - m**2

    if stride > 1:
        m = m[::stride]
        s = s[::stride]

    return m, s


def get_runs_info(
    run_config_names: List[str],
    validation_score_names: List[str],
    plot_losses: List[bool],
) -> List[RunInfo]:
    config_store = create_config_store()
    stats_store = create_stats_store()
    runs = []
    """
    Get information about runs for plotting.

    Args:
        run_config_names: Names of run configs to plot
        validation_score_names: Names of validation scores to plot
        plot_losses: Whether to plot losses
    Returns:
        runs: List of RunInfo objects  
    Raises:
        ValueError: If run_name is not found in config store
    Examples:
        >>> get_runs_info(["run_name"], ["validation_score_name"], [True])
    """

    for run_config_name, validation_score_name, plot_loss in zip(
        run_config_names, validation_score_names, plot_losses
    ):
        run_config = config_store.retrieve_run_config(run_config_name)
        validation_scores = Run.get_validation_scores(run_config)
        validation_scores.scores = stats_store.retrieve_validation_iteration_scores(
            run_config_name
        )
        run = RunInfo(
            run_config_name,
            run_config.task_config.name,
            run_config.architecture_config.name,
            run_config.trainer_config.name,
            run_config.datasplit_config.name,
            (
                stats_store.retrieve_training_stats(run_config_name)
                if plot_loss
                else None
            ),
            validation_scores,
            validation_score_name,
            plot_loss,
        )
        runs.append(run)

    return runs


def bokeh_plot_runs(
    run_config_base_names,
    smooth=100,
    validation_scores=None,
    higher_is_betters=None,
    plot_losses=None,
    return_json=False,
):
    """
    Plot runs.
    Args:
        run_config_base_names: Names of run configs to plot
        smooth: Smoothing factor
        validation_scores: Validation scores to plot
        higher_is_betters: Whether higher is better
        plot_losses: Whether to plot losses
        return_json: Whether to return JSON
    Returns:
        JSON or HTML plot
    Raises:
        ValueError: If run_name is not found in config store
    Examples:
        >>> plot_runs(["run_name"], 100, None, None, [True])

    """
    runs = get_runs_info(run_config_base_names, validation_scores, plot_losses)

    colors = itertools.cycle(palette[20])
    loss_tooltips = [
        ("task", "@task"),
        ("architecture", "@architecture"),
        ("trainer", "@trainer"),
        ("datasplit", "@datasplit"),
        ("iteration", "@iteration"),
        ("loss", "@loss"),
    ]
    loss_figure = bokeh.plotting.figure(
        tools="pan, wheel_zoom, reset, save, hover",
        x_axis_label="iterations",
        tooltips=loss_tooltips,
        # plot_width=2048,
    )
    loss_figure.background_fill_color = "#efefef"

    validation_figures = {}
    validation_datasets = set(
        itertools.chain(*[list(run.validation_scores.datasets) for run in runs])
    )

    if validation_scores:
        validation_score_names = set()
        validation_postprocessor_parameter_names = set()
        for r in runs:
            if r.validation_scores.validated_until() > 0:
                validation_score_names = validation_score_names.union(
                    r.validation_scores.criteria
                )
                validation_postprocessor_parameter_names = (
                    validation_postprocessor_parameter_names.union(
                        set(r.validation_scores.parameter_names)
                    )
                )
        validation_score_names = validation_score_names
        validation_postprocessor_parameter_names = (
            validation_postprocessor_parameter_names
        )

        validation_tooltips = (
            [
                ("run", "@run"),
                ("task", "@task"),
                ("architecture", "@architecture"),
                ("trainer", "@trainer"),
                ("datasplit", "@datasplit"),
            ]
            + [(name, "@" + name) for name in validation_score_names]
            + [(name, "@" + name) for name in validation_postprocessor_parameter_names]
        )
        for dataset in validation_datasets:
            validation_figure = bokeh.plotting.figure(
                tools="pan, wheel_zoom, reset, save, hover",
                x_axis_label="iterations",
                tooltips=validation_tooltips,
                # plot_width=2048,
            )
            validation_figure.background_fill_color = "#efefef"
            validation_figures[dataset.name] = validation_figure

    summary_tooltips = [
        ("run", "@run"),
        ("task", "@task"),
        ("architecture", "@architecture"),
        ("trainer", "@trainer"),
        ("datasplit", "@datasplit"),
        ("best iteration", "@iteration"),
        ("best voi_split", "@voi_split"),
        ("best voi_merge", "@voi_merge"),
        ("best voi_sum", "@voi_sum"),
        ("num parameters", "@num_parameters"),
    ]
    summary_figure = bokeh.plotting.figure(
        tools="pan, wheel_zoom, reset, save, hover",
        x_axis_label="model size",
        y_axis_label="best validation",
        tooltips=summary_tooltips,
        # plot_width=2048,
    )
    summary_figure.background_fill_color = "#efefef"

    include_validation_figure = False
    include_loss_figure = False

    for run, color in zip(runs, colors):
        name = run.name

        if run.plot_loss:
            iterations = [stat.iteration for stat in run.training_stats.iteration_stats]
            losses = [stat.loss for stat in run.training_stats.iteration_stats]

            if run.plot_loss:
                include_loss_figure = True
                smooth = int(np.maximum(len(iterations) / 2500, 1))
                x, _ = smooth_values(iterations, smooth, stride=smooth)
                y, s = smooth_values(losses, smooth, stride=smooth)
                source = bokeh.plotting.ColumnDataSource(
                    {
                        "iteration": x,
                        "loss": y,
                        "task": [run.task] * len(x),
                        "architecture": [run.architecture] * len(x),
                        "trainer": [run.trainer] * len(x),
                        "datasplit": [run.datasplit] * len(x),
                        "run": [name] * len(x),
                    }
                )
                loss_figure.line(
                    "iteration",
                    "loss",
                    legend_label=name,
                    source=source,
                    color=color,
                    alpha=0.7,
                )

                loss_figure.patch(
                    np.concatenate([x, x[::-1]]),
                    np.concatenate([y + 3 * s, (y - 3 * s)[::-1]]),
                    legend_label=name,
                    color=color,
                    alpha=0.3,
                )

        if run.validation_score_name and run.validation_scores.validated_until() > 0:
            validation_score_data = run.validation_scores.to_xarray().sel(
                criteria=run.validation_score_name
            )
            for dataset in run.validation_scores.datasets:
                dataset_data = validation_score_data.sel(datasets=dataset)
                include_validation_figure = True
                x = [score.iteration for score in run.validation_scores.scores]
                source_dict = {
                    "iteration": x,
                    "task": [run.task] * len(x),
                    "architecture": [run.architecture] * len(x),
                    "trainer": [run.trainer] * len(x),
                    "datasplit": [run.datasplit] * len(x),
                    "run": [run.name] * len(x),
                }
                # TODO: get_best: higher_is_better is not true for all scores
                # best_parameters, best_scores = run.validation_scores.get_best(
                #     dataset_data, dim="parameters"
                # )

                # source_dict.update(
                #     {
                #         name: np.array(
                #             [
                #                 getattr(best_parameter, name)
                #                 for best_parameter in best_parameters.values
                #             ]
                #         )
                #         for name in run.validation_scores.parameter_names
                #     }
                # )
                # source_dict.update(
                #     {run.validation_score_name: np.array(best_scores.values)}
                # )

                source = bokeh.plotting.ColumnDataSource(source_dict)
                validation_figures[dataset.name].line(
                    "iteration",
                    run.validation_score_name,
                    legend_label=name + " " + run.validation_score_name,
                    source=source,
                    color=color,
                    alpha=0.7,
                )

    # Styling
    # training
    figures = []
    if include_loss_figure:
        loss_figure.title.text_font_size = "25pt"
        loss_figure.title.text = "Training"
        loss_figure.title.align = "center"

        loss_figure.legend.label_text_font_size = "16pt"

        loss_figure.xaxis.axis_label = "Iterations"
        loss_figure.xaxis.axis_label_text_font_size = "20pt"
        loss_figure.xaxis.major_label_text_font_size = "16pt"
        loss_figure.xaxis.axis_label_text_font = "times"
        loss_figure.xaxis.axis_label_text_color = "black"

        loss_figure.yaxis.axis_label = "Loss"
        loss_figure.yaxis.axis_label_text_font_size = "20pt"
        loss_figure.yaxis.major_label_text_font_size = "16pt"
        loss_figure.yaxis.axis_label_text_font = "times"
        loss_figure.yaxis.axis_label_text_color = "black"
        loss_figure.sizing_mode = "scale_width"
        figures.append(loss_figure)

    if include_validation_figure:
        for dataset, validation_figure in validation_figures.items():
            # validation
            validation_figure.title.text_font_size = "25pt"
            validation_figure.title.text = f"{dataset} Validation"
            validation_figure.title.align = "center"

            validation_figure.legend.label_text_font_size = "16pt"

            validation_figure.xaxis.axis_label = "Iterations"
            validation_figure.xaxis.axis_label_text_font_size = "20pt"
            validation_figure.xaxis.major_label_text_font_size = "16pt"
            validation_figure.xaxis.axis_label_text_font = "times"
            validation_figure.xaxis.axis_label_text_color = "black"

            validation_figure.yaxis.axis_label = "Validation Score"
            validation_figure.yaxis.axis_label_text_font_size = "20pt"
            validation_figure.yaxis.major_label_text_font_size = "16pt"
            validation_figure.yaxis.axis_label_text_font = "times"
            validation_figure.yaxis.axis_label_text_color = "black"
            validation_figure.sizing_mode = "scale_width"
            figures.append(validation_figure)

    plot = bokeh.layouts.column(*figures)
    plot.sizing_mode = "scale_width"

    if return_json:
        print("Returning JSON")
        return json.dumps(json_item(plot, "myplot"))
    else:
        bokeh.plotting.output_file("performance_plots.html")
        bokeh.plotting.save(plot)


def plot_runs(
    run_config_base_names,
    smooth=100,
    validation_scores=None,
    higher_is_betters=None,
    plot_losses=None,
):
    """
    Plot runs.
    Args:
        run_config_base_names: Names of run configs to plot
        smooth: Smoothing factor
        validation_scores: Validation scores to plot
        higher_is_betters: Whether higher is better
        plot_losses: Whether to plot losses
    Returns:
        None
    """
    runs = get_runs_info(run_config_base_names, validation_scores, plot_losses)

    colors = itertools.cycle(plt.cm.tab20.colors)
    include_validation_figure = False
    include_loss_figure = False

    fig, axis_names = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    loss_ax = axis_names[0]
    validation_ax = axis_names[1]

    for run, color in zip(runs, colors):
        name = run.name

        if run.plot_loss:
            iterations = [stat.iteration for stat in run.training_stats.iteration_stats]
            losses = [stat.loss for stat in run.training_stats.iteration_stats]

            if run.plot_loss:
                include_loss_figure = True
                smooth = int(np.maximum(len(iterations) / 2500, 1))
                x, _ = smooth_values(iterations, smooth, stride=smooth)
                y, s = smooth_values(losses, smooth, stride=smooth)
                loss_ax.plot(x, y, label=name, color=color)

        if run.validation_score_name and run.validation_scores.validated_until() > 0:
            validation_score_data = run.validation_scores.to_xarray().sel(
                criteria=run.validation_score_name
            )
            colors_val = itertools.cycle(plt.cm.tab20.colors)
            for dataset in run.validation_scores.datasets:
                dataset_data = validation_score_data.sel(datasets=dataset)
                include_validation_figure = True
                x = [score.iteration for score in run.validation_scores.scores]
                for i, cc in zip(range(dataset_data.data.shape[1]), colors_val):
                    current_name = f"{i}_{dataset.name}"
                    validation_ax.plot(
                        x,
                        dataset_data.data[:, i],
                        label=current_name,
                        color=cc,
                        # alpha=0.5 + 0.2 * i,
                    )

    if include_loss_figure:
        loss_ax.set_title("Training")
        loss_ax.set_xlabel("Iterations")
        loss_ax.set_ylabel("Loss")
        loss_ax.legend()

    if include_validation_figure:
        validation_ax.set_title("Validation")
        validation_ax.set_xlabel("Iterations")
        validation_ax.set_ylabel("Validation Score")
        validation_ax.legend()

    plt.tight_layout()
    plt.show()

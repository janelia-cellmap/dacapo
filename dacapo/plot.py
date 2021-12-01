import json
from bokeh.embed.standalone import json_item
from dacapo.store.create_store import create_config_store, create_stats_store
from bokeh.palettes import Category20 as palette
import bokeh.layouts
import bokeh.plotting
import itertools
import numpy as np
from collections import namedtuple


def smooth_values(a, n, stride=1):

    a = np.array(a)

    # mean
    m = np.cumsum(a)
    m[n:] = m[n:] - m[:-n]
    m = m[n - 1:] / n

    # mean of squared values
    m2 = np.cumsum(a ** 2)
    m2[n:] = m2[n:] - m2[:-n]
    m2 = m2[n - 1:] / n

    # stddev
    s = m2 - m ** 2

    if stride > 1:
        m = m[::stride]
        s = s[::stride]

    return m, s


def get_runs_info(run_config_base_names,
                  validation_score_names,
                  higher_is_betters,
                  plot_losses):

    config_store = create_config_store()
    stats_store = create_stats_store()
    runs = []

    RunInfo = namedtuple("run_info",
                         ["name",
                          "task",
                          "architecture",
                          "trainer",
                          "datasplit",
                          "training_stats",
                          "validation_scores",
                          "validation_score_name",
                          "higher_is_better",
                          "plot_loss"])

    all_run_config_names = config_store.retrieve_run_config_names()
    for run_config_name in all_run_config_names:
        run_config_base_name = run_config_name.split(":")[0]
        if run_config_base_name in run_config_base_names:
            idx = run_config_base_names.index(run_config_base_name)
            validation_score_name = validation_score_names[idx]
            higher_is_better = higher_is_betters[idx]
            plot_loss = plot_losses[idx]

            run_config = config_store.retrieve_run_config(run_config_name)
            run = RunInfo(run_config_name,
                          run_config.task_config.name,
                          run_config.architecture_config.name,
                          run_config.trainer_config.name,
                          run_config.datasplit_config.name,
                          stats_store.retrieve_training_stats(
                              run_config_name) if plot_loss else None,
                          stats_store.retrieve_validation_scores(
                              run_config_name),
                          validation_score_name,
                          higher_is_better,
                          plot_loss
                          )
            runs.append(run)

    return runs


def plot_runs(run_config_base_names, smooth=100, validation_scores=None, higher_is_betters=None, plot_losses=None, return_json=False):
    runs = get_runs_info(run_config_base_names,
                         validation_scores, higher_is_betters, plot_losses)

    colors = itertools.cycle(palette[20])
    loss_tooltips = [
        ("task", "@task"),
        ("architecture", "@architecture"),
        ("trainer", "@trainer"),
        ("datasplit",  "@datasplit"),
        ("iteration", "@iteration"),
        ("loss", "@loss"),
    ]
    loss_figure = bokeh.plotting.figure(
        tools="pan, wheel_zoom, reset, save, hover",
        x_axis_label="iterations",
        tooltips=loss_tooltips,
        plot_width=2048,
    )
    loss_figure.background_fill_color = "#efefef"

    if validation_scores:
        validation_score_names = []
        validation_postprocessor_parameter_names = []
        for r in runs:
            if r.validation_scores.validated_until() > 0:
                validation_score_names += r.validation_scores.get_score_names()
                validation_postprocessor_parameter_names += r.validation_scores.get_postprocessor_parameter_names()

        validation_score_names = np.unique(validation_score_names)
        validation_postprocessor_parameter_names = np.unique(
            validation_postprocessor_parameter_names)

        validation_tooltips = [
            ("run", "@run"),
            ("task", "@task"),
            ("architecture", "@architecture"),
            ("trainer", "@trainer"),
            ("datasplit",  "@datasplit"),
        ] + [(name, "@" + name) for name in validation_score_names] \
            + [(name, "@" + name)
                for name in validation_postprocessor_parameter_names]

        validation_figure = bokeh.plotting.figure(
            tools="pan, wheel_zoom, reset, save, hover",
            x_axis_label="iterations",
            tooltips=validation_tooltips,
            plot_width=2048,
        )
        validation_figure.background_fill_color = "#efefef"

    summary_tooltips = [
        ("run", "@run"),
        ("task", "@task"),
        ("architecture", "@architecture"),
        ("trainer", "@trainer"),
        ("datasplit",  "@datasplit"),
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
        plot_width=2048,
    )
    summary_figure.background_fill_color = "#efefef"

    include_validation_figure = False
    include_loss_figure = False

    for run, color in zip(runs, colors):
        name = run.name

        if run.plot_loss:
            iterations = [stat.iteration
                          for stat in run.training_stats.iteration_stats]
            losses = [stat.loss
                      for stat in run.training_stats.iteration_stats]

            if run.plot_loss:
                include_loss_figure = True
                smooth = int(np.maximum(len(iterations)/2500, 1))
                x, _ = smooth_values(
                    iterations, smooth, stride=smooth)
                y, s = smooth_values(losses,
                                     smooth, stride=smooth)
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
            include_validation_figure = True
            x = [score.iteration
                 for score in run.validation_scores.iteration_scores]
            source_dict = {
                "iteration": x,
                "task": [run.task] * len(x),
                "architecture": [run.architecture] * len(x),
                "trainer": [run.trainer] * len(x),
                "datasplit": [run.datasplit] * len(x),
                "run": [run.name] * len(x),
            }
            # TODO: get_best: higher_is_better is not true for all scores
            validation_bests = run.validation_scores.get_best(
                run.validation_score_name, higher_is_better=run.higher_is_better
            )
            best_validation_parameters = validation_bests[0]
            best_validation_scores = validation_bests[1]

            source_dict.update(
                {
                    name: np.array(best_validation_parameters[name])
                    for name in run.validation_scores.get_postprocessor_parameter_names()
                }
            )
            source_dict.update(
                {
                    name: np.array(best_validation_scores[name])
                    for name in run.validation_scores.get_score_names()
                }
            )

            source = bokeh.plotting.ColumnDataSource(source_dict)
            validation_figure.line(
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
        loss_figure.sizing_mode = 'scale_width'
        figures.append(loss_figure)

    if include_validation_figure:
        # validation
        validation_figure.title.text_font_size = "25pt"
        validation_figure.title.text = "Validation"
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
        validation_figure.sizing_mode = 'scale_width'
        figures.append(validation_figure)

    plot = bokeh.layouts.column(*figures)
    plot.sizing_mode = 'scale_width'

    if return_json:
        return json.dumps(json_item(plot, "myplot"))
    else:
        bokeh.plotting.output_file("performance_plots.html")
        bokeh.plotting.save(plot)

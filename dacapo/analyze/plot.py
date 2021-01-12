from bokeh.palettes import Category20 as palette
import bokeh.layouts
import bokeh.plotting
import itertools
import numpy as np


def smooth_values(a, n, stride=1):

    a = np.array(a)

    # mean
    m = np.cumsum(a)
    m[n:] = m[n:] - m[:-n]
    m = m[n - 1 :] / n

    # mean of squared values
    m2 = np.cumsum(a ** 2)
    m2[n:] = m2[n:] - m2[:-n]
    m2 = m2[n - 1 :] / n

    # stddev
    s = m2 - m ** 2

    if stride > 1:
        m = m[::stride]
        s = s[::stride]

    return m, s


def plot_runs(runs, smooth=100, validation_score=None):
    relation, validation_score = validation_score.split(":")
    if relation == "min":
        higher_is_better = False
    elif relation == "max":
        higher_is_better = True
    else:
        raise Exception(f"Don't know how to handle relation: {relation}")

    run_names = [str(run) for run in runs]
    max_iteration = max([run.training_stats.trained_until() for run in runs])

    colors = itertools.cycle(palette[20])
    loss_tooltips = [
        ("task", "@task"),
        ("model", "@model"),
        ("optimizer", "@optimizer"),
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

    validation_score_names = []
    for r in runs:
        if r.validation_scores.validated_until() > 0:
            validation_score_names += r.validation_scores.get_score_names()
    validation_score_names = np.unique(validation_score_names)

    validation_tooltips = [
        ("run", "@run"),
        ("task", "@task"),
        ("model", "@model"),
        ("optimizer", "@optimizer"),
        ("iteration", "@iteration"),
    ] + [(name, "@" + name) for name in validation_score_names]
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
        ("model", "@model"),
        ("optimizer", "@optimizer"),
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

    for run, color in zip(runs, colors):

        if run.training_stats.trained_until() > 0:

            name = str(run)
            l = run.training_stats.iterations[-1]

            x, _ = smooth_values(run.training_stats.iterations, smooth, stride=smooth)
            y, s = smooth_values(run.training_stats.losses, smooth, stride=smooth)
            source = bokeh.plotting.ColumnDataSource(
                {
                    "iteration": x,
                    "loss": y,
                    "task": [run.task_config.id] * len(x),
                    "model": [run.model_config.id] * len(x),
                    "optimizer": [run.optimizer_config.id] * len(x),
                    "run": [str(run)] * len(x),
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

        if validation_score and run.validation_scores.validated_until() > 0:

            x = run.validation_scores.iterations
            source_dict = {
                "iteration": x,
                "task": [run.task_config.id] * len(x),
                "model": [run.model_config.id] * len(x),
                "optimizer": [run.optimizer_config.id] * len(x),
                "run": [str(run)] * len(x),
            }
            # TODO: get_best: higher_is_better is not true for all scores
            validation_averages = run.validation_scores.get_best(
                validation_score, higher_is_better=higher_is_better
            )
            source_dict.update(
                {
                    name: np.array(validation_averages[name])
                    for name in run.validation_scores.get_score_names()
                }
            )
            source = bokeh.plotting.ColumnDataSource(source_dict)
            validation_figure.line(
                "iteration",
                validation_score,
                legend_label=name + " " + validation_score,
                source=source,
                color=color,
                alpha=0.7,
            )

    # Styling
    # training
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

    validation_figure.yaxis.axis_label = f"{validation_score.capitalize()}"
    validation_figure.yaxis.axis_label_text_font_size = "20pt"
    validation_figure.yaxis.major_label_text_font_size = "16pt"
    validation_figure.yaxis.axis_label_text_font = "times"
    validation_figure.yaxis.axis_label_text_color = "black"

    bokeh.plotting.output_file("performance_plots.html")
    bokeh.plotting.save(bokeh.layouts.column(loss_figure, validation_figure))

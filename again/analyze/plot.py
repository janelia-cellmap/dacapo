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
    m = m[n-1:]/n

    # mean of squared values
    m2 = np.cumsum(a**2)
    m2[n:] = m2[n:] - m2[:-n]
    m2 = m2[n-1:]/n

    # stddev
    s = m2 - m**2

    if stride > 1:
        m = m[::stride]
        s = s[::stride]

    return m, s

def plot_runs(runs, smooth=100, validation_score=None):

    run_names = [str(run) for run in runs]
    max_iteration = max([run.training_stats.trained_until() for run in runs])

    colors = itertools.cycle(palette[20])
    loss_tooltips = [
        ("task", "@task"),
        ("model", "@model"),
        ("optimizer", "@optimizer"),
        ("iteration", "@iteration"),
        ("loss", "@loss")
    ]
    loss_figure = bokeh.plotting.figure(
        tools="pan, wheel_zoom, reset, save, hover",
        x_axis_label='iterations',
        tooltips=loss_tooltips,
        plot_width=2048)
    loss_figure.background_fill_color = '#efefef'


    validation_score_names = []
    for r in runs:
        if r.validation_scores.validated_until() > 0:
            validation_score_names += r.validation_scores.get_score_names()
    validation_score_names = np.unique(validation_score_names)

    validation_tooltips = [
        ("task", "@task"),
        ("model", "@model"),
        ("optimizer", "@optimizer"),
        ("iteration", "@iteration")
    ] + [
        (name, '@' + name)
        for name in validation_score_names
    ]
    validation_figure = bokeh.plotting.figure(
        tools="pan, wheel_zoom, reset, save, hover",
        x_axis_label='iterations',
        tooltips=validation_tooltips,
        plot_width=2048)
    validation_figure.background_fill_color = '#efefef'

    summary_tooltips = [
        ("task", "@task"),
        ("model", "@model"),
        ("optimizer", "@optimizer"),
        ("best iteration", "@iteration"),
        ("best voi_split", "@voi_split"),
        ("best voi_merge", "@voi_merge"),
        ("best voi_sum", "@voi_sum"),
        ("num parameters", "@num_parameters")
    ]
    summary_figure = bokeh.plotting.figure(
        tools="pan, wheel_zoom, reset, save, hover",
        x_axis_label='model size',
        y_axis_label='best validation',
        tooltips=summary_tooltips,
        plot_width=2048)
    summary_figure.background_fill_color = '#efefef'

    for run, color in zip(runs, colors):

        if run.training_stats.trained_until() > 0:

            name = str(run)
            l = run.training_stats.iterations[-1]

            x, _ = smooth_values(
                run.training_stats.iterations,
                smooth,
                stride=smooth)
            y, s = smooth_values(
                run.training_stats.losses,
                smooth,
                stride=smooth)
            source = bokeh.plotting.ColumnDataSource({
                'iteration': x,
                'loss': y,
                'task': [run.task_config.id]*len(x),
                'model': [run.model_config.id]*len(x),
                'optimizer': [run.optimizer_config.id]*len(x)
            })
            loss_figure.line(
                'iteration', 'loss',
                legend_label=name,
                source=source,
                color=color,
                alpha=0.7)

            loss_figure.patch(
                np.concatenate([x, x[::-1]]),
                np.concatenate([y + 3*s, (y - 3*s)[::-1]]),
                legend_label=name,
                color=color,
                alpha=0.3)

        if validation_score and run.validation_scores.validated_until() > 0:

            x = run.validation_scores.iterations
            source_dict = {
                'iteration': x,
                'task': [run.task_config.id]*len(x),
                'model': [run.model_config.id]*len(x),
                'optimizer': [run.optimizer_config.id]*len(x)
            }
            validation_averages = run.validation_scores.get_averages()
            source_dict.update({
                name: np.array(validation_averages[name])
                for name in run.validation_scores.get_score_names()
            })
            source = bokeh.plotting.ColumnDataSource(source_dict)
            validation_figure.line(
                'iteration', validation_score,
                legend_label=name + " " + validation_score,
                source=source,
                color=color,
                alpha=0.7)

    bokeh.plotting.show(
        bokeh.layouts.column(
            loss_figure,
            validation_figure,
            summary_figure
        )
    )


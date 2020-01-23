from bokeh.palettes import Category20 as palette
import bokeh.layouts
import bokeh.plotting
import itertools
import numpy as np


def smooth_values(y, n):

    y_split = np.array([y[i:-(n - 1) + i] if i < n -1 else y[i:] for i in range(n)])
    # mean
    m = np.mean(y_split, axis=0)
    # stddev
    s = np.mean((y_split - m)**2, axis=0)

    return m, s

def plot_runs(runs, smooth=100):

    run_names = [str(run) for run in runs]
    max_iteration = max([run.training_stats.trained_until() for run in runs])

    colors = itertools.cycle(palette[20])
    loss_figure = bokeh.plotting.figure(
        tools="pan, wheel_zoom, reset, save, hover",
        x_axis_label='iterations',
        plot_width=2048)
    loss_figure.background_fill_color = '#efefef'

    validation_figure = bokeh.plotting.figure(
        tools="pan, wheel_zoom, reset, save, hover",
        x_axis_label='iterations',
        plot_width=2048)
    validation_figure.background_fill_color = '#efefef'

    # losses


    for run, color in zip(runs, colors):

        if run.training_stats.trained_until() > 0:

            name = str(run)
            l = run.training_stats.iterations[-1]

            x, _ = smooth_values(
                run.training_stats.iterations,
                smooth)
            y, s = smooth_values(
                run.training_stats.losses,
                smooth)
            loss_figure.line(
                x, y,
                legend_label=name,
                color=color,
                alpha=0.7)

            loss_figure.patch(
                np.concatenate([x, x[::-1]]),
                np.concatenate([y + 3*s, (y - 3*s)[::-1]]),
                legend_label=name,
                color=color,
                line_alpha=0.7,
                fill_alpha=0.4)

        if run.validation_scores.validated_until() > 0:
            validation_averages = run.validation_scores.get_averages()

            if 'voi_split' in run.validation_scores.get_score_names():

                voi_sum = np.array(
                    validation_averages['voi_split'] +
                    validation_averages['voi_merge'])

                validation_figure.line(
                    run.validation_scores.iterations,
                    voi_sum,
                    legend_label=name + " VOI sum",
                    color=color,
                    alpha=0.7)

    bokeh.plotting.show(bokeh.layouts.column(loss_figure, validation_figure))


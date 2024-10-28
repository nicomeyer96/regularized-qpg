import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from helper import setup_figure_latex_layout


def read(reg, seed, angle_range, velocity_range):
    with open(os.path.join('results', 'generalization', f'reg={reg:.2f}_{seed}.pkl'), 'rb') as ff:
        data = pickle.load(ff)
    result = np.zeros((len(angle_range), len(velocity_range)))
    for angle_index, angle in enumerate(angle_range):
        for velocity_index, velocity in enumerate(velocity_range):
            # determine the fraction of converged runs
            result[angle_index, velocity_index] = np.average(data[angle][velocity] == 200)
    return result


def extract(reg, seeds, angle_range, velocity_range):
    # perform averaging operation over different models
    result = []
    for seed in seeds:
        result.append(read(reg, seed, angle_range, velocity_range))
    return np.average(result, axis=0), np.var(result, axis=0)


def plot(axis, reg, angle_range, velocity_range, seeds=range(100)):

    # helper for highlighting cells
    def highlight_cell(x, y, use_ax=None, **kwargs):
        rect = mpl.patches.Rectangle((x - .44, y - .44), 0.88, 0.88, fill=False, **kwargs)
        use_ax = use_ax or plt.gca()
        use_ax.add_patch(rect)

    # extract averaged number of converged nums (and standard deviation)
    data = extract(reg, seeds, angle_range, velocity_range)
    # extract same for baseline (non-regularized)
    data_base = extract(0.0, seeds, angle_range, velocity_range)

    im_handle = axis.imshow(data[0], cmap='gray_r', vmin=0, vmax=1, interpolation='none', origin='lower')

    # adjust grid and ticks
    axis.set_xticks(np.arange(-.5, len(velocity_range), 1), minor=True)
    axis.set_yticks(np.arange(-.5, len(angle_range), 1), minor=True)
    axis.grid(which='minor', color='k', linestyle=(0, (1, 5)), linewidth=.25)
    axis.tick_params(which='minor', bottom=False, left=False)
    axis.set_xticks(np.arange(len(velocity_range)), ['', -2, '', -1, '', 0, '', 1, '', 2, ''])
    axis.set_yticks(np.arange(len(angle_range)), [0.01, '', 0.05, '', 0.09, '', 0.13, '', 0.17, '', 0.21, '', 0.25])
    axis.set_title(r'$\lambda='+f'{reg}'+r'$', fontsize=8)

    # iterate over all angle-velocity combinations
    for row in range(len(angle_range)):
        for col in range(len(velocity_range)):
            # extract performance rates within one/half variances
            avg, var = data[0][row, col], data[1][row, col]
            avg_base, var_base = data_base[0][row, col], data_base[1][row, col]
            best, best_half = avg + var, avg + 0.5 * var
            worst, worst_half = avg - var, avg - 0.5 * var
            best_base, best_half_base = avg_base + var_base, avg_base + 0.5 * var_base
            worst_base, worst_half_base = avg_base - var_base, avg_base - 0.5 * var_base

            # determine if result is significantly/slightly better/worse than baseline
            # (skip results with very low success probability to prevent plot clustering)
            if 0.015 < avg:  # skip results with very low success probability to prevent plot clustering
                if avg > avg_base:  # better
                    if worst > best_base:
                        highlight_cell(col, row, use_ax=axis, color='g', linewidth=1.0)
                    elif worst_half > best_half_base:
                        highlight_cell(col, row, use_ax=axis, color='c', linewidth=1.0)
                else:  # worse
                    if best < worst_base:
                        highlight_cell(col, row, use_ax=axis, color='r', linewidth=1.0)
                    elif best_half < worst_half_base:
                        highlight_cell(col, row, use_ax=axis, color='orange', linewidth=1.0)
    # return handle for plotting color map
    return im_handle


if __name__ == '__main__':

    # determine figsize for paper
    figsize = setup_figure_latex_layout(height_cm=6.0, single_column=False)

    # angle and velocity ranges to show results for
    angles = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25]
    velocities = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    # set up plot
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=figsize,
                           gridspec_kw={'wspace': 0.12, 'right': 0.9, 'left': 0.08})
    fig.supxlabel(r'$\textbf{pole angular velocity}~(\pm 0.25)$', fontsize=8, y=0.02)
    fig.supylabel(r'$\textbf{pole angle}~(\pm 0.01)$', fontsize=8, x=0.01)

    for _reg, _ax in zip([0.0, 0.1, 0.2, 0.4], ax): # noqa
        im = plot(_ax, reg=_reg, angle_range=angles, velocity_range=velocities)

    # place colorbar to right of rightmost plot
    cbar_ax = fig.add_axes([0.92, 0.17, 0.015, 0.643])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)  # noqa
    cbar.set_label(r'$\textbf{attraction rate}$', x=0.015)

    # plot legend
    legend_elements = [mpl.lines.Line2D([0], [0], color='g', marker='s', markerfacecolor='none', markeredgewidth=2, lw=0,
                                        label='significantly better'),
                       mpl.lines.Line2D([0], [0], color='c', marker='s', markerfacecolor='none', markeredgewidth=2, lw=0,
                                        label='slightly better'),
                       mpl.lines.Line2D([0], [0], color='orange', marker='s', markerfacecolor='none', markeredgewidth=2, lw=0,
                                        label='slightly worse'),
                       mpl.lines.Line2D([0], [0], color='r', marker='s', markerfacecolor='none', markeredgewidth=2, lw=0,
                                        label='significantly worse')]
    leg = fig.legend(handles=legend_elements, loc='upper center', ncol=4, handletextpad=0.1, columnspacing=1.0,
                     bbox_to_anchor=(0.49, 1.015))
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(.75)

    # save figure
    filename = os.path.join(os.path.dirname(__file__), 'plots', f'generalization.pdf')
    plt.savefig(filename, format='pdf')
    print(f'Stored figure to {filename}')
    # show figure
    plt.show()

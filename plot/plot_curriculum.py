# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from helper import setup_figure_latex_layout, COLORS


def read(reg, seed, quality=0.0):
    path = os.path.join('results', 'train', f'curr[-{quality+0.25:.2f},{quality+0.25:.2f}]_reg={reg:.2f}_{seed}.pkl')
    if os.path.isfile(path):  # model has reached this accuracy
        with open(path, 'rb') as ff:
            data = pickle.load(ff)
        return data['failures']
    else:  # model has not reached this accuracy
        return None


def read_test(reg, seed, quality, velocities):
    path = os.path.join('results', 'generalization', f'velocity_curr[-{quality+0.25:.2f},{quality+0.25:.2f}]_reg={reg:.2f}_{seed}.pkl')
    if os.path.isfile(path):  # model has reached this accuracy
        with open(path, 'rb') as ff:
            data = pickle.load(ff)
        result = []
        for velocity in velocities:
            result.append(data[velocity])
        success = 200 == np.array(result)  # extract successful runs
        return np.average(success, axis=1)  # fraction of successful runs
    else:  # model has not reached this accuracy
        return None


def extract(regs, qualities, seeds):
    """ Extract how many models with this regularization rate have converged to different accuracies,
    and how many failures this took. """
    for reg in regs:
        print(f'[reg={reg:.2f}]  ', end='')
        for quality in qualities:
            success_counter, failures = 0, []
            for seed in seeds:
                failure = read(reg, seed, quality)
                if failure is not None:  # model did reach this accuracy
                    success_counter += 1
                    failures.append(failure)
                else:  # model took maximum of 1000 failures
                    failures.append(1000)
            print(f'+-{quality+0.25:.2f} -> {np.average(failures):.1f} failures ({100 * success_counter / len(seeds):.0f}%) | ', end='')
        print()


def plot(quality, axis, regs, seeds):

    velocities = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    for i, (reg, offset, color) in enumerate(zip(regs, [-0.15, 0.0, 0.15], ['gray', 'purple', 'royalblue'])):

        result = []
        for seed in seeds:
            success = read_test(reg, seed, quality, velocities)
            if success is not None:
                result.append(success)
        if 0 == len(result):  # no model has been trained to this quality
            avg, var = 0.0, 0.0
        else:  # average over trained models
            avg, var = np.average(result, axis=0), np.var(result, axis=0)

        axis.bar(np.array(velocities) + offset, avg, yerr=var, width=0.15, capsize=2,
                 error_kw={'elinewidth': .5, 'capthick': .5}, color=COLORS[reg], edgecolor='k', linewidth=.5)

    axis.set_title(f'Trained on angular velocities [{-(quality+0.25)}, {quality+0.25}]', fontsize=8)

    axis.set_xticks(velocities, [-2.0, '', -1.0, '', 0.0, '', 1.0, '', 2.0])
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, '', 0.5, '', 1.0])
    axis.grid(axis='y', linewidth=.25, linestyle='-')
    axis.spines[['right', 'top']].set_visible(False)


if __name__ == '__main__':

    # velocity ranges and quality to show results for
    _regs = [0.00, 0.10, 0.20]
    _qualities = [0.0, 0.5, 1.0, 1.5]

    # determine fraction of converged runs and failures to get there
    extract(_regs, _qualities, range(100))

    # set up plot
    figsize = setup_figure_latex_layout(height_cm=11.0, single_column=True)
    fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=figsize,
                           gridspec_kw={'hspace': 0.6, 'right': 0.95, 'left': 0.15})

    # plot
    for _quality, _ax in zip(_qualities, ax):  # noqa
        plot(_quality, _ax, regs=_regs, seeds=range(100))

    # axis labels
    fig.supylabel(r'$\textbf{attraction rate}$', fontsize=8, x=0.015)
    fig.supxlabel(r'$\textbf{pole angular velocity}~(\pm 0.25)$', fontsize=8, y=0.015, x=0.55)

    # plot legend
    legend_elements = [mpl.lines.Line2D([0], [0], color=COLORS[0.00], marker='s', linestyle='none', label=r'$\lambda=0.0$'),
                       mpl.lines.Line2D([0], [0], color=COLORS[0.10], marker='s', linestyle='none', label=r'$\lambda=0.1$'),
                       mpl.lines.Line2D([0], [0], color=COLORS[0.20], marker='s', linestyle='none', label=r'$\lambda=0.2$')]
    leg = fig.legend(handles=legend_elements, loc='upper center', ncol=3, handletextpad=0.1, columnspacing=1.0, bbox_to_anchor=(0.55, 1.0))
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(.75)

    # save figure
    filename = os.path.join(os.path.dirname(__file__), 'plots', f'curriculum.pdf')
    plt.savefig(filename, format='pdf')
    print(f'Stored figure to {filename}')
    # show figure
    plt.show()

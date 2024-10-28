# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from helper import setup_figure_latex_layout, COLORS


# perturbation values tested for
PERTURBATIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def read(reg, seed):
    with open(os.path.join('results', 'robustness', f'reg={reg:.2f}_{seed}.pkl'), 'rb') as ff:
        data = pickle.load(ff)
    rewards_test = []
    # extract and average testing runs for different perturbations
    for per in PERTURBATIONS:
        rewards_test.append(np.average(data.get(per, 0.0)))
    return np.array(rewards_test)


def plot(axis, reg, seeds):

    # extract training results, average over available seeds
    rewards_test = []
    for seed in seeds:
        rewards_test.append(read(reg, seed))
    # average over different models
    rewards_test = np.average(rewards_test, axis=0)  # noqa
    # plot
    ax.plot(np.array(range(rewards_test.shape[0])), rewards_test, COLORS[reg], linewidth=1.0, label=r'$\lambda='+f'{reg:.1f}'+r'$', marker='o', mfc='none')


if __name__ == '__main__':

    # determine figsize for paper
    figsize = setup_figure_latex_layout(5.0, single_column=True)

    # set up plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, sharex=True, sharey=True,
                           gridspec_kw={'right': 0.95, 'left': 0.15, 'bottom': 0.20, 'top': 0.9})
    ax.grid(axis='y', linewidth=.25, linestyle='-')
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.0, '', 0.2, '', 0.4, '', 0.6, '', 0.8, '', 1.0])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.spines[['right', 'top']].set_visible(False)
    plt.xlim(0, 8.5)
    plt.ylim(0, 210)

    # plot averaged testing curves
    for _reg in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]:
        plot(ax, reg=_reg, seeds=range(100))

    # plot maximum achievable rewards
    ax.plot([0, 100], [200, 200], 'k--', linewidth=0.75)

    # plot axis labels
    fig.supxlabel(r'$\textbf{perturbation}~\mathcal{N}(0;\bullet)$', fontsize=8, y=0.025, x=0.53)
    fig.supylabel(r'$\textbf{reward}$', fontsize=8, x=0.015)

    # plot legend
    leg = fig.legend(ncol=2, handletextpad=0.4, columnspacing=1.0, bbox_to_anchor=(0.965, 0.875), facecolor='white', framealpha=1.0)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(.75)

    # save figure
    filename = os.path.join(os.path.dirname(__file__), 'plots', f'robustness.pdf')
    plt.savefig(filename, format='pdf')
    print(f'Stored figure to {filename}')
    # show figure
    plt.show()

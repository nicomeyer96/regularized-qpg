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


def read(reg, seed):
    with open(os.path.join('results', 'train', f'reg={reg:.2f}_{seed}.pkl'), 'rb') as ff:
        data = pickle.load(ff)
    rewards = data['rewards_train']
    rewards = np.array([np.average(rewards[e]) for e in range(len(rewards))])  # average over batch for each epoch
    return np.pad(rewards, (0, 101-len(rewards)), 'constant', constant_values=200)  # pad early terminated results


def plot(axis, reg, seeds):

    # extract training results, average over available seeds
    rewards_train = []
    for seed in seeds:
        rewards_train.append(read(reg, seed))
    rewards_train = np.average(rewards_train, axis=0)  # noqa

    # plot training curve
    axis.plot(range(len(rewards_train)), rewards_train, COLORS[reg], linewidth=1.0, label=r'$\lambda='+f'{reg:.1f}'+r'$')


if __name__ == '__main__':

    # determine figsize for paper
    figsize = setup_figure_latex_layout(5.0, single_column=True)

    # set up plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, sharex=True, sharey=True,
                           gridspec_kw={'right': 0.95, 'left': 0.15, 'bottom': 0.20, 'top': 0.9})
    ax.grid(axis='y', linewidth=.25, linestyle='-')
    ax.grid(axis='x', linewidth=.25, linestyle='-')
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [0, '', 20, '', 40, '', 60, '', 80, '', 100])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 210)
    ax.spines[['right', 'top']].set_visible(False)
    fig.supxlabel(r'$\textbf{training epoch}$', fontsize=8, y=0.025, x=0.53)
    fig.supylabel(r'$\textbf{reward}$', fontsize=8, x=0.015)

    # plot averaged training curves
    for _reg in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]:
        plot(ax, reg=_reg, seeds=range(100))

    # plot maximum achievable rewards
    ax.plot([0, 100], [200, 200], 'k--', linewidth=0.75)

    # plot legend
    leg = fig.legend(ncol=2, handletextpad=0.4, columnspacing=1.0, bbox_to_anchor=(0.875, 0.625), framealpha=1.0)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(.75)

    # save figure
    filename = os.path.join(os.path.dirname(__file__), 'plots', f'train.pdf')
    plt.savefig(filename, format='pdf')
    print(f'Stored figure to {filename}')
    # show figure
    plt.show()

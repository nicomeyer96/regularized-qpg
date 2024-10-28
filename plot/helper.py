# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import matplotlib.pyplot as plt


# colors for plots
COLORS = {
    0.00: 'indigo',
    0.10: 'royalblue',
    0.20: 'limegreen',
    0.30: 'gold',
    0.40: 'darkorange',
    0.50: 'red'
}


# compute figsize for paper
def setup_figure_latex_layout(height_cm, single_column=True):
    tex_fonts = {
        # Use LaTeX to write all text, load fonts
        "text.usetex": True,
        'text.latex.preamble': r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": "STIX",
        "mathtext.fontset": 'stix',
        # Set font sizes
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        'xtick.major.size': 3,
        'xtick.major.width': .5,
        'ytick.major.size': 3,
        'ytick.major.width': .5,
    }
    plt.style.use('default')
    plt.rcParams.update(tex_fonts)

    cm = 1/2.54  # cm to inches
    width_cm = 8.4 if single_column else 17.4  # QMI journal requirement
    assert height_cm <= 23.4  # QMI journal requirement

    return width_cm * cm, height_cm * cm

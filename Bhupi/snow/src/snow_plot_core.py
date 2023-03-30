import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np


def plot_aDMG_maxDim2(dgm, ax, ax_min, ax_max, title_):
    """Returns a plot of presistence homology diagram.
               The Birth/Death plot. Currently, it does it only for 2-dimensions;
               H_0, H_1, and H_2. The reason is that I have three colors listed here
               in ```color_dict```.

       Side Note:
       The reason for creating this funciton is that presim's plot
       function is crazy. Anytime I change the format of figures
       (i.e. settings of matplotlib), presim behaves strangely;
       e.g.
       1. if I run the same cell containing presim twice, the size of figure changes.
          The font and font size changes.
       2. It also disfunctions when doing subplots. Fonts of different subplots
          come out differently. I also could not make a 2-by-2 subplots with it.

    Hossein: March 29, 2023

    Arguments
    ---------
    dgm : dict
        ripser dictionary of presistent homology

    ax : axis
        An axis object of Matplotlib.

    ax_min : float
        A float indicating minimum limit for axes.

    ax_max : float
       A float indicating maximum limit for axes.

    title_ : str
        Start Of Season threshold

    Returns
    -------
    A plot of Birth/Death from persistent homology diagram.
    """
    len_dgm = len(dgm)
    color_dict = {"dgm0_c": "dodgerblue", "dgm1_c": "orange", "dgm2_c": "green"}

    scatter_size = 10
    y_hor = dgm[0][:, 1][-2]
    for ii in range(len_dgm):
        ax.scatter(
            dgm[ii][:, 0],
            dgm[ii][:, 1],
            s=scatter_size,
            c=color_dict["dgm" + str(ii) + "_c"],
            label=f"$H_{ii}$",
        )

    ax.set_title(title_)
    ax.set_ylim([ax_min, ax_max])
    ax.set_xlim([ax_min, ax_max])
    x = np.linspace(ax_min, ax_max, 100)
    ax.plot(x, x, linewidth=1.0, linestyle="dashed", c="k")
    ax.plot(x, [y_hor] * len(x), linewidth=1.0, linestyle="dashed", c="k")
    ax.legend(loc="lower right")

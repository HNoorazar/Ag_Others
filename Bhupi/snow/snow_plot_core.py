import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np


def plot_aDMG_maxDim2(dgm, ax, ax_min, ax_max, title_):
    """Returns A plot with of a given VI (NDVI or EVI) with SOS and EOS points.

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
    A plot.
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

import sys
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/Kirtis_Class/Python/')
# if you need a function from the Kirtis_Class_core.py module
import Kirtis_Class_core as kcc


def plot_a_list(x_values, y_values, ax, title_, _label="NDVI", _color="red",
                lineStyle="-"):
    ax.plot(x_values, y_values,
            linestyle=lineStyle,
            label=_label, linewidth=3.5,
            color=_color,
            alpha=0.8)

    ax.set_title(title_)
    ax.set_ylabel(_label)
    ax.tick_params(axis='y', which='major')
    ax.tick_params(axis='x', which='major')
    ax.legend(loc="upper right")
    ax.set_ylim(y_values.min()-0.5,
                y_values.max()+0.5)

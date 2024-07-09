'''
Helper functions for making plots look nice.
'''
import numpy as np
import matplotlib.pyplot as plt
import params

def make_nice_spines(ax, linewidth=params.LINEWIDTH):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 2 * linewidth))
    ax.spines["bottom"].set_position(("outward", 2 * linewidth))
    ax.tick_params(width=linewidth)
    ax.tick_params(length=2.5 * linewidth)
    ax.tick_params(labelsize=params.LABEL_SIZE)
    ax.spines["left"].set_linewidth(linewidth)
    ax.spines["bottom"].set_linewidth(linewidth)
    ax.spines["top"].set_linewidth(0)
    ax.spines["right"].set_linewidth(0)
    # axis label legends
    ax.xaxis.label.set_size(params.LABEL_SIZE)
    ax.yaxis.label.set_size(params.LABEL_SIZE)
    ax = set_ticks(ax)
    return ax

def make_nice_cbar(cbar, linewidth=params.LINEWIDTH):
    cbar.outline.set_linewidth(0.5*linewidth)
    cbar.ax.tick_params(width=linewidth)
    cbar.ax.tick_params(length=2.5 * linewidth)
    cbar.ax.tick_params(labelsize=params.LABEL_SIZE)
    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    cbar.ax.yaxis.label.set_size(params.LABEL_SIZE)
    return cbar

def make_axis_disappear(ax):
    sides = ["top", "left", "right", "bottom"]
    for side in sides:
        ax.spines[side].set_visible(False)
        ax.spines[side].set_linewidth(0)
        ax.set_xticks([])
        ax.set_yticks([])

def set_ticks(ax, n_ticks:int = 5):
    """
    Set the number of ticks on the axis.
    """
    start,stop = ax.get_xlim()
    xticks = np.linspace(start, stop, n_ticks)
    xticks = [int(x) for x in xticks]
    ax.set_xticks(xticks)
    start,stop = ax.get_ylim()
    yticks = np.linspace(start, stop, n_ticks)
    yticks = [int(y) for y in yticks]
    ax.set_yticks(yticks)
    return ax

def scatter_xyz_2d(X,Y,Z, ax, cmap=params.red_heatmap, marker='o'):
    """
    Scatter plot in 2D with Z as the color.
    """
    ax.scatter(X, Y, c=Z, cmap=cmap, marker=marker)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    cbar = plt.colorbar(
        ax.collections[0],
        ax=ax,
        orientation='vertical',
        label='Z',
        )
    cbar = make_nice_cbar(cbar)
    ax = make_nice_spines(ax)
    plt.tight_layout()
    return ax
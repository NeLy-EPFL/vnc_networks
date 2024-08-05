'''
Helper functions for making plots look nice.
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import params
import seaborn as sns
import pandas as pd
import warnings
from matplotlib_venn import venn3


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

def scatter_xyz_2d(
        X,
        Y,
        Z,
        ax,
        cmap=params.red_heatmap,
        marker='o',
        z_label='Z',
        discrete_coloring: bool=False,
        ):
    """
    Scatter plot in 2D with Z as the color.
    """
    if discrete_coloring:
        unique_z = np.unique(Z)
        n_c = len(unique_z)
        colors = sns.color_palette(palette = cmap, n_colors = n_c)
        #cmap = mpl.colors.ListedColormap(colors)
        color_map = dict(zip(unique_z, colors))
        df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
        plt.scatter(
            df['X'],
            df['Y'],
            c=df['Z'].map(color_map),
            #cmap=cmap,
            marker=marker,
            alpha=0.8,
            )
        
        with warnings.catch_warnings():
            # suppress warnings from matplotlib, as plotting empty lists with
            # a label will raise a user warning
            warnings.simplefilter("ignore")
            for z in unique_z:
                ax.scatter([], [], c=color_map[z], label=z)
                ax.legend(
                    fontsize=params.LABEL_SIZE,
                    title=z_label,
                    title_fontsize=params.LABEL_SIZE
                    )
            
    else:
        ax.scatter(X, Y, c=Z, cmap=cmap, marker=marker,alpha=0.8)
        cbar = plt.colorbar(
            ax.collections[0],
            ax=ax,
            orientation='vertical',
            label=z_label,
            )
        cbar = make_nice_cbar(cbar)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax = make_nice_spines(ax)
    plt.tight_layout()
    return ax

def venn_3(
        sets: list,
        set_labels: list[str] = ['','',''],
        ax: plt.Axes = None,
        colors=params.custom_palette[:3],
        alpha=0.8,
        title: str = '',
        ):
    """
    Plot a 3-way Venn diagram.

    Parameters
    ----------
    sets : list
        List of sets to compare. Must be a list of 3 sets/lists.
    set_labels : list[str], optional
        Labels for the sets, by default None
    ax : plt.Axes, optional
        Axis to plot on, by default None
    colors : list, optional
        List of colors to use for the sets, by default params.custom_palette[:3]
    alpha : float, optional
        Transparency of the sets, by default 0.8
    title : str, optional
        Title for the plot, by default None

    Returns
    -------
    plt.Axes
        Axis with the Venn diagram.    
    """
    if len(sets) != 3:
        raise ValueError('Must provide 3 sets to compare')
    if ax is None:
        _, ax = plt.subplots(figsize=params.FIGSIZE, dpi=params.DPI)

    v = venn3(
        sets,
        set_labels,
        ax=ax,
        set_colors=colors,
        alpha=alpha,
        )
    ax.set_title(title)
    return ax

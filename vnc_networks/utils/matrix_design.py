"""
Module for matrix visualization.
"""

from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.axes
import scipy as sc

import params
import utils.plots_design as plots_design


def spy(matrix, title: Optional[str] = None):
    """
    Visualizes the sparsity pattern of a matrix.

    Parameters
    ----------
    matrix : sparse matrix
        The matrix to visualize.
    title : str, optional
        The title of the plot. The default is None.
    """
    _, ax = plt.subplots(figsize=params.FIGSIZE, dpi=params.DPI)
    plt.spy(matrix, markersize=0.1)
    ax = plots_design.make_nice_spines(ax)
    if title is not None:
        plt.title(title)
    return ax


def imshow(
    matrix,
    title: str,
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    cmap=params.diverging_heatmap,
    ax: Optional[matplotlib.axes.Axes] = None,
    xlabel: str = "postsynaptic neuron",
    ylabel: str = "presynaptic neuron",
    row_labels: Optional[list] = None,
    col_labels: Optional[list] = None,
    save: bool = False,
):
    """
    Plot a matrix with a colorbar.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=params.FIGSIZE, dpi=params.DPI)
    assert ax is not None  # just for the type hinting to work properly
    if sc.sparse.issparse(matrix):
        matrix_ = matrix.todense()
    else:
        matrix_ = matrix
    # Set the maximum value to the maximum absolute value if not provided
    if vmax is None:
        vmax = max(abs(matrix_.min()), matrix_.max())
    assert vmax is not None
    if vmin is None:
        vmin = -1 * vmax
    c = ax.imshow(
        matrix_,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        aspect="auto",
    )
    # axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)), labels=row_labels)
    if col_labels is not None:
        ax.set_xticks(range(len(col_labels)), labels=col_labels)
    ax = plots_design.make_nice_spines(ax)

    # colorbar
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label("connection strength")
    cbar = plots_design.make_nice_cbar(cbar)

    # design
    plt.tight_layout()

    # title
    if save:
        plt.savefig(title)
    return ax

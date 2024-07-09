"""
Module for matrix visualization.
"""
import matplotlib.pyplot as plt
import scipy as sc
import numpy as np

import params
import utils.plots_design as plots_design

def spy(matrix, title:str=None):
    '''
    Visualizes the sparsity pattern of a matrix.

    Parameters
    ----------
    matrix : sparse matrix
        The matrix to visualize.
    title : str, optional
        The title of the plot. The default is None.
    '''
    _, ax = plt.subplots(figsize=params.FIGSIZE, dpi=params.DPI)
    plt.spy(matrix, markersize=0.1)
    ax = plots_design.make_nice_spines(ax)
    if title is not None:
        plt.title(title)
    return ax

def imshow(
    matrix,
    title: str = None,
    vmax: float = None,
    cmap=params.diverging_heatmap,
):
    """
    Plot a matrix with a colorbar.
    """
    _, ax = plt.subplots(figsize=params.FIGSIZE, dpi=params.DPI)
    if sc.sparse.issparse(matrix):
        matrix_ = matrix.todense()
    else:
        matrix_ = matrix
    # Set the maximum value to the maximum absolute value if not provided
    if vmax is None:
        vmax = max(abs(matrix_.min()), matrix_.max())
    c = ax.imshow(
        matrix_,
        cmap=cmap,
        vmax=vmax,
        vmin=-1 * vmax,
        aspect="auto",
    )
    # axis
    ax.set_xlabel("postsynaptic neuron")
    ax.set_ylabel("presynaptic neuron")
    ax = plots_design.make_nice_spines(ax)
    
    # colorbar
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label("connection strength")
    cbar = plots_design.make_nice_cbar(cbar)

    # design

    # title
    if title is not None:
        plt.savefig(title)
    return ax
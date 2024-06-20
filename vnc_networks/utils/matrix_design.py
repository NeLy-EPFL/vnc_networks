"""
Module for matrix visualization.
"""
import matplotlib.pyplot as plt

import params
import utils.plots_graphic_design as plot_utils

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
    ax = plt.figure(figsize=params.FIGSIZE, dpi=params.DPI)
    plt.spy(matrix, markersize=0.1)
    if title is not None:
        plt.title(title)
    return
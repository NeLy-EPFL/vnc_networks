'''
Module containing utility functions for processing matrices.
'''

import numpy as np
import scipy as sc
import scipy.cluster.hierarchy as sch

def connections_up_to_n_hops(matrix, n):
    """
    input: (sparse) matrix

    computes: recursively computes all the connections possible in less than
    n hops.

    output: (sparse) matrix
    From an input (sparse) matrix compute the matrix representing the
    connectivity
    """
    return_mat = matrix
    for i in range(2, n + 1):
        return_mat += return_mat @ matrix
    return return_mat


def generate_random(matrix):
    """
    Generates a random matrix by shuffeling the elements of the matrix in
    argument
    """
    rng = np.random.default_rng()
    (n, m) = matrix.shape
    full_mat = matrix.todense()
    arr = np.array(full_mat.flatten())
    rng.shuffle(arr)  # , axis=1)
    arr = arr.reshape((n, m))
    return sc.sparse.csr_matrix(arr)


def density(matrix):
    """
    input: sparse matrix
    output: number of existing edges divided by edges on a fully-comnected
    graph of identical size
    """
    mat = matrix
    den = mat.nnz / np.prod(mat.shape)
    return den


def select_subset_matrix(matrix, sub_indices):
    """
    Extract a submatrix from a sparse matrix given by specific indices
    """
    mat_tmp = matrix[sub_indices, :]
    mat = mat_tmp[:, sub_indices]
    return mat


def connection_degree_n_hops(matrix, n, dn_indices=[]):
    """
    inputs:
        sparse matrix
        number of maximimal hops (n)
        indices to evaluate
    output:
        array of size n with densities for increasing number of hops allowed
    """

    degree = []
    for deg in range(1, n + 1):
        mat = connections_up_to_n_hops(matrix, deg)
        if len(dn_indices) == 0:
            degree.append(density(mat))
        else:
            dn_mat = select_subset_matrix(mat, dn_indices)
            degree.append(density(dn_mat))
    return degree


def cluster_matrix_hierarchical(matrix):
    """
    inputs: scipy sparse correlation matrix

    processing: clusters the matrix using scipy hierarchical clustering
        order indices such that the underlying dendrogram is a tree

    outputs: hierarchical clustered sparse matrix
    """
    # if the matrix is a scipy sparse matrix, convert to dense
    if sc.sparse.issparse(matrix):
        matrix = matrix.todense()
    dendrogram = sch.dendrogram(
        sch.linkage(matrix, method="ward"), no_plot=True
    )
    # get the order of the indices
    order = dendrogram["leaves"]
    # reorder the matrix
    clustered = matrix[order, :]
    clustered = clustered[:, order]
    return clustered, order
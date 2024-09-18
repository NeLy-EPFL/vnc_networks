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
    for _ in range(n - 1):
        return_mat += return_mat @ matrix
    return return_mat

def connections_at_n_hops(matrix, n):
    """
    input: (sparse) matrix

    computes: recursively computes all the connections possible in less than
    n hops.

    output: (sparse) matrix
    From an input (sparse) matrix compute the matrix representing the
    connectivity
    """
    if n == 1:
        return matrix
    elif n%2 == 0:
        return connections_at_n_hops(matrix @ matrix, n // 2)
    else:
        return matrix @ connections_at_n_hops(matrix @ matrix, n // 2)

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

def convert_index_to_bodyid(index, lookup):
    """
    input: index, lookup
    output: bodyid

    if index is an integer, return the bodyid as an integer
    """
    if isinstance(index, int):
        return_as_int = True
        index = [index]
    else:
        return_as_int = False
    bodyids = [
        lookup.loc[lookup["index"] == id].body_id.values[0]
        for id in index
        ]
    if return_as_int:
        return bodyids[0]
    return bodyids

def convert_bodyid_to_index(bodyid, lookup, allow_empty=True):
    """
    input: bodyid, lookup
    output: index

    if bodyid is an integer, return the index as an integer.
    if allow_empty is False, raise an error if the bodyid is not found.
    """
    # format inputs
    if bodyid is None or bodyid == []:
        raise ValueError("bodyid is None or empty.")
    if isinstance(bodyid, int):
        return_as_int = True
        bodyid = [bodyid]
    else:
        return_as_int = False
    # find indices
    empty_matches = []
    indices = []
    for id in bodyid:
        if not lookup["body_id"].isin([id]).any():
            empty_matches.append(id)
        else:
            indices.append(lookup.loc[lookup["body_id"] == id].index.values[0])
    # return results
    if not allow_empty and len(empty_matches) > 0:
        raise ValueError(f"bodyid(s) not found: {empty_matches}")
    else:
        if len(empty_matches) > 0:
            print(f"Warning: {len(empty_matches)} bodyid(s) not found.")
    if return_as_int:
        return indices[0]
    return indices

def count_nonzero(matrix, sign=None):
    """
    input: sparse matrix

    output: number of non-zero elements
    sign: if None, count all non-zero elements. 
    If 'positive', count only positive elements.
    If 'negative', count only negative elements.
    """
    if sign is None:
        return matrix.count_nonzero()
    if sign == "positive":
        pos_mat = matrix.copy()
        pos_mat.data = pos_mat.data > 0
        return pos_mat.count_nonzero()
    if sign == "negative":
        neg_mat = matrix.copy()
        neg_mat.data = neg_mat.data < 0
        return neg_mat.count_nonzero()

def sum_weights(matrix, sign=None):
    """
    input: sparse matrix

    output: sum of all weights
    sign: if None, sum all weights. 
    If 'absolute', sum the absolute value of all weights.
    If 'positive', sum only positive weights.
    If 'negative', sum only negative weights.
    """
    if sign is None:
        return matrix.sum()
    if sign == "absolute":
        return np.sum(np.abs(matrix.data))
    if sign == "positive":
        pos_mat = matrix.copy()
        pos_mat.data = pos_mat.data * (pos_mat.data > 0)
        return pos_mat.sum()
    if sign == "negative":
        neg_mat = matrix.copy()
        neg_mat.data = neg_mat.data * (neg_mat.data < 0)
        return neg_mat.sum()
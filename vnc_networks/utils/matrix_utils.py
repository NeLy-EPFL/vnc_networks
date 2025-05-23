#!/usr/bin/env python3
"""
Module containing utility functions for processing matrices.
"""

import typing

import numpy as np
import scipy as sc
import scipy.cluster.hierarchy as sch


def connections_up_to_n_hops(matrix, n: int):
    """
    input: (sparse) matrix

    computes: recursively computes all the connections possible in less than
    n hops.

    output: (sparse) matrix
    From an input (sparse) matrix compute the matrix representing the
    connectivity
    """
    if not sc.sparse.issparse(matrix):
        matrix = sc.sparse.csr_matrix(matrix)
    return_mat = matrix
    for _ in range(n - 1):
        return_mat += return_mat @ matrix
    return return_mat


def connections_at_n_hops(matrix, n: int):
    """
    input: (sparse) matrix

    computes: recursively computes all the connections possible in less than
    n hops.

    output: (sparse) matrix
    From an input (sparse) matrix compute the matrix representing the
    connectivity
    """
    # if the matrix is not sparse, convert it to sparse
    if not sc.sparse.issparse(matrix):
        matrix = sc.sparse.csr_matrix(matrix)

    if n == 1:
        return matrix
    elif n % 2 == 0:
        return connections_at_n_hops(matrix @ matrix, n // 2)
    else:
        return matrix @ connections_at_n_hops(matrix @ matrix, n // 2)


def generate_random(matrix):
    """
    Generates a random matrix by shuffling the elements of the matrix in
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
    output: number of existing edges divided by edges on a fully-connected
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


def connection_degree_n_hops(matrix, n: int, dn_indices=[]):
    """
    inputs:
        sparse matrix
        number of maximal hops (n)
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
    dendrogram = sch.dendrogram(sch.linkage(matrix, method="ward"), no_plot=True)
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
    bodyids = [lookup.loc[lookup["index"] == id].body_id.values[0] for id in index]
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


def count_nonzero(matrix, sign: typing.Literal["positive", "negative"] | None = None):
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


def sum_weights(
    matrix, sign: typing.Literal["positive", "negative", "absolute"] | None = None
):
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


def build_distance_matrix(
    matrix, method: typing.Literal["cosine_in", "cosine_out", "cosine", "euclidean"]
):
    """
    input: sparse matrix
    output: distance matrix
    """

    def _remove_nans(matrix):
        """
        input: matrix
        output: matrix without NaNs
        For Cosine distance, if a vector has norm 0 then the value is NaN.
        Design choice: set NaNs to 1 if they are off the diagonal, and to 0 if
        they are on the diagonal so that that entry is similar to itself only.
        """
        # set NaNs to 1 if they are off the diagonal
        nan_indices = np.isnan(matrix)
        nan_indices[np.diag_indices_from(nan_indices)] = False
        matrix[nan_indices] = 1
        # set NaNs to 0 if they are on the diagonal
        matrix[np.isnan(matrix)] = 0
        return matrix

    match method:
        case "cosine_in":  # cosine similarity of the input vectors
            # the vectors are the columns of the matrix, i.e. all i to each j
            columns = matrix.todense().T
            sim_mat = sc.spatial.distance.cdist(columns, columns, "cosine")
            sim_mat = _remove_nans(sim_mat)
        case "cosine_out":  # cosine similarity of the output vectors
            # the vectors are the rows of the matrix, i.e. all j to each i
            rows = matrix.todense()
            sim_mat = sc.spatial.distance.cdist(rows, rows, "cosine")
            sim_mat = _remove_nans(sim_mat)
        case "cosine":  # cosine similarity of the input and output vectors
            # the vectors are the columns of the matrix, i.e. all i to each j
            columns = matrix.todense().T
            sim_mat = sc.spatial.distance.cdist(columns, columns, "cosine")
            sim_mat = _remove_nans(sim_mat)
            # the vectors are the rows of the matrix, i.e. all j to each i
            rows = matrix.todense()
            new_sim_mat = sc.spatial.distance.cdist(rows, rows, "cosine")
            new_sim_mat = _remove_nans(new_sim_mat)
            sim_mat += new_sim_mat
            sim_mat /= 2
        case "euclidean":
            matrix = matrix.todense()
            matrix = np.concatenate((matrix, matrix.T), axis=1)
            sim_mat = sc.spatial.distance.cdist(matrix, matrix, "euclidean")
        case _:
            raise ValueError(f"Method {method} not recognized.")

    # correct for numerical errors
    # if the values are very close to 0, set them to 0
    sim_mat[sim_mat < 1e-10] = 0
    return sc.sparse.csr_matrix(sim_mat)


def markov_clustering(matrix, inflation=2, iterations=100):
    """
    input: sparse matrix
    output: clusters
    """
    import markov_clustering as mcl

    result = mcl.run_mcl(matrix, inflation=inflation, iterations=iterations)
    clusters = mcl.get_clusters(result)

    return clusters

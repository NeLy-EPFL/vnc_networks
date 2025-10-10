#!/usr/bin/env python3
"""
Initialises the cmatrix class, standing for connectome matrix.
The purpose of this class is to manipulate the adjacency matrix of the connectome,
or subset of it, while keeping track of the associated node identities.

As much as possible, all methods are defined on the scipy sparse matrix to
minimize the memory footprint and improve the performance of the computations.

External access to nodes should be done through their body ids, which are unique.
The lookup table is used to convert between body ids and matrix indices. This
conversion is kept internally to the class.
"""

import copy
import os
import typing
from collections import defaultdict
from typing import Optional

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy as sc
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN

from . import params
from .connectome_reader import ConnectomeReader, default_connectome_reader
from .params import UID, BodyId
from .utils import matrix_design, matrix_utils


class CMatrix:
    def __deepcopy__(self, memo):
        """
        Deepcopy the cmatrix.
        Only the CR is not deep copied, just referenced.
        """
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance

        for k, v in self.__dict__.items():
            if k == "CR":
                setattr(new_instance, k, v)
            else:
                setattr(new_instance, k, copy.deepcopy(v, memo))

        return new_instance

    def __init__(
        self,
        matrix: sc.sparse.csr_matrix,
        lookup: pl.DataFrame,
        CR: ConnectomeReader | None = None,
    ):
        """
        Initialises the cmatrix class, standing for connectome matrix.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The adjacency matrix of the connectome.
        lookup : polars.DataFrame
            The lookup table for the equivalence between node indices and node identities.
        """
        # verify that the matrix is a csr matrix
        if not sc.sparse.issparse(matrix):
            matrix = sc.sparse.csr_matrix(matrix)
        if not isinstance(matrix, sc.sparse.csr_matrix):
            matrix = matrix.tocsr()
        self.matrix = matrix

        # verify that the lookup is a polars dataframe
        if not isinstance(lookup, pl.DataFrame):
            raise ValueError("The lookup must be a polars dataframe.")

        # duplicate the 'index' column into a 'row_index' and 'column_index' column
        self.lookup = lookup.with_columns(
            row_index=pl.col("index"),
            column_index=pl.col("index"),
        ).drop("index")
        # lookup["row_index"] = lookup["index"].values
        # lookup["column_index"] = lookup["index"].values
        # self.lookup = lookup.drop(columns="index")

        # verify that the lookup includes indices up to the length of the matrix
        if not all(self.lookup["row_index"].is_in(range(self.matrix.shape[0]))):
            raise ValueError(
                "The lookup must include row indices up to the length of the matrix."
            )
        if not all(self.lookup["column_index"].is_in(range(self.matrix.shape[1]))):
            raise ValueError(
                "The lookup must include column indices up to the length of the matrix."
            )

        self.CR = CR or default_connectome_reader()

    # private methods
    def __update_indexing(self):
        """
        Updates the indexing of the lookup table to match the new matrix.
        Every time the matrix is restricted, the lookup table must be updated and
        the indexing shifted back to a continuous range starting from 0.
        """
        lookup = self.get_lookup()

        # rows
        lookup = lookup.sort(by="row_index")
        n_rows = lookup["row_index"].count()
        n_nan = len(lookup) - n_rows
        new_row_indices = list(range(n_rows))
        new_row_indices.extend(np.nan * np.ones(n_nan))
        lookup["row_index"] = new_row_indices

        # columns
        lookup = lookup.sort(by="column_index")
        n_columns = lookup["column_index"].count()
        n_nan = len(lookup) - n_columns
        new_column_indices = list(range(n_columns))
        new_column_indices.extend(np.nan * np.ones(n_nan))
        lookup["column_index"] = new_column_indices

        # clean up unindexed uids
        lookup = lookup.drop_nulls(
            subset=["row_index", "column_index"],
        )
        # verify that the lookup includes indices up to the length of the matrix
        if n_rows != self.matrix.shape[0]:
            raise ValueError(
                "The lookup must include row indices up to the length of the matrix."
            )
        if n_columns != self.matrix.shape[1]:
            raise ValueError(
                "The lookup must include column indices up to the length of the matrix."
            )
        self.lookup = lookup

    def __restrict_row_indices(
        self,
        indices: list,
        allow_empty: bool = True,
    ):
        """
        Restricts the adjacency matrix to a subset of indices.

        Parameters
        ----------
        indices : list
            The list of indices to which the adjacency matrix is restricted.
        allow_empty : bool, optional
            If False, raises an error if the indices are not found in the lookup.
            The default is True.
        """
        # replace elements of 'row_index' not in indices with NaN
        new_vals = [
            i if i in indices else np.nan
            for i in self.get_row_indices()  # sorted numerically!
        ]
        restrict_indices = [  # keep the original index sorting
            i for i in self.get_row_indices() if i in indices
        ]

        # restrict the indices to those defined in the lookup
        if not allow_empty and np.count_nonzero(~np.isnan(new_vals)) != len(indices):
            raise ValueError("Some row indices not found in the lookup.")

        self.lookup["row_index"] = new_vals

        # restrict the matrix to the indices
        self.matrix = self.matrix[restrict_indices, :]
        self.__update_indexing()
        return

    def __restrict_column_indices(self, indices: list, allow_empty: bool = True):
        """
        Restricts the adjacency matrix to a subset of indices.

        Parameters
        ----------
        indices : list
            The list of indices to which the adjacency matrix is restricted.
        allow_empty : bool, optional
            If False, raises an error if the indices are not found in the lookup.
            The default is True.
        """
        # restrict the indices to those defined in the lookup
        defined_columns = [
            i
            for i in indices
            if i in self.get_column_indices()  # sorted numerically!
        ]
        if not allow_empty and len(defined_columns) != len(indices):
            raise ValueError("Some column indices not found in the lookup.")
        indices = defined_columns

        # replace elements of 'column_index' not in indices with NaN
        new_vals = [i if i in indices else np.nan for i in self.get_column_indices()]
        restrict_indices = [  # keep the original index sorting
            i for i in self.get_column_indices() if i in indices
        ]
        self.lookup["column_index"] = new_vals
        # restrict the matrix to the indices
        self.matrix = self.matrix[:, restrict_indices]
        self.__update_indexing()
        return

    def __convert_uid_to_index(self, uid: UID | list[UID], allow_empty=True):
        """
        input: uid
        output: indices
        if allow_empty is False, raise an error if the uid is not found.
        """
        # format inputs
        if isinstance(uid, int):
            uid = [uid]
        if uid is None or len(uid) == 0:
            raise ValueError("uid is None or empty.")

        # find indices
        empty_matches = []
        row_indices, column_indices = [], []
        lookup = self.get_lookup()
        for id_ in uid:
            if not lookup["uid"].is_in([id_]).any():
                empty_matches.append(id_)
            else:
                row_indices.append(lookup.filter(uid=id_)[0, "row_index"])
                column_indices.append(lookup.filter(uid=id_)[0, "column_index"])
        row_indices = [i for i in row_indices if not np.isnan(i)]
        column_indices = [i for i in column_indices if not np.isnan(i)]
        # return results
        if not allow_empty and len(empty_matches) > 0:
            raise ValueError(f"uid(s) not found: {empty_matches}")
        else:
            if len(empty_matches) > 0:
                print(
                    f"Warning: {len(empty_matches)} uid(s) not found : {empty_matches}"
                )
        return row_indices, column_indices

    def __convert_index_to_uid(
        self, index: int | list[int], axis: typing.Literal["row", "column"] = "row"
    ):
        """
        Get the body ids corresponding to index (int or list[int]).
        The indexing must be either 'row' or 'column'.

        Parameters
        ----------
        index : int or list[int]
            The index or list of indices to convert to uids.
        axis : str, optional
            The axis of the index. The default is 'row'.

        Returns
        -------
        uids : list[int]
            The body ids corresponding to the index or list of indices.
        """
        if isinstance(index, int):
            index = [index]
        lookup = self.get_lookup()
        if axis == "row":
            uids = [lookup.filter(row_index=id)[0, "uid"] for id in index]
        elif axis == "column":
            uids = [lookup.filter(column_index=id)[0, "uid"] for id in index]
        else:
            raise ValueError('The axis must be either "row" or "column".')

        return uids

    def __convert_uids_to_bodyids(self, uids: list):
        """
        Get the body ids corresponding to the uids.
        """
        lookup = self.get_lookup()
        body_ids = [lookup.filter(uid=id)[0, "body_id"] for id in uids]
        return body_ids

    def __convert_index_to_bodyid(self, index, axis="row"):
        """
        Get the body ids corresponding to index (int or list[int]).
        The indexing must be either 'row' or 'column'.

        Parameters
        ----------
        index : int or list[int]
            The index or list of indices to convert to body ids.
        axis : str, optional
            The axis of the index. The default is 'row'.

        Returns
        -------
        body_ids : list[int]
            The body ids corresponding to the index or list of indices.
        """
        if isinstance(index, int):
            index = [index]
        lookup = self.get_lookup()
        if axis == "row":
            body_ids = [lookup.filter(row_index=id)[0, "body_id"] for id in index]
        elif axis == "column":
            body_ids = [lookup.filter(column_index=id)[0, "body_id"] for id in index]
        else:
            raise ValueError('The axis must be either "row" or "column".')

        return body_ids

    def __get_uids_from_bodyids(self, body_ids: list[BodyId] | list[int]) -> list[UID]:
        """
        Get the uids corresponding to the body_ids.
        Careful that this is not a 1-on-1 mapping as a given body_id might
        be subdivided into multiple sub-neurons with their own uids.
        The returned uids are not necessarily ordered as the input body_ids.
        """
        lookup = self.lookup.filter(pl.col("body_id").is_in(body_ids))
        missing = set(body_ids) - set(lookup["body_id"].to_list())
        if len(missing) > 0:
            print(f"Warning: {len(missing)} body ids not found.")
        uids = lookup["uid"].to_list()
        return uids

    def __reorder_row_indexing(self, order: list[int]):
        """
        Reorder the indexing of the lookup table according to the order.
        The order of indices given as arguments will be mapped to [0,1,2,...].
        """
        if len(order) != self.matrix.shape[0]:
            raise ValueError("The order must have the same length as the matrix.")

        def __mapping(_x: int, _order: list[int]):
            if _x in _order:
                return _order.index(_x)
            return np.nan

        # sort the column 'row_index' according to the order
        old_order = self.lookup["row_index"].to_list()
        self.lookup.with_columns(row_index=[__mapping(x, order) for x in old_order])

    def __reorder_column_indexing(self, order: list[int]):
        """
        Reorder the indexing of the lookup table according to the order.
        The order of indices given as arguments will be mapped to [0,1,2,...].
        """
        if len(order) != self.matrix.shape[1]:
            raise ValueError("The order must have the same length as the matrix.")

        def __mapping(_x: int, _order: list[int]):
            if _x in _order:
                return _order.index(_x)
            return np.nan

        # sort he column 'column_index' according to the order
        old_order = self.lookup["column_index"].to_list()
        self.lookup.with_columns(column_index=[__mapping(x, order) for x in old_order])

    # public methods

    # --- getters
    def get_matrix(
        self,
        row_ids: Optional[list] = None,
        column_ids: Optional[list] = None,
        input_type: typing.Literal["uid", "body_id", "index"] = "uid",
    ) -> sc.sparse.csr_matrix:
        """
        Returns the adjacency matrix of the connectome.

        Returns
        -------
        scipy.sparse.csr_matrix
            The adjacency matrix of the connectome.
        """
        if row_ids is None and column_ids is None:
            return self.matrix

        # If we need to return a subset
        mat = copy.deepcopy(self.matrix)

        # Indices to keep for rows
        if input_type == "index":
            if row_ids is not None:
                rows = row_ids
            else:
                rows = self.get_row_indices()
        else:
            rows = self.get_row_indices(sub_uid=row_ids, input_type=input_type)

        # Indicies to keep for columns
        if input_type == "index":
            if column_ids is not None:
                cols = column_ids
            else:
                cols = self.get_column_indices()
        else:
            cols = self.get_column_indices(sub_uid=column_ids, input_type=input_type)

        mat = mat[rows, :][:, cols]
        return mat

    def get_lookup(self) -> pl.DataFrame:
        """
        Returns the lookup table for the equivalence between node indices and node identities.

        Returns
        -------
        polars.DataFrame
            The lookup table for the equivalence between node indices and node identities.
        """
        return self.lookup

    def get_uids(
        self,
        sub_indices: Optional[list[int]] = None,
        axis: typing.Literal["row", "column"] = "row",
    ) -> list:
        """
        Returns the uids of the nodes of the adjacency matrix.

        Parameters
        ----------
        sub_indices : list, optional
            The indices of the nodes for which the uids are returned.
            If None, the uids of all nodes are returned. The default is None.

        Returns
        -------
        list
            The uids of the nodes of the adjacency matrix.
        """
        if sub_indices is None:
            # sort the lookup by indices of the axis and return the 'uid' column
            if axis == "row":
                self.lookup.sort(by="row_index")
            elif axis == "column":
                self.lookup.sort(by="column_index")
            return self.lookup["uid"].to_list()
        return self.__convert_index_to_uid(sub_indices, axis=axis)

    def get_row_indices(
        self,
        sub_uid: Optional[list] = None,
        allow_empty: bool = True,
        input_type: typing.Literal["uid", "body_id"] = "uid",
    ) -> list:
        """
        Returns the indices of the adjacency matrix references by the body_id list.

        Parameters
        ----------
        sub_uid : list, optional
            The ids of the nodes for which the indices are returned.
            If None, the indices of all nodes are returned. The default is None.
        allow_empty : bool, optional
            If False, raises an error if the ids are not found in the lookup.
            The default is True.
        input_type : str, optional
            The type of the input uids. The default is 'uid'.
            Otherwise, the input is 'body_id'.

        Returns
        -------
        list
            The row indices of the nodes of the adjacency matrix.
        """
        if sub_uid is None:
            # sort self.lookup by 'row_index' and return the 'row_index' column
            self.lookup = self.lookup.sort(by="row_index")
            return self.lookup["row_index"].to_list()

        if input_type == "body_id":
            sub_uid = self.__get_uids_from_bodyids(sub_uid)
        rows, _ = self.__convert_uid_to_index(  # already filters out the NaN values
            sub_uid,
            allow_empty=allow_empty,
        )
        if not allow_empty and len(rows) != len(sub_uid):
            raise ValueError("Some row body ids found only in the columns.")
        return rows

    def get_column_indices(
        self,
        sub_uid: Optional[list] = None,
        allow_empty: bool = True,
        input_type: typing.Literal["uid", "body_id"] = "uid",
    ) -> list:
        """
        Returns the indices of the adjacency matrix references by the body_id
        list.

        Parameters
        ----------
        sub_uid : list, optional
            The ids of the nodes for which the indices are returned.
            If None, the indices of all nodes are returned.
            The default is None.
        allow_empty : bool, optional
            If False, raises an error if the ids are not found in
            the lookup.
            The default is True.
        input_type : str, optional
            The type of the input uids. The default is 'uid'.
            Otherwise, the input is 'body_id'.

        Returns
        -------
        list
            The column indices of the nodes of the adjacency matrix.
        """
        if sub_uid is None:
            # sort self.lookup by 'column_index' and return the 'column_index' column
            self.lookup = self.lookup.sort(by="column_index")
            return self.lookup["column_index"].to_list()
        if input_type == "body_id":
            sub_uid = self.__get_uids_from_bodyids(sub_uid)
        _, columns = self.__convert_uid_to_index(
            sub_uid,
            allow_empty=allow_empty,
        )
        if not allow_empty and len(columns) != len(sub_uid):
            raise ValueError("Some row body ids found only in the columns.")
        return columns

    def get_positive_matrix(self) -> sc.sparse.csr_matrix:
        """
        Returns the positive part of the adjacency matrix.
        """
        return self.get_matrix().multiply(self.get_matrix() > 0)

    def get_negative_matrix(self) -> sc.sparse.csr_matrix:
        """
        Returns the negative part of the adjacency matrix.
        """
        return self.get_matrix().multiply(self.get_matrix() < 0)

    # --- setters
    def restrict_from_to(
        self,
        row_ids: Optional[list] = None,
        column_ids: Optional[list] = None,
        allow_empty: bool = True,
        input_type: typing.Literal["uid", "body_id"] = "uid",
        keep_initial_order: bool = True,
    ):
        """
        Restricts the adjacency matrix and lookup table to a subset of uids,
        with different uids for rows and columns.

        Parameters
        ----------
        row_ids : list
            The list of uids to which the adjacency matrix rows are restricted.
            If None, the rows are not restricted.
        column_ids : list
            The list of uids to which the adjacency matrix columns are restricted.
            If None, the columns are not restricted.
        allow_empty : bool, optional
            If False, raises an error if the uids are not found in the lookup.
            The default is True.
        input_type : str, optional
            The type of the input ids. The default is 'uid'.
        keep_initial_order : bool, optional
            If False, the rows and columns are sorted according to the input order.
            The default is True, where the order is hte same as the existing matrix.
        """
        if row_ids is None and column_ids is None:
            return
        if isinstance(row_ids, int):
            row_ids = [row_ids]
        if isinstance(column_ids, int):
            column_ids = [column_ids]
        # convert the nodes to indices
        row_indices = self.get_row_indices(
            row_ids, allow_empty=allow_empty, input_type=input_type
        )
        column_indices = self.get_column_indices(
            column_ids,
            allow_empty=allow_empty,
            input_type=input_type,
        )

        # restrict the data to the indices
        self.__restrict_row_indices(row_indices)
        self.__restrict_column_indices(column_indices)

        # reorder the matrix and lookup if necessary
        if not keep_initial_order:
            # Get the updated indices
            row_indices = self.get_row_indices(  #
                row_ids, allow_empty=allow_empty, input_type=input_type
            )
            column_indices = self.get_column_indices(
                column_ids,
                allow_empty=allow_empty,
                input_type=input_type,
            )
            # reorder the matrix and lookup
            self.matrix = self.get_matrix()[row_indices, :][:, column_indices]
            self.__reorder_row_indexing(row_indices)
            self.__reorder_column_indexing(column_indices)

    def restrict_nodes(
        self,
        nodes: list[UID],
        allow_empty: bool = True,
        keep_initial_order: bool = True,
    ):
        """
        Restricts the adjacency matrix to a subset of nodes (uids).

        Parameters
        ----------
        nodes : list
            The list of nodes to which the adjacency matrix is restricted.
        allow_empty : bool, optional
            If False, raises an error if the nodes are not found in the lookup.
            The default is True.
        """
        # restrict the matrix amd lookup to the indices
        self.restrict_from_to(
            row_ids=nodes,
            column_ids=nodes,
            allow_empty=allow_empty,
            input_type="uid",
            keep_initial_order=keep_initial_order,
        )

    def restrict_rows(
        self,
        rows: list,
        allow_empty: bool = True,
        input_type: typing.Literal["uid", "body_id"] = "uid",
    ):
        """
        Restricts the adjacency matrix to a subset of rows.

        Parameters
        ----------
        rows : list
            The list of rows to which the adjacency matrix is restricted.
        allow_empty : bool, optional
            If False, raises an error if the rows are not found in the lookup.
            The default is True.
        input_type : str, optional
            The type of the input uids. The default is 'uid'.
            Otherwise, the input is 'body_id'.
        """
        # convert the rows to matrix indices
        r_i = self.get_row_indices(rows, allow_empty=allow_empty, input_type=input_type)
        # restrict the matrix amd lookup to the indices
        self.__restrict_row_indices(r_i, allow_empty=allow_empty)

    def restrict_columns(
        self,
        columns: list,
        allow_empty: bool = True,
        input_type: typing.Literal["uid", "body_id"] = "uid",
    ):
        """
        Restricts the adjacency matrix to a subset of columns (body_ids).

        Parameters
        ----------
        columns : list
            The list of columns to which the adjacency matrix is restricted.
        allow_empty : bool, optional
            If False, raises an error if the columns are not found in the lookup.
            The default is True.
        """
        # convert the columns to indices
        c_i = self.get_column_indices(
            columns, allow_empty=allow_empty, input_type=input_type
        )
        # restrict the matrix amd lookup to the indices
        self.__restrict_column_indices(c_i, allow_empty=allow_empty)

    # --- processing (modifies in place)
    def power_n(self, n: int):
        """
        Computes the n-th power of the adjacency matrix of the connectome.

        Parameters
        ----------
        n : int
            The power to which the adjacency matrix is raised.
        """
        # check if the matrix is still sparse
        matrix_ = self.get_matrix()
        self.matrix = matrix_utils.connections_at_n_hops(matrix_, n)

    def within_power_n(self, n: int):
        """
        Computes the sum of the i-th power of the adjacency matrix of the
        connectome up to n.

        Parameters
        ----------
        n : int
            The power to which the adjacency matrix is raised.
        """
        matrix_ = self.get_matrix()
        self.matrix = matrix_utils.connections_up_to_n_hops(matrix_, n)

    def cluster_hierarchical_unilateral(self, axis="row"):
        """
        Clusters the adjacency matrix using hierarchical clustering.
        """
        matrix_ = self.get_matrix()
        if axis == "row":
            pass
        elif axis == "column":
            matrix_ = matrix_.transpose()
        else:
            raise ValueError('The axis must be either "row" or "column".')
        # clustering computation
        matrix_ = matrix_.todense()
        dendrogram = sch.dendrogram(sch.linkage(matrix_, method="ward"), no_plot=True)
        order = dendrogram["leaves"]
        # reorder the matrix and indexing
        if axis == "row":
            self.matrix = self.get_matrix()[order, :]  # sparse matrix again
            self.__reorder_row_indexing(order=order)
        else:
            self.matrix = self.get_matrix()[:, order]
            self.__reorder_column_indexing(order=order)
        return

    def hierarchical_clustering(self, cutoff: float = 0.5):
        """
        Clusters the adjacency matrix using hierarchical clustering.
        Here the clustering problem is seen as a representation of vector n in
        feature space m. Therefore similar connections to similar neurons
        yields a small effective distance and therefore clusering.
        If applied to distance/similarity matrix, use the similarity
        representation of the matrix given the input type.
        """
        matrix_ = self.get_matrix()
        matrix_ = matrix_.todense()

        # Compute the linkage matrix
        Z = sch.linkage(matrix_, method="ward")

        # Compute the clusters
        labels = sch.fcluster(Z, t=cutoff, criterion="distance")
        clusters = defaultdict(list)  # Group points by their cluster labels
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        clusters = list(clusters.values())

        # Reorder the matrix and indexing
        dendrogram = sch.dendrogram(Z, no_plot=True)
        order = dendrogram["leaves"]
        self.matrix = self.get_matrix()[order, :][:, order]
        self.__reorder_row_indexing(order=order)
        self.__reorder_column_indexing(order=order)
        return clusters

    def markov_clustering(self, inflation: int = 3, iterations: int = 100):
        """
        Applies the Markov clustering algorithm to the matrix.

        Parameters
        ----------
        inflation : int, optional
            The inflation parameter of the Markov clustering algorithm.
        iterations : int, optional
            The number of iterations of the Markov clustering algorithm.
            The default is 100.

        Returns
        -------
        clusters : list
            The list of clusters detected in the matrix.
        """
        matrix = self.get_matrix()
        clusters = matrix_utils.markov_clustering(
            matrix, inflation=inflation, iterations=iterations
        )
        new_order = [i for c in clusters for i in c]
        self.matrix = self.get_matrix()[new_order, :][:, new_order]
        self.__reorder_row_indexing(new_order)
        self.__reorder_column_indexing(new_order)

        # update the indexing
        mapping = {new_order[i]: i for i in range(len(new_order))}
        # apply the mapping to clusters
        updated_clusters = [[mapping[i] for i in c] for c in clusters]
        return updated_clusters

    def absolute(self):
        """
        Computes the absolute value of the adjacency matrix.
        """
        self.matrix = np.absolute(self.get_matrix()).tocsr()
        return

    def square_positive_paths_only(self) -> sc.sparse.csr_matrix:
        """
        Squares the adjacency matrix, where the sum over a_ik*a_kj
        is only computed if the product of a_ik and a_kj is positive.
        In practice this yield the sum of paths length 2 that are either
        twice excitatory or twice inhibitory.
        """
        matrix = copy.deepcopy(self.get_matrix())
        matrix_pos = matrix.multiply(matrix > 0)
        matrix_neg = matrix.multiply(matrix < 0)
        matrix_positive_paths = matrix_pos @ matrix_pos + matrix_neg @ matrix_neg
        self.matrix = matrix_positive_paths  # indexing remains identical

    def square_negative_paths_only(self) -> sc.sparse.csr_matrix:
        """
        Squares the adjacency matrix, where the sum over a_ik*a_kj
        is only computed if the product of a_ik and a_kj is negative.
        In practice this yield the sum of paths length 2 that are either
        excitatory then inhibitory or inhibitory then excitatory.
        """
        matrix = copy.deepcopy(self.get_matrix())
        matrix_pos = matrix.multiply(matrix > 0)
        matrix_neg = matrix.multiply(matrix < 0)
        matrix_negative_paths = matrix_pos @ matrix_neg + matrix_neg @ matrix_pos
        self.matrix = matrix_negative_paths

    def invert_elements(self, inv_of_zero=np.nan) -> sc.sparse.csr_matrix:
        """
        For each element of the adjacency matrix, invert the value to 1/value.
        """
        matrix = copy.deepcopy(self.get_matrix())
        matrix.data = 1 / matrix.data
        matrix.data = np.nan_to_num(matrix.data, nan=inv_of_zero)
        self.matrix = matrix
        return

    # --- computations (returns something)
    def list_downstream_neurons(self, uids: UID | list[UID]) -> list[UID]:
        """
        Get the downstream neurons of the input neurons.
        """
        if isinstance(uids, int):
            uids = [uids]
        cmatrix_copy = copy.deepcopy(self)
        cmatrix_copy.restrict_rows(uids)
        matrix = cmatrix_copy.get_matrix()
        non_zero_columns = set(matrix.nonzero()[1])  # columns with non-zero values
        downstream_uids = cmatrix_copy.get_uids(
            sub_indices=list(non_zero_columns), axis="column"
        )
        return downstream_uids

    def list_upstream_neurons(self, uids: UID | list[UID]) -> list[UID]:
        """
        Get the upstream neurons of the input neurons.
        """
        if isinstance(uids, int):
            uids = [uids]
        cmatrix_copy = copy.deepcopy(self)
        cmatrix_copy.restrict_columns(uids)
        matrix = cmatrix_copy.get_matrix()
        non_zero_rows = set(matrix.nonzero()[0])
        upstream_uids = cmatrix_copy.get_uids(
            sub_indices=list(non_zero_rows), axis="row"
        )
        return upstream_uids

    def list_neurons_upstream_set(
        self,
        uids: list[UID] | UID,
        ratio: float = 0.5,
    ) -> list[UID]:
        """
        List all the neurons that are upstream of {ratio}% of the input neurons.
        Useful to find neurons that frequently input to a group for instance.

        Parameters
        ----------
        uids : list
            The list of uids for which the upstream neurons are computed.
        ratio : float, optional
            The ratio of neurons to consider. The default is 0.5.

        Returns
        -------
        upstream_neurons : list
            The list of neurons that are upstream of {ratio}% of the input neurons.
        """
        if isinstance(uids, int):
            uids = [uids]
        # get all the upstream neurons
        all_upstream = self.list_upstream_neurons(uids)
        # for each, very the ratio
        selected_upstream = []
        for uid in all_upstream:
            down = self.list_downstream_neurons(uid)
            if len(set(down).intersection(uids)) / len(uids) >= ratio:
                selected_upstream.append(uid)
        return selected_upstream

    def build_distance_matrix(
        self,
        method: typing.Literal[
            "cosine_in", "cosine_out", "cosine", "euclidean"
        ] = "cosine",
    ):
        """
        Build a similarity matrix from the adjacency matrix.

        Parameters
        ----------
        similarity_function : function
            The similarity function to apply to the adjacency matrix.

        Returns
        -------
        similarity_matrix : CMatrix object
            The similarity matrix, and the associated lookup table.
        """
        new_cmatrix = copy.deepcopy(self)
        similarity_matrix = matrix_utils.build_distance_matrix(
            new_cmatrix.get_matrix(), method
        )
        new_cmatrix.matrix = similarity_matrix
        return new_cmatrix

    def cosine_similarity(self, in_out: str = "both"):
        """
        Compute the cosine similarity of the adjacency matrix.

        Parameters
        ----------
        in_out : str
            Whether to use the input or output connectivity to define the metric
            By default, 'both' concatenates both connections

        Returns
        ----------
        new_cmatrix : CMatrix object
        """
        match in_out:
            case "both":
                new_cmatrix = self.build_distance_matrix(method="cosine")
            case "in":
                new_cmatrix = self.build_distance_matrix(method="cosine_in")
            case "out":
                new_cmatrix = self.build_distance_matrix(method="cosine_out")
            case _:
                raise ValueError(
                    'The in_out parameter must be either "both", "in" or "out".'
                )
        new_cmatrix.matrix = 1 - new_cmatrix.matrix
        return new_cmatrix

    def detect_clusters(
        self,
        distance: typing.Literal[
            "cosine_in", "cosine_out", "cosine", "euclidean"
        ] = "cosine",
        method: typing.Literal[
            "hierarchical", "markov", "hierarchical_linkage", "DBSCAN"
        ] = "markov",
        cluster_size_cutoff: int = 2,
        show_plot: bool = False,
        cluster_data_type: typing.Literal["uid", "index", "body_id"] = "uid",
        cluster_on_subset: list[UID] | None = None,
        **kwargs,
    ):
        """
        Detects the clusters in the adjacency matrix.

        Parameters
        ----------
        distance : str
            The distance metric used to define the node similarity.
            The default is the bilateral 'cosine'.
        method : str
            The method used to detect the clusters. The default is 'hierarchical'.
        cluster_size_cutoff : int
            The minimum number of nodes in a cluster. The default is 2.
        show_plot : bool
            If True, shows the plot of the adjacency matrix with the cluster boundaries.
            The default is False.
        cluster_data_type : str
            The type of data returned in the clusters. The default is 'uid'.
        cluster_on_subset : list[int] | None
            If not None, the similarity matrix is computed on the entire set,
            but the clustering is performed on the subset of nodes defined by
            the list. Typical use case: the similarity matrix is computed on the
            set of motor and premotor neurons, but the clustering is performed
            on motor neurons only.
        **kwargs: parameters for specific clustering methods. Can include (not exhaustive):
            - cutoff : float, for method=hierarchical
            The cutoff value used to define the clusters. The default is 0.5.
            Actual meaning depends on the method used.
            - inflation : float, for method=markov
            Inflation parameter that effectively results in different cluster sizes.


        Returns
        -------
        clusters : list
            The list of clusters detected in the adjacency matrix.
        """
        # define the similarity matrix
        new_cmatrix = self.build_distance_matrix(method=distance)

        # restrict the similarity matrix to the subset of nodes
        if cluster_on_subset is not None:
            new_cmatrix.restrict_nodes(cluster_on_subset)
            print("Warning: the cmatrix is reduced to the subset clustered on.")

        # cluster the similarity matrix
        match method:  # to be completed
            case "hierarchical_linkage":
                # convert distance to similarity
                dense_ = new_cmatrix.matrix.todense()
                dense_ = 1 - dense_
                new_cmatrix.matrix = sc.sparse.csr_matrix(dense_)
                # hierarchical clustering, returns the tree level clusters
                cutoff = kwargs.get("cutoff", 0.5)  # Default value if not provided
                clusters = new_cmatrix.hierarchical_clustering(cutoff=cutoff)
            case "markov":
                # convert distance to similarity
                dense_ = new_cmatrix.matrix.todense()
                dense_ = 1 - dense_
                new_cmatrix.matrix = sc.sparse.csr_matrix(dense_)
                inflation = kwargs.get("inflation", 3.0)
                iterations = kwargs.get("iterations", 100)
                all_clusters = new_cmatrix.markov_clustering(
                    inflation=inflation, iterations=iterations
                )
                clusters = [c for c in all_clusters if len(c) >= cluster_size_cutoff]
            case "hierarchical":
                cutoff = kwargs.get("cutoff", 0.5)  # Default value if not provided
                # convert distance to similarity
                dense_ = new_cmatrix.matrix.todense()
                dense_ = 1 - dense_
                new_cmatrix.matrix = sc.sparse.csr_matrix(dense_)
                # hierarchical clustering, returns the clusters obtained by
                # scanning the sorted matrix for the cutoff value
                _ = new_cmatrix.hierarchical_clustering()
                # detect the clusters
                clustered_mat = new_cmatrix.get_matrix().todense()
                # replace NaN with 0
                clustered_mat = np.nan_to_num(clustered_mat, nan=0)
                clusters = []
                start_cluster = 0
                for i in range(clustered_mat.shape[0]):
                    average_similarity = np.mean(clustered_mat[i, start_cluster:i])
                    if average_similarity < cutoff:
                        new_cluster = list(
                            np.linspace(
                                start_cluster, i - 1, i - start_cluster, dtype=int
                            )
                        )
                        if len(new_cluster) >= cluster_size_cutoff:
                            clusters.append(new_cluster)
                        start_cluster = i
                last_cluster = list(
                    np.linspace(
                        start_cluster,
                        clustered_mat.shape[0] - 1,
                        clustered_mat.shape[0] - start_cluster,
                        dtype=int,
                    )
                )
                if len(last_cluster) >= cluster_size_cutoff:
                    clusters.append(last_cluster)
            case "DBSCAN":
                # works on the distance matrix, not similarity
                # density-based clustering
                # Apply DBSCAN clustering
                metric = kwargs.get("metric", "precomputed")
                eps = kwargs.get("eps", 0.5)
                min_samples = kwargs.get("min_samples", 2)
                db = DBSCAN(metric=metric, eps=eps, min_samples=min_samples)
                distance_matrix = np.array(new_cmatrix.get_matrix().todense())
                labels = db.fit_predict(distance_matrix)

                # Group points by their cluster labels
                clusters = defaultdict(list)
                for idx, label in enumerate(labels):
                    if label != -1:  # Ignore noise points
                        clusters[label].append(idx)

                # Convert to list of lists
                clusters = list(clusters.values())
            case _:
                raise ValueError(
                    "CMatrix::detect_cluster() -> The method is not recognised."
                )

        # convert cluster indices to the relevant data type
        match cluster_data_type:
            case "uid":
                return_clusters = [
                    new_cmatrix.get_uids(sub_indices=cluster, axis="row")
                    for cluster in clusters
                ]
            case "index":
                return_clusters = clusters
            case "body_id":
                return_clusters = [
                    new_cmatrix.__convert_index_to_bodyid(cluster, axis="row")
                    for cluster in clusters
                ]
            case _:
                raise ValueError(
                    "CMatrix::detect_cluster() -> The cluster_data_type is not recognised."
                )

        if show_plot:
            # draw a matrix where all entries are white except for the boundaries
            # between clusters as defined by the number of elements in each list in
            # the clusters list.

            # create a matrix of zeros
            max_cluster_element = max([cluster[-1] for cluster in clusters]) + 1
            mat = np.zeros((max_cluster_element, max_cluster_element))
            # draw the boundaries between clusters
            for cluster in clusters:
                mat[cluster[0] : cluster[-1] + 1, cluster[0] : cluster[-1] + 1] = 1
            ax, title = new_cmatrix.imshow(savefig=False)
            ax.imshow(mat, cmap="binary", alpha=0.3)
            full_title = os.path.join(self.CR.get_plots_dir(), title)
            plt.savefig(full_title)
            plt.close()

        return new_cmatrix, return_clusters, clusters

    # --- visualisation
    def spy(
        self,
        title: str = "test",
        ax: matplotlib.axes.Axes | None = None,
        savefig: bool = False,
    ):
        """
        Visualises the sparsity pattern of the adjacency matrix.

        Parameters
        ----------
        title : str, optional
            The title of the visualisation. The default is None.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=params.FIGSIZE)
        _ = matrix_design.spy(self.get_matrix(), title=title, ax=ax)
        if savefig:
            title_ = os.path.join(self.CR.get_plots_dir(), title + "_spy.pdf")
            plt.savefig(title_)

    @typing.overload
    def imshow(
        self,
        title: str = "test",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap=params.diverging_heatmap,
        ax: matplotlib.axes.Axes | None = None,
        *,
        snippet_up_to: int | None = None,
        log_scale: bool = False,
        savefig: typing.Literal[True] = True,
    ) -> None: ...
    @typing.overload
    def imshow(
        self,
        title: str = "test",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap=params.diverging_heatmap,
        ax: matplotlib.axes.Axes | None = None,
        *,
        snippet_up_to: int | None = None,
        log_scale: bool = False,
        savefig: typing.Literal[False],
    ) -> tuple[matplotlib.axes.Axes, str]: ...

    def imshow(
        self,
        title: str = "test",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap=params.diverging_heatmap,
        ax: matplotlib.axes.Axes | None = None,
        *,
        snippet_up_to: int | None = None,
        log_scale: bool = False,
        savefig: bool = True,
    ):
        """
        Visualises the adjacency matrix with a colorbar.

        Parameters
        ----------
        title : str, optional
            The title of the visualisation. The default is None.
        vmin : float, optional
            The minimum value of the colorbar. The default is None.
        vmax : float, optional
            The maximum value of the colorbar. The default is None.
        cmap : str, optional
            The colormap of the visualisation. The default is params.blue_heatmap.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the visualisation. The default is None.
        snippet_up_to : int, optional
            The maximum number of rows and columns to display. The default is None.
        log_scale : bool, optional
            If True, plot the connection matrix on a log scale.
        savefig : bool, optional
            If True, saves the figure. The default is True.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=params.FIGSIZE)
        mat = copy.deepcopy(self.get_matrix())
        if snippet_up_to is not None:
            mat = mat[:snippet_up_to, :][:, :snippet_up_to]
        if log_scale:
            # modify mat.data by applying sign(dat) * log (abs(data))
            mat.data = np.sign(mat.data) * np.log10(np.abs(mat.data))
        ax = matrix_design.imshow(
            mat,
            title=title,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
        )
        if savefig:
            title_ = os.path.join(self.CR.get_plots_dir(), title + "_imshow.pdf")
            plt.savefig(title_)
        else:
            return ax, title

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

import scipy as sc
import scipy.cluster.hierarchy as sch
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import copy

import params
import utils.matrix_utils as matrix_utils
import utils.matrix_design as matrix_design

class CMatrix:
    def __init__(
            self,
            matrix: sc.sparse.csr_matrix,
            lookup: pd.DataFrame,
        ):
        '''
        Initialises the cmatrix class, standing for connectome matrix.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The adjacency matrix of the connectome.
        lookup : pandas.DataFrame
            The lookup table for the equivalence between node indices and node identities. 
        '''
        # verify that the matrix is a csr matrix
        if not sc.sparse.issparse(matrix):
            matrix = sc.sparse.csr_matrix(matrix)
        if not isinstance(matrix, sc.sparse.csr_matrix):
            matrix = matrix.tocsr()
        self.matrix = matrix


        # verify that the lookup is a pandas dataframe
        if not isinstance(lookup, pd.DataFrame):
            raise ValueError('The lookup must be a pandas dataframe.')
        
        # duplicate the 'index' column into a 'row_index' and 'column_index' column
        lookup = lookup.copy()
        lookup['row_index'] = lookup['index'].values
        lookup['column_index'] = lookup['index'].values
        self.lookup = lookup.drop(columns='index')
         
        # verify that the lookup includes indices up to the length of the matrix
        if not all(self.lookup['row_index'].isin(range(self.matrix.shape[0]))):
            raise ValueError(
                'The lookup must include row indices up to the length of the matrix.'
                )
        if not all(self.lookup['column_index'].isin(range(self.matrix.shape[1]))):
            raise ValueError(
                'The lookup must include column indices up to the length of the matrix.'
                )

    # private methods
    def __update_indexing(self):
        '''
        Updates the indexing of the lookup table to match the new matrix.
        Everytime the matrix is restricted, the lookup table must be updated and
        the indexing shifted back to a continuous range starting from 0.
        '''
        lookup = self.get_lookup().copy()

        # rows
        lookup.sort_values(by='row_index', inplace=True)
        n_rows = lookup['row_index'].count()
        n_nan = len(lookup) - n_rows
        new_row_indices = list(range(n_rows))
        new_row_indices.extend(np.nan*np.ones(n_nan))
        lookup['row_index'] = new_row_indices

        # columns
        lookup = lookup.sort_values(by='column_index')
        n_columns = lookup['column_index'].count()
        n_nan = len(lookup) - n_columns
        new_column_indices = list(range(n_columns))
        new_column_indices.extend(np.nan*np.ones(n_nan))
        lookup['column_index'] = new_column_indices

        # clean up unindexed uids
        lookup.dropna(
            axis='index',
            subset=['row_index', 'column_index'],
            how='all',
            inplace=True,
            )
        # verify that the lookup includes indices up to the length of the matrix
        if n_rows != self.matrix.shape[0]:
            raise ValueError(
                'The lookup must include row indices up to the length of the matrix.'
                )
        if n_columns != self.matrix.shape[1]:
            raise ValueError(
                'The lookup must include column indices up to the length of the matrix.'
                )
        self.lookup = lookup
        return

    def __restrict_row_indices(self, indices: list, allow_empty: bool = True):
        '''
        Restricts the adjacency matrix to a subset of indices.

        Parameters
        ----------
        indices : list
            The list of indices to which the adjacency matrix is restricted.
        allow_empty : bool, optional
            If False, raises an error if the indices are not found in the lookup.
            The default is True.
        '''
        # replace elements of 'row_index' not in indices with NaN
        new_vals = [
            i if i in indices else np.nan
            for i in self.get_row_indices() # sorted numerically!
            ]
        restrict_indices = [ # keep the original index sorting
            i for i in self.get_row_indices()
            if i in indices
        ]
        
        # restrict the indices to those defined in the lookup
        if (
            not allow_empty
            and np.count_nonzero(~np.isnan(new_vals)) != len(indices)
        ):
            raise ValueError("Some row indices not found in the lookup.")

        self.lookup['row_index'] = new_vals

        # restrict the matrix to the indices
        self.matrix = self.matrix[restrict_indices, :]
        self.__update_indexing()
        return
    
    def __restrict_column_indices(
            self,
            indices: list,
            allow_empty: bool = True
            ):
        '''
        Restricts the adjacency matrix to a subset of indices.

        Parameters
        ----------
        indices : list
            The list of indices to which the adjacency matrix is restricted.
        allow_empty : bool, optional
            If False, raises an error if the indices are not found in the lookup.
            The default is True.
        '''
        # restrict the indices to those defined in the lookup
        defined_columns = [
            i for i in indices
            if i in self.get_column_indices() # sorted numerically!
        ]
        if not allow_empty and len(defined_columns) != len(indices):
            raise ValueError("Some column indices not found in the lookup.")
        indices = defined_columns
        
        # replace elements of 'column_index' not in indices with NaN
        new_vals = [
            i if i in indices else np.nan
            for i in self.get_column_indices()
            ]
        restrict_indices = [ # keep the original index sorting
            i for i in self.get_column_indices()
            if i in indices
        ]
        self.lookup['column_index'] = new_vals
        # restrict the matrix to the indices
        self.matrix = self.matrix[:, restrict_indices]
        self.__update_indexing()
        return
    
    def __convert_uid_to_index(self, uid: list, allow_empty=True):
        """
        input: uid
        output: indices
        if allow_empty is False, raise an error if the uid is not found.
        """
        # format inputs
        if uid is None or len(uid) == 0:
            raise ValueError("uid is None or empty.")
        if isinstance(uid, int):
            uid = [uid]

        # find indices
        empty_matches = []
        row_indices, column_indices = [], []
        lookup = self.get_lookup()
        for id_ in uid:
            if not lookup["uid"].isin([id_]).any():
                empty_matches.append(id_)
            else:
                row_indices.append(
                    lookup.loc[lookup["uid"] == id_].row_index.values[0]
                    )
                column_indices.append(
                    lookup.loc[lookup["uid"] == id_].column_index.values[0]
                    )
        row_indices = [i for i in row_indices if not pd.isna(i)]
        column_indices = [i for i in column_indices if not pd.isna(i)]
        # return results
        if not allow_empty and len(empty_matches) > 0:
            raise ValueError(f"uid(s) not found: {empty_matches}")
        else:
            if len(empty_matches) > 0:
                print(f"Warning: {len(empty_matches)} uid(s) not found.")
        return row_indices, column_indices

    def __convert_index_to_uid(self, index, axis='row'):
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
        if axis == 'row':
            uids = [
                lookup.loc[lookup["row_index"] == id].uid.values[0]
                for id in index
                ]
        elif axis == 'column':
            uids = [
                lookup.loc[lookup["column_index"] == id].uid.values[0]
                for id in index
                ]
        else:
            raise ValueError('The axis must be either "row" or "column".')

        return uids

    def __convert_uids_to_bodyids(self, uids: list):
        """
        Get the body ids corresponding to the uids.
        """
        lookup = self.get_lookup()
        body_ids = [
            lookup.loc[lookup["uid"] == id].body_id.values[0]
            for id in uids
            ]
        return body_ids

    def __get_uids_from_bodyids(self, body_ids: list):
        """
        Get the uids corresponding to the body_ids.
        Careful that this is not a 1-on-1 mapping as a given body_id might
        be subdivided into multiple sub-neurons with their own uids.
        The returned uids are not necessarily ordered as the input body_ids.
        """
        lookup = self.get_lookup().copy()
        lookup = lookup[lookup['body_id'].isin(body_ids)]
        missing = set(body_ids) - set(lookup['body_id'].tolist())
        if len(missing) > 0:
            print(f"Warning: {len(missing)} body ids not found.")
        uids = lookup['uid'].tolist()
        return uids

    def __reorder_row_indexing(self, order):
        """
        Reorder the indexing of the lookup table according to the order.
        The order of indices given as arguments will be mapped to [0,1,2,...].
        """
        lookup = self.get_lookup().copy()
        mapping = dict(zip(order, range(len(order))))
        mapping[np.nan] = np.nan
        # sort he column 'row_index' according to the order
        lookup['row_index'] = lookup['row_index'].map(
            mapping
            ).astype(int)
        self.lookup = lookup
        return
    
    def __reorder_column_indexing(self, order):
        """
        Reorder the indexing of the lookup table according to the order.
        The order of indices given as arguments will be mapped to [0,1,2,...].
        """
        lookup = self.get_lookup().copy()
        mapping = dict(zip(order, range(len(order))))
        mapping[np.nan] = np.nan
        # sort he column 'column_index' according to the order
        lookup['column_index'] = lookup['column_index'].map(
            mapping
            )
        self.lookup = lookup
        return
        

    # public methods
    
    # --- getters
    def get_matrix(self) -> sc.sparse.csr_matrix:
        '''
        Returns the adjacency matrix of the connectome.

        Returns
        -------
        scipy.sparse.csr_matrix
            The adjacency matrix of the connectome.
        '''
        return self.matrix
    
    def get_lookup(self) -> pd.DataFrame:
        '''
        Returns the lookup table for the equivalence between node indices and node identities.

        Returns
        -------
        pandas.DataFrame
            The lookup table for the equivalence between node indices and node identities.
        '''
        return self.lookup
        
    def get_uids(self, sub_indices: list = None, axis: str = 'row') -> list:
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
            return self.lookup['uid'].tolist()
        return self.__convert_index_to_uid(
            sub_indices,
            axis=axis
            )
    
    def get_row_indices(
            self,
            sub_uid: list = None,
            allow_empty: bool = True,
            input_type: str = 'uid',
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
            self.lookup.sort_values(by='row_index', inplace=True)
            return self.lookup['row_index'].tolist()
        if input_type == 'body_id':
            sub_uid = self.__get_uids_from_bodyids(sub_uid)
        rows, _ = self.__convert_uid_to_index( # already filters out the NaN values
            sub_uid,
            allow_empty=allow_empty,
            )
        if not allow_empty and len(rows) != len(sub_uid):
            raise ValueError("Some row body ids found only in the columns.")
        return rows

    def get_column_indices(
            self,
            sub_uid: list = None,
            allow_empty: bool = True,
            input_type: str = 'uid',
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
            self.lookup.sort_values(by='column_index', inplace=True)
            return self.lookup['column_index'].tolist()
        if input_type == 'body_id':
            sub_uid = self.__get_uids_from_bodyids(sub_uid)
        _, columns = self.__convert_uid_to_index(
            sub_uid,
            allow_empty=allow_empty,
            )
        if not allow_empty and len(columns) != len(sub_uid):
            raise ValueError("Some row body ids found only in the columns.")
        return columns

    # --- setters
    def restrict_from_to(
            self,
            row_ids: list = None,
            column_ids: list = None,
            allow_empty: bool = True,
            input_type: str = 'body_id',
        ):
        '''
        Restricts the adjacency matrixand lookup table to a subset of uids,
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
            The type of the input uids. The default is 'body_id'. It will cover all uids 
            that have the matching 'body_id' in the lookup.
            Otherwise, the input is 'uid'.
        '''
        if row_ids is None and column_ids is None:
            return
        if isinstance(row_ids, int):
            row_ids = [row_ids]
        if isinstance(column_ids, int):
            column_ids = [column_ids]
        # convert the nodes to indices
        row_indices = self.get_row_indices(
            row_ids,
            allow_empty=allow_empty,
            input_type=input_type
            )
        column_indices = self.get_column_indices(
            column_ids,
            allow_empty=allow_empty,
            input_type=input_type,
            )
                    
        # restrict the data to the indices
        self.__restrict_row_indices(row_indices)
        self.__restrict_column_indices(column_indices)
        return
    
    def restrict_nodes(self, nodes: list, allow_empty: bool = True):
        '''
        Restricts the adjacency matrix to a subset of nodes (uids).

        Parameters
        ----------
        nodes : list
            The list of nodes to which the adjacency matrix is restricted.
        allow_empty : bool, optional
            If False, raises an error if the nodes are not found in the lookup.
            The default is True.
        '''
        # restrict the matrix amd lookup to the indices
        self.restrict_from_to(
            row_ids=nodes,
            column_ids=nodes,
            allow_empty=allow_empty,
            input_type='uid'
            )
        return

    def restrict_rows(
            self,
            rows: list,
            allow_empty: bool = True,
            input_type: str = 'uid',
            ):
        '''
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
        '''
        # convert the rows to matrix indices
        r_i = self.get_row_indices(
            rows,
            allow_empty=allow_empty,
            input_type=input_type
            )
        # restrict the matrix amd lookup to the indices
        self.__restrict_row_indices(r_i, allow_empty=allow_empty)
        return
    
    def restrict_columns(
            self,
            columns: list,
            allow_empty: bool = True,
            input_type: str = 'uid',
            ):
        '''
        Restricts the adjacency matrix to a subset of columns (body_ids).

        Parameters
        ----------
        columns : list
            The list of columns to which the adjacency matrix is restricted.
        allow_empty : bool, optional
            If False, raises an error if the columns are not found in the lookup.
            The default is True.
        '''
        # convert the columns to indices
        c_i = self.get_column_indices(
            columns,
            allow_empty=allow_empty,
            input_type=input_type
            )
        # restrict the matrix amd lookup to the indices
        self.__restrict_column_indices(c_i, allow_empty=allow_empty)
        return

    # --- processing
    def power_n(self, n:int):
        '''
        Computes the n-th power of the adjacency matrix of the connectome.

        Parameters
        ----------
        n : int
            The power to which the adjacency matrix is raised.
        '''
        # check if the matrix is still sparse
        matrix_ = self.get_matrix()
        self.matrix = matrix_utils.connections_at_n_hops(matrix_, n)

    def within_power_n(self, n:int):
        '''
        Computes the sum of the i-th power of the adjacency matrix of the
        connectome up to n.

        Parameters
        ----------
        n : int
            The power to which the adjacency matrix is raised.
        '''
        matrix_ = self.get_matrix()
        self.matrix = matrix_utils.connections_up_to_n_hops(matrix_, n)

    def cluster_hierarchical_unilateral(self, axis='row'):
        '''
        Clusters the adjacency matrix using hierarchical clustering.
        '''
        matrix_ = self.get_matrix()
        if axis == 'row':
            pass
        elif axis == 'column':
            matrix_ = matrix_.transpose()
        else:
            raise ValueError('The axis must be either "row" or "column".')
        # clustering computation
        matrix_ = matrix_.todense()
        dendrogram = sch.dendrogram(
            sch.linkage(matrix_, method="ward"), no_plot=True
        )
        order = dendrogram["leaves"]
        # reorder the matrix and indexing
        if axis == 'row':
            self.matrix = self.get_matrix()[order,:] # sparse matrix again
            self.__reorder_row_indexing(order=order)
        else:
            self.matrix = self.get_matrix()[:,order]
            self.__reorder_column_indexing(order=order)
        return

    def absolute(self):
        '''
        Computes the absolute value of the adjacency matrix.
        '''
        self.matrix = np.absolute(self.get_matrix()).tocsr()
        return
    
    # --- computations
    def list_downstream_neurons(self, uids: list[int]):
        """
        Get the downstream neurons of the input neurons.
        """
        if isinstance(uids, int):
            uids = [uids]
        cmatrix_copy = copy.deepcopy(self)
        cmatrix_copy.restrict_rows(uids)
        matrix = cmatrix_copy.get_matrix()
        non_zero_columns = set(matrix.nonzero()[1]) # columns with non-zero values
        downstream_uids = cmatrix_copy.get_uids(
            sub_indices=non_zero_columns,
            axis='column'
            )
        return downstream_uids
    
    # --- visualisation
    def spy(self, title:str='test'):
        '''
        Visualises the sparsity pattern of the adjacency matrix.

        Parameters
        ----------
        title : str, optional
            The title of the visualisation. The default is None.
        '''
        _ = matrix_design.spy(self.get_matrix(), title=title) 
        title_ = os.path.join(params.PLOT_DIR, title + "_spy.pdf")
        plt.savefig(title_)
        return
    
    def imshow(
            self,
            title:str='test',
            vmax:float=None,
            cmap=params.diverging_heatmap,
        ):
        '''
        Visualises the adjacency matrix with a colorbar.

        Parameters
        ----------
        title : str, optional
            The title of the visualisation. The default is None.
        vmax : float, optional
            The maximum value of the colorbar. The default is None.
        cmap : str, optional
            The colormap of the visualisation. The default is params.blue_heatmap.
        '''
        _ = matrix_design.imshow(
            self.get_matrix(),
            title=title,
            vmax=vmax,
            cmap=cmap,
            )
        title_ = os.path.join(params.PLOT_DIR, title + "_imshow.pdf")
        plt.savefig(title_)
        return


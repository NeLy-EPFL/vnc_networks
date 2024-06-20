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
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

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
        # verify that the lookup has the correct columns
        if not 'index' in lookup.columns or not 'body_id' in lookup.columns:
            raise ValueError('The lookup must have a column named "index"\
                             and a column named "body_id".')
        # duplicate the 'index' column into a 'row_index' and 'column_index' column
        lookup['row_index'] = lookup['index']
        lookup['column_index'] = lookup['index']
        lookup = lookup.drop(columns=['index'])
        self.lookup = lookup
        # verify that the lookup includes indices up to the length of the matrix
        if not all(self.lookup['row_index'].isin(range(self.matrix.shape[0]))):
            raise ValueError(
                'The lookup must include row indices up to the length of the matrix.'
                )
        if not all(self.lookup['column_index'].isin(range(self.matrix.shape[1]))):
            raise ValueError(
                'The lookup must include column indices up to the length of the matrix.'
                )
        return
    

    # private methods
    def __update_indexing(self):
        '''
        Updates the indexing of the lookup table to match the new matrix.
        Everytime the matrix is restricted, the lookup table must be updated and
        the indexing shifted back to a continuous range starting from 0.
        '''
        lookup = self.get_lookup()
        # rows
        lookup = lookup.sort_values(by='row_index')
        n_rows = lookup['row_index'].count()
        n_nan = len(lookup) - n_rows
        new_row_indices = range(n_rows).extend([np.nan]*n_nan)
        lookup['row_index'] = new_row_indices
        # columns
        lookup = lookup.sort_values(by='column_index')
        n_columns = lookup['column_index'].count()
        n_nan = len(lookup) - n_columns
        new_column_indices = range(n_columns).extend([np.nan]*n_nan)
        lookup['column_index'] = new_column_indices
        # clean up unindexed bodyids
        lookup = lookup.dropna(
            subset=['row_index', 'column_index'],
            how = 'all',
            ignore_index=True,
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

    def __restrict_row_indices(self, indices:list, allow_empty:bool=True):
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
        defined_rows = [
            i for i in indices
            if i in self.get_row_indices()
        ]
        if not allow_empty and len(defined_rows) != len(indices):
            raise ValueError("Some row indices not found in the lookup.")
        indices = defined_rows
        
        # restrict the lookup to the indices
        lookup = self.get_lookup()
        lookup = lookup[
            lookup['row_index'].isin(indices)
            ]
        self.lookup = lookup
        # restrict the matrix to the indices
        self.matrix = self.matrix[indices,:]
        self.__update_indexing()
        return
    
    def __restrict_column_indices(self, indices:list, allow_empty:bool=True):
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
            if i in self.get_column_indices()
        ]
        if not allow_empty and len(defined_columns) != len(indices):
            raise ValueError("Some column indices not found in the lookup.")
        indices = defined_columns
        
        # restrict the lookup to the indices
        lookup = self.get_lookup()
        lookup = lookup[
            lookup['column_index'].isin(indices)
            ]
        self.lookup = lookup
        # restrict the matrix to the indices
        self.matrix = self.matrix[:,indices]
        self.__update_indexing()
        return
    
    def __convert_bodyid_to_index(self, bodyid, allow_empty=True):
        """
        input: bodyid
        output: indices
        if allow_empty is False, raise an error if the bodyid is not found.
        """
        # format inputs
        if bodyid is None or bodyid == []:
            raise ValueError("bodyid is None or empty.")
        if isinstance(bodyid, int):
            bodyid = [bodyid]

        # find indices
        empty_matches = []
        row_indices, column_indices = [], []
        lookup = self.get_lookup()
        for id in bodyid:
            if not lookup["body_id"].isin([id]).any():
                empty_matches.append(id)
            else:
                row_indices.append(
                    lookup.loc[lookup["body_id"] == id].row_index.values[0]
                    )
                column_indices.append(
                    lookup.loc[lookup["body_id"] == id].column_index.values[0]
                    )
        row_indices = [i for i in row_indices if not pd.isna(i)]
        column_indices = [i for i in column_indices if not pd.isna(i)]
        # return results
        if not allow_empty and len(empty_matches) > 0:
            raise ValueError(f"bodyid(s) not found: {empty_matches}")
        else:
            if len(empty_matches) > 0:
                print(f"Warning: {len(empty_matches)} bodyid(s) not found.")
        return row_indices, column_indices

    def __convert_index_to_bodyid(self, index, axis='row'):
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
        bodyids : list[int]
            The body ids corresponding to the index or list of indices.
        """      
        if isinstance(index, int):
            index = [index]
        lookup = self.get_lookup()
        if axis == 'row':
            bodyids = [
                lookup.loc[lookup["row_index"] == id].body_id.values[0]
                for id in index
                ]
        elif axis == 'column':
            bodyids = [
                lookup.loc[lookup["column_index"] == id].body_id.values[0]
                for id in index
                ]
        else:
            raise ValueError('The axis must be either "row" or "column".')

        return bodyids

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
        
    def get_body_ids(self, sub_indices: list = None, axis: str = 'row') -> list:
        """
        Returns the body ids of the nodes of the adjacency matrix.

        Parameters
        ----------
        sub_indices : list, optional
            The indices of the nodes for which the body ids are returned.
            If None, the body ids of all nodes are returned. The default is None.

        Returns
        -------
        list
            The body ids of the nodes of the adjacency matrix.
        """
        if sub_indices is None:
            return self.lookup['body_id'].tolist()
        return self.__convert_index_to_bodyid(
            sub_indices,
            axis=axis
            )
    
    def get_row_indices(
            self,
            sub_bodyid: list = None,
            allow_empty: bool = True,
        ) -> list:
        """
        Returns the indices of the adjacency matrix references by the body_id list.

        Parameters
        ----------
        sub_bodyid : list, optional
            The body_ids of the nodes for which the indices are returned.
            If None, the indices of all nodes are returned. The default is None.
        allow_empty : bool, optional
            If False, raises an error if the body ids are not found in the lookup.
            The default is True.

        Returns
        -------
        list
            The row indices of the nodes of the adjacency matrix.
        """
        if sub_bodyid is None:
            return self.lookup['row_index'].tolist()
        rows, _ = self.__convert_bodyid_to_index( # already filters out the NaN values
            sub_bodyid,
            allow_empty=allow_empty,
            )
        if not allow_empty and len(rows) != len(sub_bodyid):
            raise ValueError("Some row body ids found only in the columns.")
        return rows

    def get_column_indices(
            self,
            sub_bodyid: list = None,
            allow_empty: bool = True,
        ) -> list:
        """
        Returns the indices of the adjacency matrix references by the body_id list.

        Parameters
        ----------
        sub_bodyid : list, optional
            The body_ids of the nodes for which the indices are returned.
            If None, the indices of all nodes are returned. The default is None.
        allow_empty : bool, optional
            If False, raises an error if the body ids are not found in the lookup.
            The default is True.

        Returns
        -------
        list
            The column indices of the nodes of the adjacency matrix.
        """
        if sub_bodyid is None:
            return self.lookup['column_index'].tolist()
        _, columns = self.__convert_bodyid_to_index(
            sub_bodyid,
            allow_empty=allow_empty,
            )
        if not allow_empty and len(columns) != len(sub_bodyid):
            raise ValueError("Some row body ids found only in the columns.")
        return columns

    # --- setters
    def restrict_from_to(
            self,
            row_ids:list = None,
            column_ids:list = None,
            allow_empty:bool=True
        ):
        '''
        Restricts the adjacency matrixand lookup table to a subset of bodyids,
        with different bodyids for rows and columns.

        Parameters
        ----------
        row_ids : list
            The list of bodyids to which the adjacency matrix rows are restricted.
            If None, the rows are not restricted.
        column_ids : list
            The list of bodyids to which the adjacency matrix columns are restricted.
            If None, the columns are not restricted.
        allow_empty : bool, optional
            If False, raises an error if the bodyids are not found in the lookup.
            The default is True.
        '''
        if row_ids is None and column_ids is None:
            return
        if row_ids is None:
            row_ids = self.get_row_indices()
        if column_ids is None:
            column_ids = self.get_column_indices()
        # convert the nodes to indices
        row_indices = self.get_row_indices(
            row_ids,
            allow_empty=allow_empty
            )
        column_indices = self.get_column_indices(
            column_ids,
            allow_empty=allow_empty
            )
        
        # restrict the data to the indices
        self.__restrict_row_indices(row_indices)
        self.__restrict_column_indices(column_indices)

        return
    
    def restrict_nodes(self, nodes:list, allow_empty:bool=True):
        '''
        Restricts the adjacency matrix to a subset of nodes (body_ids).

        Parameters
        ----------
        nodes : list
            The list of nodes to which the adjacency matrix is restricted.
        allow_empty : bool, optional
            If False, raises an error if the nodes are not found in the lookup.
            The default is True.
        '''
        # convert the nodes to indices
        r_i, c_i = self.get_indices(nodes)
        # restrict the matrix amd lookup to the indices
        self.restrict_from_to(r_i, c_i, allow_empty=allow_empty)
        return

    def restrict_rows(self, rows:list, allow_empty:bool=True):
        '''
        Restricts the adjacency matrix to a subset of rows (body_ids).

        Parameters
        ----------
        rows : list
            The list of rows to which the adjacency matrix is restricted.
        allow_empty : bool, optional
            If False, raises an error if the rows are not found in the lookup.
            The default is True.
        '''
        # convert the rows to indices
        r_i, _ = self.get_indices(rows)
        # restrict the matrix amd lookup to the indices
        self.__restrict_row_indices(r_i, allow_empty=allow_empty)
        return
    
    def restrict_columns(self, columns:list, allow_empty:bool=True):
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
        _, c_i = self.get_indices(columns)
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

    # --- visualisation
    def spy(self, title:str=None):
        '''
        Visualises the sparsity pattern of the adjacency matrix.

        Parameters
        ----------
        title : str, optional
            The title of the visualisation. The default is None.
        '''
        matrix_design.spy(self.get_matrix(), title=title) 
        title_ = os.path.join(params.PLOT_DIR, title + "_spy.pdf")
        plt.savefig(title_)
        return


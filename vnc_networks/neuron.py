'''
Initialises the Neuron class.
Meant to be used to load data for a single neuron.
Possible use cases are visualisation, or looking at synapse distributions.

Possible upgrade: possible to match synapses on the pre and post neurons
by using the dataset "Neuprint_Synapse_Connections_manc_v1.ftr" which
has two columns: ':START_ID(Syn-ID)' and ':END_ID(Syn-ID)'.
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np
from sklearn.cluster import KMeans


import params
from  get_nodes_data import load_data_neuron
import utils.plots_design as plot_design

NEURON_BASE_ATTRIBUTES = [
    'systematicType:string', 
    'hemilineage:string', 
    'somaSide:string', 
    'class:string', 
    'subclass:string', 
    'group:int', 
    'cellBodyFiber:string', 
    'size:long', 
    'target:string', 
    'predictedNtProb:float', 
    'predictedNt:string', 
    'tag:string',
]

class Neuron:
    def __init__(self, bodyId: int = None, from_file: str = None):
        """
        Initialise the neuron.
        Loading possible either from scratch using only the bodyId, or
        from a file. Loading from a file is useful if the neuron has already
        been processed and saved. Loading from scratch is useful if the neuron
        is new, but is computationally expensive and time-consuming.

        Parameters
        ----------
        bodyId : int, optional
            The body id of the neuron.
            The default is None.
        from_file : str, optional
            The name of the file to load the neuron from.
            The default is None.
        """
        if from_file is not None:
            self.__load(from_file)
        else:
            self.bodyId = bodyId
            self.data = load_data_neuron(bodyId, NEURON_BASE_ATTRIBUTES)
            self.__initialise_base_attributes()

    # private methods

    def __load(self, name: str):
        """
        Initialise the neuron from a file.
        """
        with open(os.path.join(params.NEURON_DIR, name+'.txt'), 'rb') as file:
            neuron = pickle.load(file)
        self.__dict__.update(neuron)

    def __initialise_base_attributes(self):
        self.type = self.data['systematicType:string'].values[0]
        self.hemilineage = self.data['hemilineage:string'].values[0]
        self.soma_side = self.data['somaSide:string'].values[0]
        self.class_ = self.data['class:string'].values[0]
        self.subclass = self.data['subclass:string'].values[0]
        self.group = self.data['group:int'].values[0]
        self.cell_body_fiber = self.data['cellBodyFiber:string'].values[0]
        self.size = self.data['size:long'].values[0]
        self.target = self.data['target:string'].values[0]
        self.predicted_nt_prob = self.data['predictedNtProb:float'].values[0]
        self.predicted_nt = self.data['predictedNt:string'].values[0]
        self.tag = self.data['tag:string'].values[0]

    def __load_synapse_ids(self):
        """
        Load the synapse ids for the neuron.
        """
        # neuron to synapse set
        neuron_to_synapse = pd.read_feather(
            params.NEUPRINT_NEURON_SYNAPSESSET_FILE
            )
        synset_list = neuron_to_synapse.loc[
            neuron_to_synapse[':START_ID(Body-ID)'] == self.bodyId
            ][':END_ID(SynSet-ID)'].values
         

        # synapse set to synapse
        synapses = pd.read_feather(
            params.NEUPRINT_SYNAPSSET_FILE
            )
        synapses = synapses.loc[
            synapses[':START_ID(SynSet-ID)'].isin(synset_list)
            ]
        synapses.reset_index(drop=True, inplace=True)
        
        # build a dataframe with columns 'syn_id', 'synset_id'
        synapse_df = pd.DataFrame(
            {
                'syn_id': synapses[':END_ID(Syn-ID)'],
                'synset_id': synapses[':START_ID(SynSet-ID)']
            }
        )
        synapse_df['start_id'] = synapse_df['synset_id'].apply(
            lambda x: int(x.split('_')[0])
            )  # body id of the presynaptic neuron
        synapse_df['end_id'] = synapse_df['synset_id'].apply(
            lambda x: int(x.split('_')[1])
            )  # body id of the postsynaptic neuron
        synapse_df['position'] = synapse_df['synset_id'].apply(
            lambda x: x.split('_')[2]
            )  # pre or post
        
        # remove the synapses that belong to partner neurons
        synapse_df = synapse_df[synapse_df['position'] == 'pre']
        synapse_df.drop(columns=['position'], inplace=True)

        # set the synapse ids
        self.synapse_df = synapse_df
        del synapses, neuron_to_synapse
        return

    def __load_synapse_locations(self):
        """
        Convert the synapse ids to MANC format.

        Parameters
        ----------
        subset : list, optional
            The subset of synapse ids to convert.
            The default is None, which converts all synapse ids.
        """

        # load the synapse data
        data = pd.read_feather(
            params.NEUPRINT_SYNAPSE_FILE, columns=[':ID(Syn-ID)','location:point{srid:9157}']
            )
        data = data.loc[data[':ID(Syn-ID)'].isin(self.synapse_df['syn_id'])]
              
        # merge with existing synapse df
        self.synapse_df = self.synapse_df.merge(
            data,
            right_on=':ID(Syn-ID)',
            left_on='syn_id',
            how='inner'
            )
        del data
        return
    
    def __categorical_neuropil_information(self):
        """
        Add a column 'neuropil' to the synapse df that contains the neuropil
        in which the synapse is located. This is done by finding the roi column 
        that has a bool value of True.
        """
        if 'neuropil' in self.synapse_df.columns:  # already done
            return
        
        roi_file = os.path.join(
            params.NEUPRINT_RAW_DIR,
            'all_ROIs.txt'
        )
        rois = list(pd.read_csv(roi_file, sep='\t').values.flatten())

        # find the subset of the synapses in the dataset that we care about for this neuron
        # then for each possible ROI, check which synapses are in that ROI
        # store each synapse's ROI in the neuropil column
        synapses_we_care_about = pd.read_feather(params.NEUPRINT_SYNAPSE_FILE, columns=[':ID(Syn-ID)'])[':ID(Syn-ID)'].isin(self.synapse_df['syn_id'])
        self.synapse_df['neuropil'] = 'None'
        for roi in rois:
            column_name = roi + ':boolean'
            roi_column = pd.read_feather(params.NEUPRINT_SYNAPSE_FILE, columns=[column_name])[synapses_we_care_about].iloc[:,0]
            synapses_in_roi = roi_column[roi_column == True].index
            self.synapse_df.loc[self.synapse_df['syn_id'].isin(synapses_in_roi), 'neuropil'] = roi
        
    def __explicit_synapse_positions(self):
        """
        convert the synapse locations deifned in thte text version of a dict
        to explicit position columns in the synapse df.
        """
        if 'X' in self.synapse_df.columns: # already done
            return
        
        locations = self.synapse_df['location:point{srid:9157}'].values
        X, Y, Z = [], [], []
        x,y,z = 0,1,2  # needed for dict parsing as the positions are samed as text
        for loc in locations:
            pos = eval(loc)  # read loc as a dict, use of x,y,z under the hood
            if not isinstance(pos, dict):
                pos = {0: np.nan, 1: np.nan, 2: np.nan}
            X.append(pos[0])
            Y.append(pos[1])
            Z.append(pos[2])
        self.synapse_df['X'] = X
        self.synapse_df['Y'] = Y
        self.synapse_df['Z'] = Z
        return

    # public methods
    # --- getters 
    def get_data(self):
        return self.data
    
    def get_synapse_table(self):
        '''
        Get the synapse table of the connections.
        '''
        synapses = self.synapse_df
        return synapses

    def get_synapse_distribution(self, threshold: bool = False):
        """
        Get the synapse distribution for the neuron.

        Parameters
        ----------
        threshold : bool, optional
            Whether to apply a threshold to the synapse distribution.
            The default is False.
        """
        # check if the synapse df is loaded
        try:
            self.synapse_df
        except AttributeError:
            print("Synapse data not loaded. Loading now...")
            self.__load_synapse_ids()
            self.__load_synapse_locations()
            print("... Synapse data loaded.")

        # define synapse positions
        self.__explicit_synapse_positions()
        # threshold if for a given neuron, the number of synapses is below the threshold
        if threshold:
            syn_count = self.synapse_df.groupby('end_id').size().reset_index()
            syn_count.columns = ['end_id', 'syn_count']
            syn_count = syn_count[
                syn_count['syn_count'] >= params.SYNAPSE_CUTOFF
                ]
            to_keep = syn_count['end_id'].values
            synapses = self.synapse_df[
                self.synapse_df['end_id'].isin(to_keep)
                ]
        else:
            synapses = self.synapse_df
        # get the synapse positions
        X = synapses['X'].values
        Y = synapses['Y'].values
        Z = synapses['Z'].values
        # save the positions in a convenient format in the neuron object
        return X, Y, Z
        
    def get_subdivisions(self):
        """
        Get the subdivisions of the synapses.
        """
        return self.subdivisions
    
    def get_body_id(self):
        """
        Get the body id of the neuron.
        """
        return self.bodyId

    def get_synapse_count(self, to=None):
        """
        Get the number of synapses for the neuron.

        Parameters
        ----------
        to : int, optional
            The body id of the postsynaptic neuron.
            If None, the total number of synapses is returned.
            The default is None.
        """
        if to is None:
            return len(self.synapse_df)
        else:
            return len(self.synapse_df[self.synapse_df['end_id'] == to])
        
    # --- setters
    def create_synapse_groups(self, attribute: str):
        """
        Create synapse groups based on an attribute.
        This will be used to split neurons in the Connections class.
        The table is filtered such that connections form a neuron
        to another are removed if the total number of synapses is below 
        the threshold in the params file.
        """
        if attribute == 'neuropil':
            self.__categorical_neuropil_information()
        if attribute not in self.synapse_df.columns:
            raise (f"Attribute {attribute} not in synapse dataframe.")
        # create a connections_df table involving the Neuron.
        # It has two columns 'id_pre' and 'id_post' with the bodyId of the pre and post neurons
        # and a columns 'subdivision_pre'  with the index associated to the attribute split on.
        # Unclassified synapses have a -1 index.
        # Finally, it has a 'synapse_ids' column with a list of the synapse ids.
        synapses = self.synapse_df[[
            attribute, 'syn_id', 'start_id', 'end_id'
            ]
        ]
        # find end_id for which the number of synapses is below the threshold
        syn_count = synapses.groupby('end_id').size().reset_index()
        syn_count.columns = ['end_id', 'syn_count']
        syn_count = syn_count[syn_count['syn_count'] < params.SYNAPSE_CUTOFF]
        to_discard = syn_count['end_id'].values
        synapses = synapses[~synapses['end_id'].isin(to_discard)]

        # complete the table
        synapses.fillna({attribute: -1}, inplace=True)  # unclassified synapses get together
        # define the subdivision a synapse belongs to by mapping the attribute to an index
        mapping = {
            val: i for i, val in enumerate(
                synapses[attribute].dropna().unique()
                )
            }
        mapping[np.nan] = -1
        synapses['subdivision_start'] = synapses[attribute].map(mapping)
        synapses['subdivision_start_name'] = synapses[attribute]
        synapses.drop(columns=[attribute], inplace=True)
        synapses = synapses.groupby(
            ['start_id', 'subdivision_start','subdivision_start_name', 'end_id']
            ).agg(list).reset_index()
        synapses['syn_count'] = synapses['syn_id'].apply(len)

        self.subdivisions = synapses

    def clear_not_connected(self, not_connected: list[int]):
        """
        Clear the subdivions table from neurons that have their bodyid in 
        the not_connected list.
        """
        self.subdivisions = self.subdivisions[
            ~self.subdivisions['end_id'].isin(not_connected)
        ]
        self.synapse_df = self.synapse_df[
            ~self.synapse_df['end_id'].isin(not_connected)
        ]
        
    # --- computations
    def cluster_synapses_spatially(
            self,
            n_clusters: int = 3,
            ):
        """
        Cluster the synapses spatially using K-Means clustering.
        """
        # get the synapse positions
        X, Y, Z = self.get_synapse_distribution()
        
        # cluster the synapses
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(np.array([X, Y, Z]).T)
        self.synapse_df['KMeans_cluster'] = kmeans.predict(
            np.array([
                self.synapse_df['X'],  # keep the order of the synapses
                self.synapse_df['Y'],
                self.synapse_df['Z']]
                ).T 
        )
        return
        
    # --- visualisation
    def plot_synapse_distribution(
            self,
            color_by: str = None,  
            discrete_coloring: bool = True,
            threshold: bool = False,
            cmap: str = params.blue_colorscale,
            ):
        """
        Plot the synapse distribution for the neuron.
        """
        X, Y, Z = self.get_synapse_distribution(threshold=threshold)
        fig, ax = plt.subplots(1, 1, figsize=params.FIGSIZE, dpi=params.DPI)
        if color_by is None:
            plot_design.scatter_xyz_2d(X, Y, Z=Z, ax=ax)
        else:
            if color_by not in self.synapse_df.columns:
                raise (f"Attribute {color_by} not in synapse dataframe.")
            # map self.synapse_df[color_by] categorical values to integers
            if threshold:
                syn_count = self.synapse_df.groupby('end_id').size().reset_index()
                syn_count.columns = ['end_id', 'syn_count']
                syn_count = syn_count[syn_count['syn_count'] >= params.SYNAPSE_CUTOFF]
                to_keep = syn_count['end_id'].values
                synapses = self.synapse_df[
                    self.synapse_df['end_id'].isin(to_keep)
                    ]
            else:
                synapses = self.synapse_df

            Z = synapses[color_by].dropna().values
            plot_design.scatter_xyz_2d(
                X,
                Y,
                Z=Z,
                z_label=color_by,
                ax=ax,
                cmap=cmap,
                discrete_coloring=discrete_coloring,
            )
        os.makedirs(params.PLOT_DIR, exist_ok=True)
        plt.savefig(
            f"{params.PLOT_DIR}/synapse_distribution_{self.bodyId}_{color_by}.pdf"
            )
        return ax
    
    # --- loading and saving
    def save(self, name: str):
        """
        Save the neuron to a file.

        Parameters
        ----------
        name : str
            The name of the file to save to.
        """
        os.makedirs(params.NEURON_DIR, exist_ok=True)
        with open(os.path.join(params.NEURON_DIR, name+'.txt'), 'wb') as file:
            pickle.dump(self.__dict__, file)


# --- helper functions

def split_neuron_by_neuropil(neuron_id):
    '''
    Define neuron subdivisions based on synapse distribution.
    Saves the subdivisions in a new neuron to file, which can be loaded by its
    name.
    '''
    name = 'neuron-'+str(neuron_id)+'_neuropil-split'
    # check there are files starting with name
    os.makedirs(params.NEURON_DIR, exist_ok=True)
    files = [f for f in os.listdir(params.NEURON_DIR) if f.startswith(name)]
    if files:
        return name
    else:
        neuron = Neuron(neuron_id)
        _ = neuron.get_synapse_distribution(threshold=True)
        neuron.create_synapse_groups(attribute='neuropil')
        neuron.save(name=name)
    return name
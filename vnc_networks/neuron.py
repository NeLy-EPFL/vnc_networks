'''
Initialises the Neuron class.
Meant to be used to load data for a single neuron.
Possible use cases are visualisation, or looking at synapse distributions.
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

class Neuron:
    def __init__(self, bodyId: int = None, from_file: str = None):
        if from_file is not None:
            self.__load(from_file)
        else:
            self.bodyId = bodyId
            self.data = load_data_neuron(bodyId)
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
        synset_list_pre = []
        synset_list_post = []
        for syn in synset_list:
            if syn[-3:] == 'pre':
                synset_list_pre.append(syn)
            elif syn[-4:] == 'post':
                synset_list_post.append(syn)
            else:
                continue
 
        # synapse set to synapse
        synapses = pd.read_feather(
            params.NEUPRINT_SYNAPSSET_FILE
            )
        synapses = synapses.loc[
            synapses[':START_ID(SynSet-ID)'].isin(synset_list)
            ]
        synapses.reset_index(drop=True, inplace=True)
        
        # build a datframe with columns 'syn_id', 'synset_id'
        synapse_df = pd.DataFrame(
            {
                'syn_id': synapses[':END_ID(Syn-ID)'],
                'synset_id': synapses[':START_ID(SynSet-ID)']
            }
        )
        synapse_df['start_id'] = synapse_df['synset_id'].apply(
            lambda x: x.split('_')[0]
            ) # body id of the presynaptic neuron
        synapse_df['end_id'] = synapse_df['synset_id'].apply(
            lambda x: x.split('_')[1]
            ) # body id of the postsynaptic neuron
        synapse_df['position'] = synapse_df['synset_id'].apply(
            lambda x: x.split('_')[2]
            ) # pre or post

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
            params.NEUPRINT_SYNAPSE_FILE
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

    def get_synapse_distribution(self, pre_or_post: str = None):
        """
        Get the synapse distribution for the neuron.


        Parameters
        ----------
        pre_or_post : str, optional
            Whether to get the pre or post synapse distribution.
            The default is None, which returns both.
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
        # get the synapse positions
        if not pre_or_post is None:
            synapses = self.synapse_df.loc[
                self.synapse_df['position'] == pre_or_post
                ]
        else:
            synapses = self.synapse_df
        X = synapses['X'].values
        Y = synapses['Y'].values
        Z = synapses['Z'].values
        # save the positions in a convenient format in the neuron object
        return X, Y, Z
        
    # --- setters
    def create_synapse_groups(self):
        pass # TODO: save the synapse groups to an attribute that can
        # be interfaced with the connections class

    # --- computations
    def cluster_synapses_spatially(
            self,
            n_clusters: int = 3,
            pre_or_post: str = None,
            ):
        """
        Cluster the synapses spatially using K-Means clustering.
        """
        # get the synapse positions
        X, Y, Z = self.get_synapse_distribution(pre_or_post=pre_or_post)
        
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
        if pre_or_post == 'pre':
            # set the cluster number to nan for the post synapses
            self.synapse_df.loc[
                self.synapse_df['position'] == 'post',
                'KMeans_cluster'
                ] = np.nan
        elif pre_or_post == 'post':
            # set the cluster number to nan for the pre synapses
            self.synapse_df.loc[
                self.synapse_df['position'] == 'pre',
                'KMeans_cluster'
                ] = np.nan
        return
        

    # --- visualisation
    def plot_synapse_distribution(
            self,
            pre_or_post: str = None,
            color_by: str = None,  
            ):
        """
        Plot the synapse distribution for the neuron.

        Parameters
        ----------
        pre_or_post : str, optional
            Whether to plot the pre or post synapse distribution.
            The default is None, which plots both.
        """
        X, Y, Z = self.get_synapse_distribution(pre_or_post)
        fig, ax = plt.subplots(1, 1, figsize=params.FIGSIZE, dpi=params.DPI)
        if color_by is None:
            plot_design.scatter_xyz_2d(X, Y, Z=Z, ax=ax)
        else:
            if color_by not in self.synapse_df.columns:
                raise (f"Attribute {color_by} not in synapse dataframe.")
            plot_design.scatter_xyz_2d(
                X,
                Y,
                Z=self.synapse_df[color_by].values,
                z_label=color_by,
                ax=ax,
                cmap=params.blue_colorscale,
            )
        plt.savefig(
            f"{params.PLOT_DIR}/synapse_distribution\
                _{self.bodyId}_{pre_or_post}_{color_by}.pdf"
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
        with open(os.path.join(params.NEURON_DIR,name+'.txt'), 'wb') as file:
            pickle.dump(self.__dict__, file)

    
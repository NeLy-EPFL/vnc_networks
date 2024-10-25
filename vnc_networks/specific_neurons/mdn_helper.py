'''
Functions specific to working with MDNs to avoid copying code.
'''
import os
import matplotlib.pyplot as plt

import neuron
from neuron import Neuron
from connections import Connections
import get_nodes_data
import params

FOLDER_NAME = 'MDN_specific'
FOLDER = os.path.join(params.FIG_DIR, FOLDER_NAME)
os.makedirs(FOLDER, exist_ok=True)

def get_mdn_bodyids():
    return get_nodes_data.get_neuron_bodyids({'type:string': 'MDN'})

def get_mdn_uids(data: Connections):
    return data.get_neuron_ids({'type:string': 'MDN'})

def get_subdivided_mdns(VNC, neuropil, side):
    '''
    Get the uids of MDNs split by neuropil and side.
    neuropil format and side format are flexible to account for different
    naming conventions accross the dataset.
    '''
    if neuropil in ['LegNp(T1)', 'T1', 'f', 'fl']:
        neuropil_ = 'LegNp(T1)'
    elif neuropil in ['LegNp(T2)', 'T2', 'm', 'ml']:
        neuropil_ = 'LegNp(T2)'
    elif neuropil in ['LegNp(T3)', 'T3', 'h', 'hl']:
        neuropil_ = 'LegNp(T3)'
    else:
        raise ValueError('Neuropil not recognized.')
    if side in ['L', 'Left', 'l', 'left', 'LHS']:
        side_ = 'L'
    elif side in ['R', 'Right', 'r', 'right', 'RHS']:
        side_ = 'R'

    mdns = VNC.get_neuron_ids({'type:string': 'MDN'})
    specific_mdns = [
            mdn for mdn in mdns if (
                (neuropil_ in VNC.get_node_label(mdn))
                & (side_ == VNC.get_node_label(mdn)[-2]) # names finishing with (L|R)
                ) # soma side not given for MDNs, but exists in the name
        ]
    return specific_mdns

def get_vnc_split_MDNs_by_neuropil(not_connected: list[int] = None):
    '''
    Get the VNC Connections object with MDNs split by neuropil.
    '''
    try:
        VNC = Connections(from_file='VNC_split_MDNs_by_neuropil')
        print('Loaded VNC Connections object with MDNs split by neuropil.')
    except:
        print('Creating VNC Connections object with MDNs split by neuropil...')
        MDNs = []
        for neuron_id in get_mdn_bodyids():
            neuron_name = neuron.split_neuron_by_neuropil(neuron_id)
            MDN = Neuron(from_file=neuron_name)
            MDNs.append(MDN)
        VNC = Connections()  # full VNC
        VNC.initialize(
            split_neurons=MDNs,
            not_connected=not_connected,
            )  # split MDNs according to the synapse data
        VNC.save(name='VNC_split_MDNs_by_neuropil')
    return VNC

def mdn_synapse_distribution(n_clusters: int = 3):
    '''
    show for each MDN the distribution of synapses in the neuropils.
    Each row is an MDN.
    The first figure is depth-colour coded.
    The second in neuropil-colored.
    The third is clustering-colored, specific to T3.

    Save the data for later use.
    '''
    mdn_bodyids = get_nodes_data.get_neuron_bodyids({'type:string': 'MDN'})
    for i in range(4): # each MDN
        MDN = Neuron(mdn_bodyids[i])
        _ = MDN.get_synapse_distribution()

        # Column 1: depth-color coded
        ax = MDN.plot_synapse_distribution(
            cmap=params.r_red_colorscale,
            savefig=False,
            ) # default is depth-color coded
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FOLDER,
                f'mdn_{mdn_bodyids[i]}_synapse_distribution_depth.pdf'
                )
            )
        plt.close()

        # Column 2: neuropil-colored
        MDN.add_neuropil_information()
        ax = MDN.plot_synapse_distribution(
            color_by='neuropil',
            discrete_coloring=True,
            savefig=False
            )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FOLDER,
                f'mdn_{mdn_bodyids[i]}_synapse_distribution_neuropil.pdf'
                )
            )
        plt.close()
        
        # Column 3: clustering-colored
        MDN.remove_defined_subdivisions()
        MDN.cluster_synapses_spatially(
            n_clusters=n_clusters,
            on_attribute = {'neuropil': 'T3'}
            )
        MDN.create_synapse_groups(attribute='KMeans_cluster')
        ax = MDN.plot_synapse_distribution(
            color_by='KMeans_cluster',
            discrete_coloring=True,
            savefig=False
            )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FOLDER,
                f'mdn_{mdn_bodyids[i]}_synapse_distribution_clustering.pdf'
                )
            )
        plt.close()
        MDN.save(name=f'mdn_{mdn_bodyids[i]}_T3_synapses_split')
        del MDN

def get_connectome_with_MDN_t3_branches(n_clusters: int = 3):
    '''
    Build the connectome with the T3 neuropil branches of MDN split.
    '''
    connectome_name = 'VNC_split_MDNs_by_T3_synapses'
    try:
        VNC = Connections(from_file=connectome_name)
        print('Loaded the connectome with T3 branches of MDN split.')
    except:
        print('Creating the connectome with T3 branches of MDN split...')
    
        # === creating the split neurons
        mdn_bodyids = get_nodes_data.get_neuron_bodyids({'type:string': 'MDN'})
        MDNs = []
        # if not already defined (tested on the first), define the split neurons
        filename = os.path.join(
            params.NEURON_DIR,
            f'mdn_{mdn_bodyids[0]}_T3_synapses_split.txt'
            )
        if not os.path.isfile(filename):
            mdn_synapse_distribution(n_clusters=n_clusters)
        # load
        for i in range(4):
            MDN = Neuron(from_file=f'mdn_{mdn_bodyids[i]}_T3_synapses_split')
            MDNs.append(MDN)
        # === create the connections
        VNC_full = Connections()
        VNC_full.initialize(
            split_neurons=MDNs, # split the MDNs according to the synapse data
            not_connected=mdn_bodyids # exclude connections from MDNs to MDNs
            )
        VNC = VNC_full.get_connections_with_only_traced_neurons() # remove neurons not fully traced
        VNC.save(name=connectome_name) # for later use
    return VNC
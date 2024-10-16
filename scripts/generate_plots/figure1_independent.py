"""
Figure 1

Show that the branches of MDN do recruit circuits that are somewhat independent
in the VNC. This is done by splitting the MDNs according to their synapse
locations.
"""
import os
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

import specific_neurons.mdn_helper as mdn_helper
from utils import plots_design
import params
from connections import Connections
from neuron import Neuron
from get_nodes_data import get_neuron_bodyids


FOLDER_NAME = 'Figure_1_independent'
FOLDER = os.path.join(params.FIG_DIR, FOLDER_NAME)
os.makedirs(FOLDER, exist_ok=True)


# -------------------------- Helper functions -------------------------- #
def write_neuron_dataframe(
        neurons: list[int],
        VNC: Connections,
        filename: str = 'test_df'
    ):
    '''
    Write infromation about the neurons in the list in a csv file.
    '''
    # get the boyids from the uids
    bodyids = VNC.get_bodyids_from_uids(neurons)
    # get neuron-specific information
    df = pd.DataFrame()
    for bid in tqdm(bodyids, desc='writing information about neurons to file'):
        neuron = Neuron(bid) # intialises information in subfields
        neuron_data = neuron.get_data() # df
        print(neuron_data)
        df = pd.concat([df, neuron_data], axis=0, ignore_index=True)
    df.to_csv(os.path.join(FOLDER, filename+'.csv'))


# -------------------------- Main functions -------------------------- #
def venn_mdn_branches_neuropil_direct(attribute: str = 'class:string'):
    '''
    Draw a Venn diagram of the neurons directly downstream of MDN split by where
    the synapses are.
    Draw a bar plot of the attribute of the neurons in the intersection of the
    3 groups, as well as the 3 groups separately.
    '''
    # Loading the connectivity data
    full_VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
        ) # exclude connections from MDNs to MDNs
    VNC = full_VNC.get_connections_with_only_traced_neurons() # exclude untraced neurons for statistics
    # Working with matrix representation, get n-th order connections
    cmatrix = VNC.get_cmatrix(type_='norm')
    
    # Get the uids of motor neurons split by MDN synapses in leg neuropils
    mdn_uids = VNC.get_neuron_ids({'type:string': 'MDN'})
    list_down_neurons = [] # will become a list with 3 sets
    for i in range(3):
        neuropil = 'LegNp(T'+str(i+1)+')' # synapses of MDNs in T1/T2/T3
        mdn_neuropil = [
            uid for uid in mdn_uids if neuropil in VNC.get_node_label(uid)
            ]
        down_partners = cmatrix.list_downstream_neurons(mdn_neuropil) # uids
        list_down_neurons.append(set(down_partners))

    # === I. Plot the Venn diagram
    _ = plots_design.venn_3(
        sets=list_down_neurons,
        set_labels=['MDNs|T1', 'MDNs|T2', 'MDNs|T3'],
        title='Neurons directly downstream of MDNs',
        )
    
    # saving
    plt.savefig(os.path.join(FOLDER, f'venn_mdn_branches_neuropil_direct.pdf'))
    plt.close()

    # === II. Plot the neuron 'attribute' distribution for the intersection
    full_intersection = list_down_neurons[0].intersection(
        list_down_neurons[1].intersection(list_down_neurons[2])
        )
    _, ax = plt.subplots(
        figsize=(params.FIG_WIDTH, params.FIG_HEIGHT),
        dpi=params.DPI,
        )
    VNC.draw_bar_plot(full_intersection, attribute, ax=ax)
    ax.set_title('Intersection of all 3 groups')
    plt.tight_layout()
    plt.savefig(
        os.path.join(FOLDER, 'venn_mdn_branches_neuropil_bar_intersection.pdf')
        )
    plt.close()

    # save the Connection data for the neurons in the intersection
    write_neuron_dataframe(
        full_intersection,
        VNC,
        filename='venn_mdn_branches_neuropil_intersection'
        )

    # === III. Plot the neuron 'attribute' distribution for the neurons
    # restricted to a single group
    _, axs = plt.subplots(
        1,
        3,
        figsize=(3 * params.FIG_WIDTH, params.FIG_HEIGHT),
        dpi=params.DPI,
        )
    for i in range(3):
        set_neurons = list_down_neurons[
            i
            ] - list_down_neurons[(i+1)%3] - list_down_neurons[(i+2)%3]
        VNC.draw_bar_plot(set_neurons, attribute, ax=axs[i])
        axs[i].set_title(f'unique downstream of MDNs|T{i+1}')
    plt.tight_layout()
    plt.savefig(
        os.path.join(FOLDER, 'venn_mdn_branches_neuropil_bar_exclusion.pdf')
        )
    plt.close()

def mdn_synapse_distribution():
    '''
    show for each MDN the distribution of synapses in the neuropils.
    Each row is an MDN.
    The first column is depth-colour coded.
    The second in neruopil-colored.
    The third is clustering-colored.
    '''
    mdn_bodyids = get_neuron_bodyids({'type:string': 'MDN'})
    for i in range(1): # each MDN
        fig, axs = plt.subplots(
            1, 
            3,
            figsize=(3*params.FIG_WIDTH, params.FIG_HEIGHT),
            dpi=params.DPI
            )

        MDN = Neuron(mdn_bodyids[i])
        _ = MDN.get_synapse_distribution()

        # Column 1: depth-color coded
        MDN.plot_synapse_distribution(
            cmap=params.r_red_colorscale,
            savefig=False,
            ax = axs[0]
            ) # default is depth-color coded

        # Column 2: neuropil-colored
        MDN.add_neuropil_information()
        MDN.plot_synapse_distribution(
            color_by='neuropil',
            discrete_coloring=True,
            ax=axs[1],
            savefig=False
            )
        
        # Column 3: clustering-colored
        MDN.remove_defined_subdivisions()
        MDN.cluster_synapses_spatially(n_clusters=8)
        MDN.create_synapse_groups(attribute='KMeans_cluster')
        MDN.plot_synapse_distribution(
            color_by='KMeans_cluster',
            discrete_coloring=True,
            ax=axs[2],
            savefig=False
            )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FOLDER,
                f'mdn_{mdn_bodyids[i]}_synapse_distribution.pdf'
                )
            )
        plt.close()
        del MDN, axs



def venn_t3_subbranches():
    '''
    Split the T3 part in subbranches and look at the isolated circuits.
    Add the previous figure the spatial visualisation of the synapses.
    '''
    # TODO
    # 1. Split the MDNs in spatially defined subbranches
    MDNs = []
    mdn_bodyids = get_neuron_bodyids({'type:string': 'MDN'})

    for i in range(4):
        MDN = Neuron(mdn_bodyids[i])
        _ = MDN.get_synapse_distribution(threshold=True)
        MDN.cluster_synapses_spatially(n_clusters=3)
        MDN.create_intersected_attribute(
            attributes = ['neuropil', 'KMeans_cluster']
        )
        #MDN.create_synapse_groups(attribute='KMeans_cluster')
        MDN.create_synapse_groups(attribute='neuropil_KMeans_cluster')
        MDN.plot_synapse_distribution(
            color_by='neuropil',
            discrete_coloring=True,
            threshold=True,
            cmap="Spectral")
        MDN.save(name='MDN_split-neuropil_'+str(i))  # if you want to save the neuron
        MDNs.append(MDN)
    # 2. Keep only the subranches that are in T3
    # 3. Get the downstream neurons of these subbranches

if __name__ == "__main__":
    #venn_mdn_branches_neuropil_direct()
    mdn_synapse_distribution()

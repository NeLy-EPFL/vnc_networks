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

FOLDER_NAME = 'Figure_1_independent'

# -------------------------- Helper functions -------------------------- #
def write_neuron_dataframe(
        neurons: list[int],
        VNC: Connections,
        filename: str = 'test_df'
    ):
    '''
    Write infromation about the neurons in the list in a csv file.
    '''
    folder = os.path.join(params.FIG_DIR, FOLDER_NAME)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # get the boyids from the uids
    bodyids = VNC.get_bodyids_from_uids(neurons)
    print(neurons)
    print(bodyids)
    # get neuron-specific information
    df = pd.DataFrame()
    for bid in tqdm(bodyids, desc='writing information about neurons to file'):
        neuron = Neuron(bid) # intialises information in subfields
        neuron_data = neuron.get_data() # df
        print(neuron_data)
        df = pd.concat([df, neuron_data], axis=0, ignore_index=True)
    df.to_csv(os.path.join(folder, filename+'.csv'))


# -------------------------- Main functions -------------------------- #
def venn_mdn_branches_neuropil_direct(attribute: str = 'class:string'):
    '''
    Draw a Venn diagram of the neurons directly downstream of MDN split by where
    the synapses are.
    Draw a bar plot of the attribute of the neurons in the intersection of the
    3 groups, as well as the 3 groups separately.
    '''
    # Loading the connectivity data
    VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
        ) # exclude connections from MDNs to MDNs

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
    folder = os.path.join(params.FIG_DIR, FOLDER_NAME)
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, f'venn_mdn_branches_neuropil_direct.pdf'))
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
        os.path.join(folder, 'venn_mdn_branches_neuropil_bar_intersection.pdf')
        )
    plt.close()

    # save the Connection data for the neurons in the intersection
    write_neuron_dataframe(
        full_intersection,
        VNC,
        filename='venn_mdn_branches_neuropil_bar_intersection'
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
        os.path.join(folder, 'venn_mdn_branches_neuropil_bar_exclusion.pdf')
        )
    plt.close()

    # print the neurons down of T3 to verify that proofreading is okish
    write_neuron_dataframe(
        list_down_neurons[2],
        VNC,
        filename='venn_mdn_branches_neuropil_t3'
        )

def venn_t3_subbranches():
    '''
    Split the T3 part in subbranches and look at the isolated circuits.
    Add the previous figure the spatial visualisation of the synapses.
    '''
    # TODO
    pass

if __name__ == "__main__":
    venn_mdn_branches_neuropil_direct()

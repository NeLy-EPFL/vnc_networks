'''
Fig 1a: Venn diagram of motor neurons downstream of MDN split by where the 
motor neurons are located (neuropil T1-T3). Connection within 2 hops.

Fig 1b: Venn diagram of motor neurons downstream of MDN split by where MDN
synapses are (neuropil T1-T3). Connection within 2 hops.
'''
import os
import matplotlib.pyplot as plt

from connections import Connections
from neuron import Neuron
import params
from get_nodes_data import get_neuron_bodyids
from utils import plots_design

# ----- Helper functions -----
def split_neuron_by_neuropil(neuron_id):
    '''
    Define neuron subdivisions based on synapse distribution.
    Saves the subdivisions in a new neuron to file, which can be loaded by its
    name.
    '''
    name = 'neuron-'+str(neuron_id)+'_neuropil-split'
    # check there are files starting with name
    files = [f for f in os.listdir(params.NEURON_DIR) if f.startswith(name)]
    if files:
        return name
    else:
        neuron = Neuron(neuron_id)
        _ = neuron.get_synapse_distribution(threshold=True)
        neuron.create_synapse_groups(attribute='neuropil')
        neuron.save(name=name)
    return name

# ----- Figure functions -----
def fig1a():
    pass

def fig1b(n_hops: int = 2):
    """
    Venn diagram of motor neurons downstream of MDN split by where MDN
    synapses are (neuropil T1-T3). Connection within n hops.
    """
    # Loading the connectivity data
    try:
        VNC = Connections(from_file='VNC_split_MDNs_by_neuropil')
    except:
        MDNs = []
        for neuron_id in get_neuron_bodyids({'type:string': 'MDN'}):
            neuron_name = split_neuron_by_neuropil(neuron_id)
            MDN = Neuron(from_file=neuron_name)
            MDNs.append(MDN)
        VNC = Connections()  # full VNC
        VNC.initialize(split_neurons=MDNs)  # split MDNs according to the synapse data
        VNC.save(name='VNC_split_MDNs_by_neuropil')

    # Working with matrix representation, get n-th order connections
    cmatrix = VNC.get_cmatrix(type_='norm')
    cmatrix.power_n(n_hops)

    # Get the postsynaptics neurons: leg motor neurons
    leg_motor_neurons = []
    for i in range(3):
        neuropil = 'T'+str(i+1)
        selection_dict = {
            'somaNeuromere:string': neuropil,
            'class:string': 'motor neuron'
            }
        neurons_post = VNC.get_neuron_ids(selection_dict) # uids
        leg_motor_neurons.extend(neurons_post)
    leg_motor_neurons = set(leg_motor_neurons)
    print('Number of leg motor neurons:', len(leg_motor_neurons))

    # Get the uids of motor neurons split by MDN synapses in leg neuropils
    mdn_uids = VNC.get_neuron_ids({'type:string': 'MDN'})
    list_down_mns = [] # will become a list with 3 lists
    for i in range(3):
        neuropil = 'LegNp(T'+str(i+1)+')'
        mdn_neuropil = [
            uid for uid in mdn_uids if neuropil in VNC.get_node_label(uid)
            ]
        down_partners = cmatrix.list_downstream_neurons(mdn_neuropil) # uids
        mns_downstream = set(down_partners).intersection(leg_motor_neurons)
        list_down_mns.append(mns_downstream)

    # Plot the Venn diagram
    ax = plots_design.venn_3(
        sets=list_down_mns,
        set_labels=['MDNs|T1', 'MDNs|T2', 'MDNs|T3'],
        title='Motor neurons downstream of MDN split by synapse locations',
        )
    
    # saving
    folder = os.path.join(params.FIG_DIR, 'Fig1')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, f'Fig1b_{n_hops}_hops.pdf'))
    
    



if __name__ == '__main__':
    #fig1a()
    fig1b(n_hops=2)
    fig1b(n_hops=3)
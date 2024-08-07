'''
For all figures, connections between MDNs are deleted.

Fig 1a: Venn diagram of motor neurons downstream of MDN split by where the 
motor neurons are located (neuropil T1-T3). Connection within 2 hops.

Fig 1b: Venn diagram of motor neurons downstream of MDN split by where MDN
synapses are (neuropil T1-T3). Connection within 2 hops.

Fig 1c: Fig 1b split by leg neuropils.

Fig 1d: Venn diagram of neurons directly downstream of MDNs, split by where
the MDN synapses are (neuropil T1-T3).
'''
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from connections import Connections
from neuron import Neuron
import params
import get_nodes_data
from utils import plots_design

# ----- Helper functions -----
def get_mdn_bodyids():
    return get_nodes_data.get_neuron_bodyids({'type:string': 'MDN'})

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

def get_vnc_split_MDNs_by_neuropil(not_connected: list[int] = None):
    try:
        VNC = Connections(from_file='VNC_split_MDNs_by_neuropil')
    except:
        MDNs = []
        for neuron_id in get_mdn_bodyids():
            neuron_name = split_neuron_by_neuropil(neuron_id)
            MDN = Neuron(from_file=neuron_name)
            MDNs.append(MDN)
        VNC = Connections()  # full VNC
        VNC.initialize(
            split_neurons=MDNs,
            not_connected=not_connected,
            )  # split MDNs according to the synapse data
        VNC.save(name='VNC_split_MDNs_by_neuropil')
    return VNC

def get_full_vnc(not_connected: list[int] = None):
    try:
        VNC = Connections(from_file='full_VNC')
    except:
        VNC = Connections()
        VNC.initialize(not_connected=not_connected)
        VNC.save(name='full_VNC')
    return VNC

def get_leg_motor_neurons(data: Connections, leg: str = None):
    '''
    Get the uids of leg motor neurons.
    f: front leg
    m: middle leg
    h: hind leg
    '''
    match leg:
        case 'f':
            target = ['fl']
        case 'm':
            target = ['ml']
        case 'h':
            target = ['hl']
        case None:
            target = ['fl', 'ml', 'hl']
    leg_motor_neurons = []
    for t in target:
        selection_dict = {
            'subclass:string': t,
            'class:string': 'motor neuron'
            }
        neurons_post = data.get_neuron_ids(selection_dict) # uids
        leg_motor_neurons.extend(neurons_post)
    return set(leg_motor_neurons)
            

# ----- Figure functions -----
def fig1a(n_hops: int = 2):
    """
    Venn diagram of motor neurons downstream of MDN split by where the motor
    neurons are located (neuropil T1-T3). Connection within 2 hops.
    nb: Obvious that the circles are disjoint, but safety check and gives the
    number of neurons in each group.
    """
    # Loading the connectivity data
    VNC = get_full_vnc(not_connected=get_mdn_bodyids())

    # Working with matrix representation, get n-th order connections
    cmatrix = VNC.get_cmatrix(type_='norm')
    cmatrix.power_n(n_hops)

    # Get the postsynaptic neurons: leg motor neurons
    list_down_mns = []
    mdn_uids = VNC.get_neuron_ids({'type:string': 'MDN'})
    down_partners = cmatrix.list_downstream_neurons(mdn_uids) # uids
    for leg_ in tqdm(['f', 'm', 'h']):
        leg_motor_neurons = get_leg_motor_neurons(VNC, leg=leg_)
        mns_downstream = set(down_partners).intersection(leg_motor_neurons)
        list_down_mns.append(mns_downstream)

    # Plot the Venn diagram
    _ = plots_design.venn_3(
        sets=list_down_mns,
        set_labels=['T1 MNs', 'T2 MNs', 'T3 MNs'],
        title='Motor neurons downstream of MDNs split by neuropil',
        )
    
    # saving
    folder = os.path.join(params.FIG_DIR, 'Fig1')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, f'Fig1a_{n_hops}_hops.pdf'))

def fig1b(n_hops: int = 2):
    """
    Venn diagram of motor neurons downstream of MDN split by where MDN
    synapses are (neuropil T1-T3). Connection within n hops.
    """
    # Loading the connectivity data
    VNC = get_vnc_split_MDNs_by_neuropil(not_connected=get_mdn_bodyids())

    # Working with matrix representation, get n-th order connections
    cmatrix = VNC.get_cmatrix(type_='norm')
    cmatrix.power_n(n_hops)

    # Get the postsynaptics neurons: leg motor neurons
    leg_motor_neurons = get_leg_motor_neurons(VNC) # uids
    # Get the uids of motor neurons split by MDN synapses in leg neuropils
    mdn_uids = VNC.get_neuron_ids({'type:string': 'MDN'})
    list_down_mns = [] # will become a list with 3 lists
    for i in tqdm(range(3)):
        neuropil = 'LegNp(T'+str(i+1)+')' # synapses of MDNs in T1/T2/T3
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

def fig1c(n_hops: int = 2):
    # Loading the connectivity data
    VNC = get_vnc_split_MDNs_by_neuropil(not_connected=get_mdn_bodyids())

    # Working with matrix representation, get n-th order connections
    cmatrix = VNC.get_cmatrix(type_='norm')
    cmatrix.power_n(n_hops)

    # create the plotting ax object:
    _, axs = plt.subplots(
        1,
        3,
        figsize=(3 * params.FIG_WIDTH, params.FIG_HEIGHT),
        dpi=params.DPI,
        )
    
    # MDN uids
    mdn_uids = VNC.get_neuron_ids({'type:string': 'MDN'})

    # Split by neuropil for motor neurons
    for i, leg_ in tqdm(enumerate(['f', 'm', 'h'])):
        leg_motor_neurons = get_leg_motor_neurons(VNC, leg=leg_)

        # Get the uids of motor neurons split by MDN synapses in leg neuropils
        list_down_mns = [] # will become a list with 3 lists
        for j in range(3):
            neuropil = 'LegNp(T'+str(j+1)+')' # synapses of MDNs in T1/T2/T3
            mdn_neuropil = [
                uid for uid in mdn_uids if neuropil in VNC.get_node_label(uid)
                ]
            down_partners = cmatrix.list_downstream_neurons(mdn_neuropil) # uids
            mns_downstream = set(down_partners).intersection(leg_motor_neurons)
            list_down_mns.append(mns_downstream)

        # Plot the Venn diagram
        _ = plots_design.venn_3(
            ax=axs[i],
            sets=list_down_mns,
            set_labels=['MDNs|T1', 'MDNs|T2', 'MDNs|T3'],
            title=f'LegNp(T{i+1})',
            )
        
    # saving
    folder = os.path.join(params.FIG_DIR, 'Fig1')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, f'Fig1c_{n_hops}_hops.pdf'))

def fig1d():
    # Loading the connectivity data
    VNC = get_vnc_split_MDNs_by_neuropil(not_connected=get_mdn_bodyids())

    # Working with matrix representation, get n-th order connections
    cmatrix = VNC.get_cmatrix(type_='norm')
    
    # Get the uids of motor neurons split by MDN synapses in leg neuropils
    mdn_uids = VNC.get_neuron_ids({'type:string': 'MDN'})
    list_down_neurons = [] # will become a list with 3 sets
    for i in tqdm(range(3)):
        neuropil = 'LegNp(T'+str(i+1)+')' # synapses of MDNs in T1/T2/T3
        mdn_neuropil = [
            uid for uid in mdn_uids if neuropil in VNC.get_node_label(uid)
            ]
        down_partners = cmatrix.list_downstream_neurons(mdn_neuropil) # uids
        list_down_neurons.append(set(down_partners))

    # Plot the Venn diagram
    _ = plots_design.venn_3(
        sets=list_down_neurons,
        set_labels=['MDNs|T1', 'MDNs|T2', 'MDNs|T3'],
        title='Neurons directly downstream of MDNs',
        )
    
    # saving
    folder = os.path.join(params.FIG_DIR, 'Fig1')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, f'Fig1d.pdf'))

if __name__ == '__main__':
    fig1a(n_hops=2)
    fig1a(n_hops=3)
    fig1b(n_hops=2)
    fig1b(n_hops=3)
    fig1c()
    fig1d()
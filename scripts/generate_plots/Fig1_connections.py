'''
Plots for the connections between MDNs and motor neurons.

For all figures, connections between MDNs are deleted.

Fig 1a: Venn diagram of motor neurons downstream of MDN split by where the 
motor neurons are located (neuropil T1-T3). Connection within 2 hops.

Fig 1b: Venn diagram of motor neurons downstream of MDN split by where MDN
synapses are (neuropil T1-T3). Connection within 2 hops.

Fig 1c: Fig 1b split by leg neuropils.

Fig 1d: Venn diagram of neurons directly downstream of MDNs, split by where
the MDN synapses are (neuropil T1-T3).

Fig 1e: Same as Fig 1d, but for each area of the Venn diagram, make a barplot
of the type of neurons, given by the attribute chosen (default 'class:string').

Fig 1f: Graph generated by the neurons contacting MDNs|Ti -> motor neurons in
Ti within 2 hops, for each i in [1,2,3]. Left and right are plotted separately.

Fig 1g: Same as Fig 1f, but the graph is concentric with MDNs at the center,
intermediate neurons in the middle, and motor neurons at the periphery. The
motor neurons are grouped in subclusters according to the attribute chosen.

Fig 1h: Zooming in on a single muscle, show the graph of the neurons contacting
MDNs -> motor neurons within n hops.
'''
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from connections import Connections
import params
from utils import plots_design
import specific_neurons.mdn_helper as mdn_helper
import specific_neurons.all_neurons_helper as all_neurons_helper
import specific_neurons.motor_neurons_helper as mns_helper

# ----- Helper functions -----            
def draw_bar_plot(
    data: Connections,
    neurons: set[int],
    attribute: str = 'class:string',
    ax=None
    ):
    '''
    Draw a bar plot of the attribute of the neurons in the set.
    '''
    # Get the attribute values
    values = []
    for uid in neurons:
        values.append(data.get_node_attribute(uid, attribute))
    values.sort()
    values = pd.Series(values)
    counts = values.value_counts()
    counts.plot(kind='bar', ax=ax, colormap='grey')
    ax.set_xlabel(attribute)
    ax.set_ylabel('# neurons')
    ax = plots_design.make_nice_spines(ax)
    return ax

# ----- Figure functions -----
def fig1a(n_hops: int = 2):
    """
    Venn diagram of motor neurons downstream of MDN split by where the motor
    neurons are located (neuropil T1-T3). Connection within 2 hops.
    nb: Obvious that the circles are disjoint, but safety check and gives the
    number of neurons in each group.
    """
    print('> Fig 1a')
    # Loading the connectivity data
    VNC = all_neurons_helper.get_full_vnc(not_connected=mdn_helper.get_mdn_bodyids())

    # Working with matrix representation, get n-th order connections
    cmatrix = VNC.get_cmatrix(type_='norm')
    cmatrix.power_n(n_hops)

    # Get the postsynaptic neurons: leg motor neurons
    list_down_mns = []
    mdn_uids = VNC.get_neuron_ids({'type:string': 'MDN'})
    down_partners = cmatrix.list_downstream_neurons(mdn_uids) # uids
    for leg_ in tqdm(['f', 'm', 'h']):
        leg_motor_neurons = mns_helper.get_leg_motor_neurons(VNC, leg=leg_)
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
    plt.close()

def fig1b(n_hops: int = 2):
    """
    Venn diagram of motor neurons downstream of MDN split by where MDN
    synapses are (neuropil T1-T3). Connection within n hops.
    """
    print('> Fig 1b')

    # Loading the connectivity data
    VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
        )

    # Working with matrix representation, get n-th order connections
    cmatrix = VNC.get_cmatrix(type_='norm')
    cmatrix.power_n(n_hops)

    # Get the postsynaptics neurons: leg motor neurons
    leg_motor_neurons = mns_helper.get_leg_motor_neurons(VNC) # uids
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
    plt.close()

def fig1c(n_hops: int = 2):
    print('> Fig 1c')

    # Loading the connectivity data
    VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
        )

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
        leg_motor_neurons = mns_helper.get_leg_motor_neurons(VNC, leg=leg_)

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
    plt.close()

def fig1d():
    print('> Fig 1d')
    
    # Loading the connectivity data
    VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
        )

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
    plt.close()

def fig1e(attribute: str = 'class:string'):
    print('> Fig 1e')
    
    # Loading the connectivity data
    VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
        )
    VNC.include_node_attributes([attribute])
    VNC.save(name='VNC_split_MDNs_by_neuropil')

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

    # Plotting
    folder = os.path.join(params.FIG_DIR, 'Fig1')
    # Fig 1e - a) Unique neurons in each group
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
        draw_bar_plot(VNC, set_neurons, attribute, ax=axs[i])
        axs[i].set_title(f'unique downstream of MDNs|T{i+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'Fig1e_{attribute}.pdf'))
    plt.close()

    # Fig 1e - b) Overlapping neurons 1-1
    _, axs = plt.subplots(
        1,
        3,
        figsize=(3 * params.FIG_WIDTH, params.FIG_HEIGHT),
        dpi=params.DPI,
        )
    for i in range(3):
        set_neurons = list_down_neurons[i].intersection(
            list_down_neurons[(i+1)%3]
            )
        draw_bar_plot(VNC, set_neurons, attribute, ax=axs[i])
        axs[i].set_title(
            f'overlap downstream of MDNs|T{i+1} & MDNs|T{(i+1)%3 + 1}'
            )
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'Fig1e_{attribute}_overlap_1-1.pdf'))
    plt.close()

    # Fig 1e - c) Intersection of all 3 groups
    set_neurons = list_down_neurons[0].intersection(
        list_down_neurons[1].intersection(list_down_neurons[2])
        )
    _, ax = plt.subplots(
        figsize=(params.FIG_WIDTH, params.FIG_HEIGHT),
        dpi=params.DPI,
        )
    draw_bar_plot(VNC, set_neurons, attribute, ax=ax)
    ax.set_title('Intersection of all 3 groups')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'Fig1e_{attribute}_intersection.pdf'))
    plt.close()

def fig1f(syn_thresh: int = None, label_nodes: bool = False):
    print('> Fig 1f')

    # Loading the connectivity data
    VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
        )
    mdns = VNC.get_neuron_ids({'type:string': 'MDN'})

    # figure template
    _, axs = plt.subplots(
        3,
        2,
        figsize=(2 * params.FIG_WIDTH, 3 * params.FIG_HEIGHT),
        dpi=params.DPI,
        )

    # Analyse each neuropil separately
    for j, side in enumerate(['L' ,'R']):
        for i, leg in enumerate(['f', 'm', 'h']):
            neuropil = 'LegNp(T'+str(i+1)+')' # synapses of MDNs in T1/T2/T3
            source_neurons = [
                mdn for mdn in mdns if (
                    (neuropil in VNC.get_node_label(mdn))
                    & (side in VNC.get_node_label(mdn))
                    ) # soma side not given for MDNs, but exists in the name
                ]
            leg_motor_neurons = mns_helper.get_leg_motor_neurons(VNC,leg=leg) # uids
            target_neurons = VNC.get_neuron_ids({
                'somaSide:string': side + 'HS',
                })
            target_neurons = list(
                set(target_neurons).intersection(leg_motor_neurons)
            )
            l2_graph = VNC.paths_length_n(2, source_neurons, target_neurons)
            subconnections = VNC.subgraph(
                l2_graph.nodes,
                edges=l2_graph.edges(), # only the edges directly involved in the paths
                )  # new Connections object
            _ = subconnections.display_graph(
                title=f'MDNs|{side}_{neuropil}_to_MNs_2_hops',
                ax=axs[i,j],
                method='spring',
                save=False,
                syn_threshold=syn_thresh,
                label_nodes=label_nodes,
                )
            del subconnections, l2_graph
    plt.tight_layout()
    folder = os.path.join(params.FIG_DIR, 'Fig1')
    if not os.path.exists(folder):
        os.makedirs(folder)
    title = f'Fig1f_syn-threshold={syn_thresh}'
    if label_nodes:
        title += '_labeled-nodes'
    plt.savefig(os.path.join(folder, title+'.pdf'))
    plt.close()
            
def fig1g(
    attribute: str = 'class:string',
    syn_thresh: int = None,
    ):

    print('> Fig 1g')

    # Loading the connectivity data
    VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
        )

    # figure template
    _, axs = plt.subplots(
        3,
        2,
        figsize=(2 * params.FIG_WIDTH, 3 * params.FIG_HEIGHT),
        dpi=params.DPI,
        )

    # Analyse each neuropil separately
    for j, side in enumerate(['L' ,'R']):
        for i, leg in enumerate(['f', 'm', 'h']):
            neuropil = 'LegNp(T'+str(i+1)+')' # synapses of MDNs in T1/T2/T3
            source_neurons = mdn_helper.get_subdivided_mdns(VNC, neuropil, side)
            leg_motor_neurons = mns_helper.get_leg_motor_neurons(VNC,leg=leg) # uids
            target_neurons = VNC.get_neuron_ids({
                'somaSide:string': side + 'HS',
                })
            target_neurons = list( # get the motor neurons for a single leg
                set(target_neurons).intersection(leg_motor_neurons)
            )
            l2_graph = VNC.paths_length_n(2, source_neurons, target_neurons)
            subconnections = VNC.subgraph(
                l2_graph.nodes,
                edges=l2_graph.edges(), # only the edges directly involved in the paths
                )  # new Connections object
            _ = subconnections.draw_graph_concentric_by_attribute(
                title=f'MDNs|{side}_{neuropil}_to_MNs_2_hops',
                ax=axs[i,j],
                attribute='target:string',
                center_nodes=source_neurons,
                target_nodes=target_neurons,
                save=False,
                syn_threshold=syn_thresh,
                )
            del subconnections, l2_graph
    plt.tight_layout()
    folder = os.path.join(params.FIG_DIR, 'Fig1')
    if not os.path.exists(folder):
        os.makedirs(folder)
    title = f'Fig1g_attribute={attribute}_syn-threshold={syn_thresh}'
    plt.savefig(os.path.join(folder, title+'.pdf'))
    plt.close()       

def fig1h(
        target: dict,
        n_hops: int = 2,
        label_nodes: bool = False,
        method: str = None,
    ):
    '''
    Zooming in on a single muscle, show the graph of the neurons contacting
    MDNs -> motor neurons within n hops.

    target: dictionary with the target neuron attributes
    necessary keys: 'class:string', 'somaSide:string', 'subclass:string',
    'target:string'
    '''
    print('> Fig 1h')

    # Loading the connectivity data
    VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
        )
    side = target['somaSide:string']
    neuropil = target['subclass:string']
    mdns = mdn_helper.get_subdivided_mdns(VNC, neuropil, side)
    target_neurons = VNC.get_neuron_ids(target)

    # Get the graph made of neurons contacting MDNs -> motor neurons within n hops
    graph = VNC.paths_length_n(n_hops, mdns, target_neurons)

    # Make 2 graphs: on the left only the connections directly involved in the
    # paths, on the right the graph generate by the neurons in the paths
    _, axs = plt.subplots(
        1,
        2,
        figsize=(2 * params.FIG_WIDTH, params.FIG_HEIGHT),
        dpi=params.DPI,
        )
    
    # Left
    subconnections = VNC.subgraph(
        graph.nodes,
        edges=graph.edges(), # only the edges directly involved in the paths
        )  # new Connections object
    if method is None:
        _, pos = subconnections.draw_graph_in_out_center_circle(
                    title='direct',
                    ax=axs[0],
                    input_nodes=mdns,
                    output_nodes=target_neurons,
                    save=False,
                    label_nodes=label_nodes,
                    return_pos=True,
                    )
    else:
        _, pos = subconnections.display_graph(
            title='direct',
            ax=axs[0],
            method=method,
            save=False,
            label_nodes=label_nodes,
            return_pos=True
            )
    del subconnections

    # Right
    subconnections = VNC.subgraph(
        graph.nodes,
        )  # new Connections object 
    _ = subconnections.display_graph(
                title='complete',
                ax=axs[1],
                pos=pos, #reuse the positions of the left graph
                save=False,
                label_nodes=label_nodes,
                )
    del subconnections

    plt.tight_layout()

    # Saving
    title = f'MDNs|{side}_{neuropil}_to_{target["target:string"]}_{n_hops}_hops'
    title = title.replace('/', '|')
    title = title.replace(' ', '-')
    title = 'Fig1h_' + title
    if method is not None:
        title += f'_method={method}'
    folder = os.path.join(params.FIG_DIR, 'Fig1')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if label_nodes:
        title += '_labeled-nodes'
    plt.savefig(os.path.join(folder, title+'.pdf'))
    plt.close()


if __name__ == '__main__':
    #fig1a(n_hops=2)
    #fig1a(n_hops=3)
    #fig1b(n_hops=2)
    #fig1b(n_hops=3)
    #fig1c()
    #fig1d()
    #fig1e(attribute='class:string')
    #fig1e(attribute='subclass:string')
    #fig1e(attribute='somaNeuromere:string')
    #fig1e(attribute='target:string')
    #fig1e(attribute='somaSide:string')
    #fig1f()
    #fig1f(syn_thresh=40, label_nodes=True)
    #fig1g(attribute='target:string')
    #fig1g(attribute='target:string',syn_thresh=40)
    # Example target:
    target = { # right hind leg
        'class:string': 'motor neuron',
        'somaSide:string': 'RHS',
        'subclass:string': 'hl',
        'target:string': 'Tr flexor'
        }
    fig1h(target, n_hops=2, label_nodes=True)
    pass
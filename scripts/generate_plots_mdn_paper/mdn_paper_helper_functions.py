'''
Helper functions for generating plots for the MDN paper.
'''

import matplotlib.pyplot as plt

from specific_neurons import mdn_helper
import params


def graph_from_mdn_to_muscle(
        target: dict,
        n_hops: int = 2,
        label_nodes: bool = False,
        method: str = None,
        interneuron_conditions: dict = None,
    ):
    '''
    Zooming in on a single muscle, show the graph of the neurons contacting
    MDNs -> motor neurons within n hops.

    Parameters
    ----------
    target: dict
        dictionary with the target neuron attributes
        necessary keys: 'class:string', 'somaSide:string', 'subclass:string',
        'target:string'
    n_hops: int
        number of hops to consider
    label_nodes: bool
        whether to label the nodes
    method: str
        method to use for the graph layout
    interneuron_conditions: dict
        conditions to filter the interneurons

    Returns
    -------
    title: str
        title of the plot
    axs: list of matplotlib axes
        axes of the plot
    '''
    # Loading the connectivity data
    full_VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
        )
    VNC = full_VNC.get_connections_with_only_traced_neurons()
    side = target['somaSide:string']
    neuropil = target['subclass:string']
    mdns = mdn_helper.get_subdivided_mdns(VNC, neuropil, side)
    target_neurons = VNC.get_neuron_ids(target)

    # Get the graph made of neurons contacting MDNs -> motor neurons within n hops
    graph = VNC.paths_length_n(n_hops, mdns, target_neurons)
    if interneuron_conditions is None:
        nodes = graph.nodes
    else:
        interneurons = set(graph.nodes) - set(mdns) - (target_neurons)
        interneurons = interneurons.intersection(
            VNC.get_neuron_ids(interneuron_conditions)
            )
        nodes = list(mdns.union(target_neurons).union(interneurons))

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
        nodes,
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
        nodes,
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
    if method is not None:
        title += f'_method={method}'
    if label_nodes:
        title += '_labeled-nodes'
    return title, axs
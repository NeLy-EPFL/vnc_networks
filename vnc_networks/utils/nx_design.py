'''
Helper functions for making networkx graphs look nice and standardized.
'''
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter
from matplotlib.lines import Line2D

import params
from utils.graph_processing_utils import (
    remove_excitatory_connections,
    remove_inhbitory_connections 
)   
from utils.math_utils import sigmoid

#--- Constants ---#
MAX_EDGE_WIDTH = 5
FIGSIZE = (8, 8)
DPI = 300
NODE_SIZE = 100
NT_TYPES = { 
    "gaba": {"color": params.INHIB_COLOR, "linestyle": "-"},  # ":"
    "acetylcholine": {"color": params.EXCIT_COLOR, "linestyle": "-"},  # "--"
    "glutamate": {"color": params.GLUT_COLOR, "linestyle": "-"},
    "unknown": {"color": params.LIGHTGREY, "linestyle": "-"},
    None: {"color": params.LIGHTGREY, "linestyle": "-"},
}
FONT_SIZE = 5
FONT_COLOR = "black"

#--- CODE ---#

def draw_graph(
    G: nx.Graph,
    pos: dict = None,
    ax: plt.Axes = None,
    node_size: int = NODE_SIZE,
    return_pos: bool = False,
) -> dict:
    """Plots the network using the network specs and
    returns the positions of the nodes.

    Parameters
    ----------
    G : nx.Graph
        Networkx graph object.
    ax : plt.Axes, optional
        Matplotlib axes object, by default None
    pos : dict, optional
        Dictionary containing the positions of the nodes, by default None
    node_size : int, optional
        Size of the nodes, by default as defined in the constants

    Returns
    -------
    pos: Dict, optional
        Dictionary containing node ID as keys and positions of nodes
        on the figure as values.
    ax: plt.Axes
    """

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if pos is None:
        pos = nx.circular_layout(G)

    # Normalize the edge width of the network
    weights = list(nx.get_edge_attributes(G, "eff_weight").values())
    normalized_weights = (weights / np.abs(weights).max()) * MAX_EDGE_WIDTH

    # Color nodes and edges
    node_colors = params.LIGHTGREY # to update
    edge_type = list(nx.get_edge_attributes(G, "predictedNt:string").values())
    edge_colors = [NT_TYPES[nt_name]["color"] for nt_name in edge_type]

    # Plot graph
    nx.draw(
        G,
        pos,
        with_labels=False,
        labels="",
        width=normalized_weights,
        alpha=0.75,
        node_size=node_size,
        node_color=node_colors,
        edge_color=edge_colors,
        connectionstyle="arc3,rad=0.1",
        font_size=FONT_SIZE,
        font_color=FONT_COLOR,
        ax=ax,
    )

    # Add legend
    ax = add_edge_legend(
            ax,
            normalized_weights,
            edge_colors,
            MAX_EDGE_WIDTH/np.abs(weights).max(),
            )

    # return
    if return_pos:
        return ax, pos
    return ax

def add_edge_legend(
    ax: plt.Axes,
    weights: list,
    color_list: list,
    arrow_norm: float,
):
    """
    Add a legend to the plot with the edge weights.
    """
    color = Counter(color_list).most_common(1)[0][0]
    lines = []
    edges_weight_list = sorted(
        [np.abs(w) for w in weights if w != 0], reverse=True
    )
    for _, width in enumerate(edges_weight_list):
        lines.append(Line2D([], [], linewidth=width, color=color))
    # keep only 3 legend entries, evenly spaced, including the first and last
    if len(edges_weight_list) > 3:
        edges_weight_list = [
            edges_weight_list[0],
            edges_weight_list[len(edges_weight_list) // 2],
            edges_weight_list[-1],
        ]
        lines = [lines[0], lines[len(lines) // 2], lines[-1]]

    edges_weight_list = [
        f"{int(weight/arrow_norm)}" for weight in edges_weight_list
    ]
    ax.legend(
        lines,
        edges_weight_list,
        bbox_to_anchor=(0.1, 0.25),
        frameon=False,
    )
    return ax

def get_ring_pos_sorted_alphabetically(
    node_list: list,
    graph: nx.DiGraph,
    radius: float = 3,
    center: tuple = (0,0),
    ):
    """
    Define the positions of the nodes in a circular layout, sorted alphabetically.
    """
    if center is None:
        center = (0,0)
    subgraph = graph.subgraph(node_list)
    raw_pos = nx.circular_layout(subgraph, scale=radius, center=center)
    ordered_nodes = sorted(
        subgraph.nodes,
        key=lambda x: subgraph.nodes[x]["node_label"],
    )
    pos = {node_: pos_ for node_,pos_ in zip(ordered_nodes,list(raw_pos.values()))}
    return pos

def draw_graph_concentric_circles(
    graph: nx.DiGraph,
    ax: plt.Axes = None,
    edge_norm: float = None,
    pos=None,
    center=None,
    radius_scaling=1,
    output='graph'
):
    """
    Draw a graph with the nodes and edges attributes based on the
    nx.draw_networkx function and the properties of the graph itself.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to draw
    ax : plt.Axes, optional
        The axes to plot on, by default None
    edge_norm : float, optional
        The normalisation factor for the edge width, by default None
    pos : dict, optional
        The positions of the nodes, by default None
    center : tuple, optional
        The center of the graph, by default None
    output : str, optional
        Whether to return the figure 'graph' or the positions 'pos', by default 'graph'
    """
    

    # prune network: if a an edge has weight 0, remove it
    graph = graph.copy()
    edges_to_remove = []
    for u, v, data in graph.edges(data=True):
        if data["eff_weight"] == 0:
            edges_to_remove.append((u, v))
    graph.remove_edges_from(edges_to_remove)

    # outer ring: nodes that don't output to other nodes
    outer_ring = [
        n for n in graph.nodes if len(list(graph.successors(n))) == 0
    ]
    pos_outer = get_ring_pos_sorted_alphabetically(outer_ring, graph, radius=radius_scaling, center=center)

    # inner ring: nodes that only output to other nodes
    input_only_nodes = [
        n for n in graph.nodes if len(list(graph.predecessors(n))) == 0 and len(list(graph.successors(n))) > 0
        ]
    pos_inner = get_ring_pos_sorted_alphabetically(input_only_nodes, graph, radius=radius_scaling*0.5, center=center)

    # central ring: nodes that both input and output to other nodes
    central_ring = graph.nodes - set(outer_ring) - set(input_only_nodes)
    pos_central = get_ring_pos_sorted_alphabetically(central_ring, graph, radius=radius_scaling*0.75, center=center)

    # merge the positions
    pos = {**pos_outer, **pos_inner, **pos_central}

    # define the colors of the nodes
    node_colors = [
        graph.nodes[n]["node_color"]
        if "node_color" in graph.nodes[n].keys()
        else params.LIGHTGREY
        for n in graph.nodes
    ]

    node_labels = {
        n: graph.nodes[n]["node_label"]
        if "node_label" in graph.nodes[n].keys()
        else ""
        for n in graph.nodes
    }

    # define the colors and widths of the edges based on the weights
    if edge_norm is None:
        try:
            edge_norm = max([np.abs(graph.edges[e]["eff_weight"]) for e in graph.edges]) / 5
        except ValueError:
            edge_norm = 1
    widths = [np.abs(graph.edges[e]["eff_weight"]) / edge_norm for e in graph.edges]
    edge_type = list(nx.get_edge_attributes(graph, "predictedNt:string").values())
    edge_colors = [NT_TYPES[nt_name]["color"] for nt_name in edge_type]

    if output == 'graph':
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8), dpi=120)
        # Plot graph
        nx.draw(
            graph,
            pos,
            nodelist=graph.nodes,
            with_labels=True,
            labels=node_labels,
            alpha=0.5,
            node_size=100,
            node_color=node_colors,
            edge_color=edge_colors,
            width=widths,
            connectionstyle="arc3,rad=0.1",
            font_size=5,
            font_color="black",
            ax=ax,
        )
        if len(widths) > 0:
            add_edge_legend(
                ax,
                normalized_weights=widths,
                color_list=edge_colors,
                arrow_norm=1 / edge_norm,
            )
        return ax
    elif output == 'pos':
        return pos

def draw_graph_grouped_by_attribute(
        graph: nx.DiGraph,
        attribute: str,
        filename: str = 'network',
        restricted_connections: str = None,
        position_reference: str = None,
        center_instance: str = None,
        ):
    """
    The plot resembles a large flower, where each petal is a cluster of neurons.
    Each petal is coloured according to the type of the neurons it contains.
    Each petal is made of three concentric circles, where the neurons are placed.
    The neurons are placed according to their connectiivty within the cluster.
    The inner circle contains only neurons that output to other neurons within the cluster
    without receiving any input from them.
    The middle circle contains neurons that receive inputs and outputs from other neurons within the cluster. 
    The outer circle is the rest (only inputs or no connections within the cluster
    in case we restrict to a subset).
    In the middle of the flower, there is a circle containing the neurons for which the attribute is not defined.
    Alternatively, the center can be occupied by a given predefined instance of the attribute.
    The positions can be defined with regards to a specific type of connection
    (inhibitory or excitatory) or all connections. 

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to plot.
    attribute : str
        The attribute to group the neurons by.
    filename : str
        The name of the file to save the plot.
    restricted_connections : str, optional
        The type of connections to consider, by default None
    position_reference : str, optional
        The type of connections to consider for the positions, by default None
    center_instance : str, optional 
        The attribute to place at the center, by default None

    Raises
    ------
    ValueError
        If the requested attribute is not defined in the graph.

    Returns
    -------
    plt.Axes
    """
    # verify that the attribute is defined for all nodes.
    # if not, add a default value 'unknown'
    for node in graph.nodes():
        if not attribute in graph.nodes[node].keys():
            graph.nodes[node][attribute] = 'None'

    # count the number of clusters defined
    cluster_list = [graph.nodes[node][attribute] for node in graph.nodes()]
    clusters = list(set(cluster_list))

    if clusters == []:
        raise ValueError(f'No clustering defined by {attribute} in the graph')
    
    if (not center_instance is None) and (center_instance in clusters):
        clusters.remove(center_instance) # will be placed in the center afterwards
    nb_clusters = len(clusters)


    cluster_centers = [
        ((0.85*FIGSIZE[0])*np.cos(2*np.pi*cluster_nb/nb_clusters),
        (0.85*FIGSIZE[1])*np.sin(2*np.pi*cluster_nb/nb_clusters))
        #((1+1.2*nb_clusters)*np.cos(2*np.pi*cluster_nb/nb_clusters),
        #(1+1.2*nb_clusters)*np.sin(2*np.pi*cluster_nb/nb_clusters))
        for cluster_nb in range(nb_clusters)
        ]

    positions = {}
    leaf_scaling = 0.5 / nb_clusters * FIGSIZE[0]

    for cluster_idx, cluster_nb in enumerate(clusters):
        # get a subset of the graph known_graph, where the 'cluster' attribute is equal to cluster_of_interest
        subgraph = graph.subgraph(
            [
                node
                for node in graph.nodes()
                if graph.nodes[node][attribute] == cluster_nb
            ]
            )
        # use the network with specific connectivity to define the positions of the neurons
        if position_reference == 'inhibitory': 
            subgraph = remove_excitatory_connections(subgraph)
        elif position_reference == 'excitatory':
            subgraph = remove_inhbitory_connections(subgraph)

        # draw the graph
        pos = draw_graph_concentric_circles(
            subgraph,
            center=cluster_centers[cluster_idx],
            output='pos',
            radius_scaling=leaf_scaling,
            )
        positions = {**positions, **pos}

    # additional nodes in the center of the flower
    additional_nodes = [
        node for node in graph.nodes()
        if graph.nodes[node][attribute] == center_instance
        ]
    subgraph = graph.subgraph(additional_nodes)
    if position_reference == 'inhibitory': 
        subgraph = remove_excitatory_connections(subgraph)
    elif position_reference == 'excitatory':
        subgraph = remove_inhbitory_connections(subgraph)
    pos = draw_graph_concentric_circles(
        subgraph,
        center=(0,0),
        output='pos',
        radius_scaling=0.3 * FIGSIZE[0] * sigmoid(nb_clusters),
        )
    positions = {**positions, **pos}


    # draw the graph
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if restricted_connections == 'inhibitory': 
        graph = remove_excitatory_connections(graph)
    elif restricted_connections == 'excitatory':
        graph = remove_inhbitory_connections(graph)

    edge_norm = max([np.abs(graph.edges[e]["eff_weight"]) for e in graph.edges]) / 5
    widths = [np.abs(graph.edges[e]["eff_weight"]) / edge_norm for e in graph.edges]
    edge_type = list(nx.get_edge_attributes(graph, "predictedNt:string").values())
    edges_colors = [NT_TYPES[nt_name]["color"] for nt_name in edge_type]
    node_labels = {
            n: graph.nodes[n]["node_label"]
            if "node_label" in graph.nodes[n].keys()
            else ""
            for n in graph.nodes
    }
    node_colors = [
            graph.nodes[n]["node_color"]
            if "node_color" in graph.nodes[n].keys()
            else "grey"
            for n in graph.nodes
        ]
    nx.draw(
        graph,
        pos=positions,
        nodelist=graph.nodes,
        with_labels=True,
        labels=node_labels,
        alpha=0.5,
        node_size=NODE_SIZE,
        node_color=node_colors,
        edge_color=edges_colors,
        width=widths,
        connectionstyle="arc3,rad=0.1",
        font_size=2,
        font_color="black",
        ax=ax,
    )
    add_edge_legend(ax,
                    weights=widths,
                    color_list=edges_colors,
                    arrow_norm=1 / edge_norm,)
    return ax
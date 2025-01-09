"""
Figure 5: T3 Oscillations

Show how oscillations and coordinated movements are generated in T3.
"""

import os

import matplotlib.pyplot as plt
import mdn_paper_helper_functions as paper_funcs
import networkx as nx
import numpy as np
import pandas as pd
import params
import specific_neurons.mdn_helper as mdn_helper
import specific_neurons.motor_neurons_helper as mns_helper
import utils.matrix_design as matrix_design
import utils.nx_design as nx_design
import utils.nx_utils as nx_utils

FOLDER_NAME = "Figure_5_t3_oscillations"
FOLDER = os.path.join(params.FIG_DIR, FOLDER_NAME)
os.makedirs(FOLDER, exist_ok=True)


# -------------------------- Helper functions -------------------------- #
def get_length_n_t3_connections(n_hops: int = 2):
    # Loading the connectivity data
    full_VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
    )
    VNC = full_VNC.get_connections_with_only_traced_neurons()
    mdns = mdn_helper.get_subdivided_mdns(VNC, neuropil="hl", side="RHS")
    target_neurons = mns_helper.get_leg_motor_neurons(VNC, leg="h", side="RHS")

    l2_graph = VNC.paths_length_n(n_hops, mdns, target_neurons)
    # get the connections object with nodes involved in paths of length 2
    l2_connections = VNC.subgraph(
        nodes=l2_graph.nodes(),
    )
    graph = l2_connections.get_graph()
    edges = graph.edges()
    # get the edges that are not from target to target
    edges = [edge for edge in edges if not all([uid in target_neurons for uid in edge])]
    subconnections = l2_connections.subgraph(
        edges=edges
    )  # all nodes involved in the path, new Connections object

    return subconnections


def read_clusters_from_file(cluster_file: str):
    """
    Read the clusters from a file and return them as a list of lists of int.
    """
    clusters = []
    with open(cluster_file, "r") as f:
        cluster = []
        for line in f:
            if line == "\n":
                clusters.append(cluster)
                cluster = []
            else:
                cluster.append(int(line.strip()))
    return clusters


# -------------------------- Main functions -------------------------- #


def show_length2_t3(
    attribute: str = "class:string",
    syn_thresh: int = None,
):
    side = "RHS"
    neuropil = "hl"
    # Loading the connectivity data
    subconnections = get_length_n_t3_connections(n_hops=2)
    mdns = mdn_helper.get_subdivided_mdns(
        subconnections,
        neuropil=neuropil,
        side=side,
    )
    target_neurons = mns_helper.get_leg_motor_neurons(
        subconnections, leg="h", side=side
    )

    # draw the graph
    ax = subconnections.draw_graph_concentric_by_attribute(
        title=f"MDNs|{side}_{neuropil}_to_MNs_2_hops",
        attribute="target:string",
        center_nodes=mdns,
        target_nodes=target_neurons,
        save=False,
        syn_threshold=syn_thresh,
    )

    plt.tight_layout()
    title = f"mdn_to_mns_T3_attribute={attribute}_syn-threshold={syn_thresh}"
    title = title.replace("/", "|")
    title = title.replace(" ", "-")
    plt.savefig(os.path.join(FOLDER, title + ".pdf"))
    plt.close()


def hind_leg_muscles_graph(
    muscle_: str,
    side_: str = "RHS",
    n_hops: int = 2,
    label_nodes: bool = True,
):
    """
    Show the graph of the neurons contacting MDNs -> motor neurons within n hops
    for the front leg muscles.
    """
    target = {
        "class:string": "motor neuron",
        "somaSide:string": side_,
        "subclass:string": "hl",
        "target:string": muscle_,
    }
    title, axs = paper_funcs.graph_from_mdn_to_muscle(
        target,
        n_hops=n_hops,
        label_nodes=label_nodes,
    )
    plt.savefig(os.path.join(FOLDER, title + ".pdf"))
    plt.close()


def cluster_t3_graph(n_hops: int = 2):
    """
    Show the graph of the neurons contacting MDNs -> motor neurons within n hops
    for the hind leg muscles.
    """
    subconnections = get_length_n_t3_connections(n_hops=n_hops)
    mdns = mdn_helper.get_subdivided_mdns(subconnections, neuropil="hl", side="RHS")
    target_neurons = mns_helper.get_leg_motor_neurons(
        subconnections, leg="h", side="RHS"
    )

    # Cluster based on connectivity similarity
    t3_cmatrix = subconnections.get_cmatrix(type_="unnorm")
    t3_sim, uid_clusters, index_clusters = t3_cmatrix.detect_clusters(
        distance="cosine",
        method="hierarchical",
        cutoff=0.5,
        cluster_size_cutoff=2,
        cluster_data_type="uid",
    )

    # draw a matrix where all entries are white except for the boundaries
    # between clusters as defined by the number of elements in each list in
    # the clusters list.
    clustered_mat = t3_sim.get_matrix()
    mat = np.zeros((clustered_mat.shape[0], clustered_mat.shape[1]))
    for cluster in index_clusters:
        mat[cluster[0] : cluster[-1] + 1, cluster[0] : cluster[-1] + 1] = 1
    ax, _ = t3_sim.imshow(savefig=False, title="Clustered T3 matrix")
    ax.imshow(mat, cmap="binary", alpha=0.5)
    plt.savefig(os.path.join(FOLDER, "clustered_t3_matrix.pdf"))
    plt.close()

    # Plot the different graphs
    fig, axs = plt.subplots(
        1,
        len(uid_clusters),
        figsize=(len(uid_clusters) * params.FIGSIZE[0], params.FIGSIZE[1]),
    )
    for i, cluster in enumerate(uid_clusters):
        nodes = cluster + list(mdns) + list(target_neurons)
        try:
            graph = subconnections.subgraph(nodes)
            _ = graph.draw_graph_concentric_by_attribute(
                title=f"Cluster {i+1}",
                ax=axs[i],
                center_nodes=mdns,
                target_nodes=target_neurons,
                save=False,
                label_nodes=True,
                attribute="target:string",
            )
        except:
            pass
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "clusters_as_graph.pdf"))
    plt.close()

    # Save the clusters as csv file
    cluster_file = os.path.join(FOLDER, "clusters.csv")
    with open(cluster_file, "w") as f:
        for i, cluster in enumerate(uid_clusters):
            for uid in cluster:
                f.write(f"{uid}\n")
            f.write("\n")
    return cluster_file


def display_l2_t3_graph_with_clusters(cluster_file: str):
    """
    Represent the graph of the neurons contacting MDNs -> motor neurons within
    2 hops. Nodes are colored based on the cluster they belong to.
    """
    subconnections = get_length_n_t3_connections(n_hops=2)
    # read the clusters from the file
    uid_clusters = read_clusters_from_file(cluster_file)
    mdns = mdn_helper.get_subdivided_mdns(subconnections, neuropil="hl", side="RHS")
    target_neurons = mns_helper.get_leg_motor_neurons(
        subconnections, leg="h", side="RHS"
    )

    # get the graph
    graph = subconnections.get_graph()
    _, pos = subconnections.draw_graph_concentric_by_attribute(
        center_nodes=mdns,
        target_nodes=target_neurons,
        save=False,
        return_pos=True,
        attribute="target:string",
    )  # get the positions of the nodes according to their nature

    # create a color map for the clusters
    colors = plt.cm.tab20.colors
    colors = [colors[i] for i in range(len(uid_clusters))]

    # add the 'color' attribute to the nodes as the cluster they belong to
    for i, cluster in enumerate(uid_clusters):
        for node in cluster:
            graph.nodes[node]["color"] = colors[i]
    # give -1 to the nodes that are not in a cluster
    remaining_nodes = set(graph.nodes()) - set(
        [node for cluster in uid_clusters for node in cluster]
    )
    for node in remaining_nodes:
        graph.nodes[node]["color"] = params.DARKGREY

    # draw the graph
    ax = nx_design.draw_graph(
        graph,
        pos,
        label_nodes=True,
    )
    plt.savefig(os.path.join(FOLDER, "l2_t3_graph_with_clusters.pdf"))
    plt.close()


def display_l2_t3_to_motor_neuron_clusters(cluster_file):
    """
    Explore potential motor synergies through motor neurons that are clustered
    together based on their connectivity.
    """
    clusters = read_clusters_from_file(cluster_file)
    subconnections = get_length_n_t3_connections(n_hops=2)
    motor_neurons = mns_helper.get_leg_motor_neurons(
        subconnections, leg="h", side="RHS"
    )
    mdns = mdn_helper.get_subdivided_mdns(subconnections, neuropil="hl", side="RHS")

    # restrict the clusters to the motor neurons
    motor_clusters = []
    for cluster in clusters:
        motor_cluster = [uid for uid in cluster if uid in motor_neurons]
        if motor_cluster and len(motor_cluster) > 3:
            motor_clusters.append(motor_cluster)

    # draw the graphs with direct connections only
    fig, axs = plt.subplots(
        1,
        len(motor_clusters),
        figsize=(len(motor_clusters) * params.FIGSIZE[0], params.FIGSIZE[1]),
    )
    positions = []

    for i, cluster in enumerate(motor_clusters):
        graph = subconnections.paths_length_n(2, mdns, cluster)
        specific_connections = subconnections.subgraph(
            nodes=graph.nodes(),
            edges=graph.edges(),  # only the edges directly involved in the paths
        )
        _, pos = specific_connections.draw_graph_in_out_center_circle(
            title=f"Motor cluster {i+1}",
            ax=axs[i],
            input_nodes=mdns,
            output_nodes=cluster,
            save=False,
            label_nodes=True,
            return_pos=True,
        )
        positions.append(pos)
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "motor_clusters.pdf"))
    plt.close()

    # draw the complete graphs
    fig, axs = plt.subplots(
        1,
        len(motor_clusters),
        figsize=(len(motor_clusters) * params.FIGSIZE[0], params.FIGSIZE[1]),
    )
    for i, cluster in enumerate(motor_clusters):
        graph = subconnections.paths_length_n(2, mdns, cluster)
        specific_connections = subconnections.subgraph(
            nodes=graph.nodes(),
        )
        _ = specific_connections.display_graph(
            title=f"Motor cluster {i+1}",
            ax=axs[i],
            save=False,
            label_nodes=True,
            pos=positions[i],
        )
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "motor_clusters_complete.pdf"))
    plt.close()

    # save the motor clusters to a file
    motor_cluster_file = os.path.join(FOLDER, "motor_clusters.csv")
    _ = subconnections.include_node_attributes(["target:string"])
    df = subconnections.list_neuron_properties(
        neurons=motor_neurons,
        input_type="uid",
    )

    # add a 'cluster' column to the dataframe based on bodyid indexing
    df["cluster"] = -1
    print(df)
    for i, cluster in enumerate(motor_clusters):
        for c in cluster:
            df.loc[df["uid"] == c, "cluster"] = i
    df = df.sort_values(by=["cluster"])
    df.to_csv(motor_cluster_file, index=False)
    return motor_cluster_file


def intersection_upstream_motor_neuron_pools(
    motor_neuron_file,
    syn_threshold: int = None,
):
    """
    Plot the overlap of the upstream motor neuron pools for the different
    clusters of motor neurons as a matrix.
    """
    cluster_data = pd.read_csv(motor_neuron_file)
    cluster_ids = list(set(cluster_data["cluster"]) - {-1})
    nb_clusters = len(cluster_ids)

    # load connectivity data
    subconnections = get_length_n_t3_connections(n_hops=2)
    motor_neurons = mns_helper.get_leg_motor_neurons(
        subconnections, leg="h", side="RHS"
    )
    mdns = mdn_helper.get_subdivided_mdns(subconnections, neuropil="hl", side="RHS")

    # draw matrix to show how much the interneurons upstream overlap
    neurons_up = {}
    for i, c in enumerate(cluster_ids):
        cluster = list(cluster_data[cluster_data["cluster"] == c]["uid"])
        graph = subconnections.paths_length_n(
            n=2,
            source=mdns,
            target=cluster,
            syn_threshold=syn_threshold,
        )
        interneurons = list(graph.nodes() - set(motor_neurons) - set(mdns))
        neurons_up[i] = interneurons

    nb_interneurons = np.zeros((nb_clusters, nb_clusters))
    for i in range(nb_clusters):
        for j in range(nb_clusters):
            nb_interneurons[i, j] = len(set(neurons_up[i]) & set(neurons_up[j]))
    ax = matrix_design.imshow(
        nb_interneurons,
        title="Overlap of upstream interneurons",
        xlabel="Cluster",
        ylabel="Cluster",
        save=False,
        vmin=0,
        cmap=params.grey_heatmap,
    )
    title = "motor_neuron_clusters_upstream_overlap"
    if syn_threshold:
        title += f"_syn-threshold={syn_threshold}"
    plt.savefig(os.path.join(FOLDER, title + ".pdf"))
    plt.close()


def t3_nttype_interneurons(
    nt_type: str,
    n_hops: int = 2,
    min_degree: int = 1,
):
    """
    Plot the MDN to motor neurons graph for the T3 neurons with the interneurons
    restricted to a specific neurotransmitter type.

    Parameters
    ----------
    nt_type: str
        neurotransmitter type to filter the interneurons. Can be 'acetylcholine',
        'gaba', 'glutamate'.
    n_hops: int
        number of hops to consider
    min_degree: int
        minimum out degree of the interneurons
    """
    # Data loading
    subconnections = get_length_n_t3_connections(n_hops=n_hops)
    motor_neurons = mns_helper.get_leg_motor_neurons(
        subconnections, leg="h", side="RHS"
    )
    mdns = mdn_helper.get_subdivided_mdns(subconnections, neuropil="hl", side="RHS")
    # Data filtering
    graph = subconnections.paths_length_n(n_hops, mdns, motor_neurons)
    interneurons = set(graph.nodes) - set(mdns) - set(motor_neurons)
    interneurons = interneurons.intersection(
        subconnections.get_neuron_ids({"predictedNt:string": nt_type})
    )
    interneurons = [
        interneuron
        for interneuron in interneurons
        if graph.out_degree(interneuron) >= min_degree
    ]
    nodes = list(set(mdns).union(set(motor_neurons)).union(interneurons))
    specific_connections = subconnections.subgraph(
        nodes=nodes,
        # edges = graph.edges(), # only the edges directly involved in the paths
    )
    # Plot the graph
    _ = specific_connections.draw_graph_in_out_center_circle(
        title=f"l2 T3 with {nt_type} interneurons",
        input_nodes=mdns,
        output_nodes=motor_neurons,
        save=False,
        label_nodes=True,
        return_pos=True,
    )
    plt.tight_layout()
    title = f"l{n_hops}_t3_nttype={nt_type}_min_degree={min_degree}"
    plt.savefig(os.path.join(FOLDER, title + ".pdf"))
    plt.close()


def focus_strongest_inhibitors_t3(
    min_degree: int = 5,
):
    """
    WARNING: this looks like an unintelligible mess, shaped as a star.

    Focus on the strongest inhibitors in the T3 MDN-MNs neurons.
    """
    # Identify the core inhibitors
    l2_subconnections = get_length_n_t3_connections(n_hops=2)
    motor_neurons = mns_helper.get_leg_motor_neurons(
        l2_subconnections, leg="h", side="RHS"
    )
    mdns = mdn_helper.get_subdivided_mdns(l2_subconnections, neuropil="hl", side="RHS")
    graph = l2_subconnections.paths_length_n(2, mdns, motor_neurons)
    interneurons = set(graph.nodes) - set(mdns) - set(motor_neurons)
    inhibitors = interneurons.intersection(
        l2_subconnections.get_neuron_ids({"predictedNt:string": "gaba"})
    )
    inhibitors = [
        inhibitor
        for inhibitor in inhibitors
        if graph.out_degree(inhibitor) >= min_degree
    ]
    # Place back in the larger graph
    full_vnc = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
    )
    vnc = full_vnc.get_connections_with_only_traced_neurons()
    interconnected_graph = vnc.paths_length_n(
        n=2,
        source=inhibitors,
        target=inhibitors,
        syn_threshold=None,
    )
    specific_connections = vnc.subgraph(
        nodes=interconnected_graph.nodes(),
        edges=interconnected_graph.edges(),
    )
    # Plot the graph
    _ = specific_connections.draw_graph_concentric_by_attribute(
        title=f"T3 with strongest inhibitors",
        save=False,
        label_nodes=True,
        center_nodes=[],
        target_nodes=inhibitors,
        attribute="uid",  # cheating to split the nodes individually
    )
    plt.tight_layout()
    title = f"t3_strongest_inhibitors_min_degree={min_degree}"
    plt.savefig(os.path.join(FOLDER, title + ".pdf"))
    plt.close()


def focus_strongest_inhibitors_t3_effective(
    min_degree: int = 5,
):
    """
    Focus on the strongest inhibitors in the T3 MDN-MNs neurons.
    """
    # Identify the core inhibitors
    l2_subconnections = get_length_n_t3_connections(n_hops=2)
    motor_neurons = mns_helper.get_leg_motor_neurons(
        l2_subconnections, leg="h", side="RHS"
    )
    mdns = mdn_helper.get_subdivided_mdns(l2_subconnections, neuropil="hl", side="RHS")
    graph = l2_subconnections.paths_length_n(2, mdns, motor_neurons)
    interneurons = set(graph.nodes) - set(mdns) - set(motor_neurons)
    inhibitors = interneurons.intersection(
        l2_subconnections.get_neuron_ids({"predictedNt:string": "gaba"})
    )
    inhibitors = [
        inhibitor
        for inhibitor in inhibitors
        if graph.out_degree(inhibitor) >= min_degree
    ]
    # Place back in the larger graph
    full_vnc = mdn_helper.get_vnc_split_MDNs_by_neuropil(  # only works because it's the same dataset at the one given by get_length_n_t3_connections
        not_connected=mdn_helper.get_mdn_bodyids()
    )
    vnc = full_vnc.get_connections_with_only_traced_neurons()
    # Get the effective connectivity matrix, separating positive and negative paths
    vnc_mat_exc = vnc.get_cmatrix(type_="norm")
    vnc_mat_exc.square_positive_paths_only()
    vnc_mat_exc.restrict_from_to(inhibitors, inhibitors, input_type="uid")
    vnc_mat_inh = vnc.get_cmatrix(type_="norm")
    vnc_mat_inh.square_negative_paths_only()
    vnc_mat_inh.restrict_from_to(inhibitors, inhibitors, input_type="uid")
    # Get a graph view of the effective connections
    graph_inh = nx.from_numpy_array(
        vnc_mat_inh.get_matrix(),
        nodelist=vnc_mat_inh.get_uids(),
        create_using=nx.DiGraph(),
    )
    graph_exc = nx.from_numpy_array(
        vnc_mat_exc.get_matrix(),
        nodelist=vnc_mat_exc.get_uids(),
        create_using=nx.DiGraph(),
    )

    # Plot the graph
    _, axs = plt.subplots(1, 3, figsize=(3 * params.FIGSIZE[0], params.FIGSIZE[1]))
    _, pos = nx_design.draw_graph(
        graph_inh,
        ax=axs[0],
        add_legend=False,
        label_nodes=True,
        pos_nx_method=nx.circular_layout,
        return_pos=True,
    )
    _ = nx_design.draw_graph(
        graph_exc,
        pos=pos,
        ax=axs[1],
        add_legend=False,
        label_nodes=True,
    )
    # Right: inhibition and excitation are combined
    graph = nx_utils.sum_graphs(graph_inh, graph_exc)
    _ = nx_design.draw_graph(
        graph,
        pos=pos,
        ax=axs[2],
        add_legend=False,
        label_nodes=True,
    )
    plt.tight_layout()
    title = f"t3_strongest_inhibitors_min_degree={min_degree}_effective"
    plt.savefig(os.path.join(FOLDER, title + ".pdf"))
    plt.close()


def focus_specific_neurons_effective(
    focus_neurons: list,
):
    """
    Plot the effective connection (2 hops) between the focus neurons.
    """
    # Place back in the larger graph
    full_vnc = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
    )
    vnc = full_vnc.get_connections_with_only_traced_neurons()
    # print out information
    for neuron in focus_neurons:
        print(f"Neuron {neuron}: {vnc.get_node_label(neuron)}")
    # Get the effective connectivity matrix, separating positive and negative paths
    vnc_mat_exc = vnc.get_cmatrix(type_="norm")
    vnc_mat_exc.square_positive_paths_only()
    vnc_mat_exc.restrict_from_to(focus_neurons, focus_neurons, input_type="uid")
    vnc_mat_inh = vnc.get_cmatrix(type_="norm")
    vnc_mat_inh.square_negative_paths_only()
    vnc_mat_inh.restrict_from_to(focus_neurons, focus_neurons, input_type="uid")
    # Get a graph view of the effective connections
    graph_inh = nx.from_numpy_array(
        vnc_mat_inh.get_matrix(),
        nodelist=vnc_mat_inh.get_uids(),
        create_using=nx.DiGraph(),
    )
    graph_exc = nx.from_numpy_array(
        vnc_mat_exc.get_matrix(),
        nodelist=vnc_mat_exc.get_uids(),
        create_using=nx.DiGraph(),
    )

    # Plot the graph
    # Left: inhibition and excitation are split
    _, axs = plt.subplots(1, 3, figsize=(3 * params.FIGSIZE[0], params.FIGSIZE[1]))
    _, pos = nx_design.draw_graph(
        graph_inh,
        ax=axs[0],
        add_legend=False,
        label_nodes=True,
        pos_nx_method=nx.circular_layout,
        return_pos=True,
    )
    _ = nx_design.draw_graph(
        graph_exc,
        pos=pos,
        ax=axs[1],
        add_legend=False,
        label_nodes=True,
    )
    # Right: inhibition and excitation are combined
    graph = nx_utils.sum_graphs(graph_inh, graph_exc)
    _ = nx_design.draw_graph(
        graph,
        pos=pos,
        ax=axs[2],
        add_legend=False,
        label_nodes=True,
    )
    plt.tight_layout()
    title = "specific_neurons_inh_effective"
    for neuron in focus_neurons:
        title += f"_{neuron}"
    plt.savefig(os.path.join(FOLDER, title + ".pdf"))
    plt.close()


def print_name_neurons(focus_neurons):
    # full graph
    full_vnc = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
    )
    vnc = full_vnc.get_connections_with_only_traced_neurons()
    # print out information
    for neuron in focus_neurons:
        print(f"Neuron {neuron}: {vnc.get_node_label(neuron)}")
        bodyid = vnc.get_bodyids_from_uids([neuron])[0]
        print(f"Bodyid of {neuron}: {bodyid}")


def control_t3_motor_neuron_clusters():
    """
    Cluster motor neurons based on all their inputs in T3.
    """
    # Data loading
    full_VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
    )
    VNC = full_VNC.get_connections_with_only_traced_neurons()
    motor_neurons = mns_helper.get_leg_motor_neurons(VNC, leg="h", side="RHS")

    # Restricting
    cmatrix = VNC.get_cmatrix(type_="unnorm")
    premotor_neurons = cmatrix.list_upstream_neurons(motor_neurons)
    nodes = list(set(motor_neurons).union(premotor_neurons))
    cmatrix.restrict_nodes(nodes)

    # Clustering
    sim_matrix, uid_clusters, index_clusters = cmatrix.detect_clusters(
        distance="cosine",
        method="hierarchical",
        cutoff=0.5,
        cluster_size_cutoff=2,
        cluster_data_type="uid",
    )

    # Draw the matrix
    clustered_mat = sim_matrix.get_matrix()
    mat = np.zeros((clustered_mat.shape[0], clustered_mat.shape[1]))
    for cluster in index_clusters:
        mat[cluster[0] : cluster[-1] + 1, cluster[0] : cluster[-1] + 1] = 1
    ax, _ = sim_matrix.imshow(savefig=False, title="Clustered T3 (pre)-motor neurons")
    ax.imshow(mat, cmap="binary", alpha=0.5)
    plt.savefig(os.path.join(FOLDER, "control_clustered_t3_motor_neurons_matrix.pdf"))
    plt.close()

    # Save the clusters as csv file
    cluster_file = os.path.join(FOLDER, "control_clusters.csv")
    with open(cluster_file, "w") as f:
        for i, cluster in enumerate(uid_clusters):
            for uid in cluster:
                f.write(f"{uid}\n")
            f.write("\n")
    return cluster_file


def compare_mn_clusters_control_vs_mdnl2():
    """
    See to what extent the clusters obtained from all premotor neurons differ
    from the ones obtained from the MDN induced graph.
    """
    # Data loading
    full_VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
    )
    VNC = full_VNC.get_connections_with_only_traced_neurons()
    motor_neurons = mns_helper.get_leg_motor_neurons(VNC, leg="h", side="RHS")
    mdn_induced_clusters = read_clusters_from_file(os.path.join(FOLDER, "clusters.csv"))
    control_clusters = read_clusters_from_file(
        os.path.join(FOLDER, "control_clusters.csv")
    )

    # Restrict clusters to motor neurons
    motor_mdn_clusters = []
    for cluster in mdn_induced_clusters:
        motor_cluster = [uid for uid in cluster if uid in motor_neurons]
        if motor_cluster and len(motor_cluster) > 3:
            motor_mdn_clusters.append(motor_cluster)
    motor_control_clusters = []
    for cluster in control_clusters:
        motor_cluster = [uid for uid in cluster if uid in motor_neurons]
        if motor_cluster and len(motor_cluster) > 3:
            motor_control_clusters.append(motor_cluster)

    for i, c1 in enumerate(motor_mdn_clusters):
        for j, c2 in enumerate(motor_control_clusters):
            print(
                f"Overlap between clusters {i} (lenght({len(c1)})) and {j} (length {len(c2)}): {len(set(c1) & set(c2))}"
            )


if __name__ == "__main__":
    # show_length2_t3()
    # hind_leg_muscles_graph(muscle_ = 'Tr flexor')
    # cluster_file = cluster_t3_graph(n_hops=2) # to run only once
    # cluster_file = os.path.join(FOLDER, 'clusters.csv')
    # display_l2_t3_graph_with_clusters(cluster_file)
    # motor_neuron_clusters_file = display_l2_t3_to_motor_neuron_clusters(
    #    cluster_file
    #    )
    # motor_neuron_clusters_file = os.path.join(FOLDER, 'motor_clusters.csv')
    # intersection_upstream_motor_neuron_pools(motor_neuron_clusters_file)
    # t3_nttype_interneurons(nt_type='gaba', n_hops=2, min_degree=5)
    # focus_strongest_inhibitors_t3(min_degree=5)
    # focus_strongest_inhibitors_t3_effective(min_degree=5)
    # central_inhibitors = [267449,345370,291605,84314]
    secondary_inhibitors = [123784,158676,229541,246719] # control 'motor primitive clusters'
    focus_specific_neurons_effective(secondary_inhibitors)
    # print_name_neurons(central_inhibitors)
    # control_t3_motor_neuron_clusters()
    # compare_mn_clusters_control_vs_mdnl2()
    pass

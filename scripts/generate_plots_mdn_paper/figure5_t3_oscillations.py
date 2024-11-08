'''
Figure 5: T3 Oscillations

Show how oscillations and coordinated movements are generated in T3.
'''

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import params
import mdn_paper_helper_functions as paper_funcs
import specific_neurons.mdn_helper as mdn_helper
import specific_neurons.motor_neurons_helper as mns_helper
import utils.nx_design as nx_design
import utils.matrix_design as matrix_design


FOLDER_NAME = 'Figure_5_t3_oscillations'
FOLDER = os.path.join(params.FIG_DIR, FOLDER_NAME)
os.makedirs(FOLDER, exist_ok=True)

# -------------------------- Helper functions -------------------------- #
def get_length_n_t3_connections(n_hops: int = 2):
    # Loading the connectivity data
    full_VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
        )
    VNC = full_VNC.get_connections_with_only_traced_neurons()
    mdns = mdn_helper.get_subdivided_mdns(
        VNC,
        neuropil = 'hl',
        side = 'RHS'
        )
    target_neurons = mns_helper.get_leg_motor_neurons(
        VNC,
        leg = 'h',
        side = 'RHS'
    )

    l2_graph = VNC.paths_length_n(n_hops, mdns, target_neurons)
    # get the connections object with nodes involved in paths of length 2
    l2_connections = VNC.subgraph(
        nodes = l2_graph.nodes(),
        )
    graph = l2_connections.get_graph()
    edges = graph.edges()
    # get the edges that are not from target to target
    edges = [
        edge
        for edge in edges
        if not all([uid in target_neurons for uid in edge])
        ]
    subconnections = l2_connections.subgraph(
        edges = edges
        ) # all nodes involved in the path, new Connections object
    
    return subconnections

def read_clusters_from_file(cluster_file: str):
    '''
    Read the clusters from a file and return them as a list of lists of int.
    '''
    clusters = []
    with open(cluster_file, 'r') as f:
        cluster = []
        for line in f:
            if line == '\n':
                clusters.append(cluster)
                cluster = []
            else:
                cluster.append(int(line.strip()))
    return clusters

# -------------------------- Main functions -------------------------- #

def show_length2_t3(
    attribute: str = 'class:string',
    syn_thresh: int = None,
    ):

    side = 'RHS'
    neuropil = 'hl'
    # Loading the connectivity data
    subconnections = get_length_n_t3_connections(n_hops=2)
    mdns = mdn_helper.get_subdivided_mdns(
        subconnections,
        neuropil = neuropil,
        side = side
        )
    target_neurons = mns_helper.get_leg_motor_neurons(
        subconnections,
        leg = 'h',
        side = side
    )

    # draw the graph
    ax = subconnections.draw_graph_concentric_by_attribute(
        title=f'MDNs|{side}_{neuropil}_to_MNs_2_hops',
        attribute='target:string',
        center_nodes=mdns,
        target_nodes=target_neurons,
        save=False,
        syn_threshold=syn_thresh,
        )
    
    plt.tight_layout()
    title = f'mdn_to_mns_T3_attribute={attribute}_syn-threshold={syn_thresh}'
    title = title.replace('/', '|')
    title = title.replace(' ', '-')
    plt.savefig(os.path.join(FOLDER, title+'.pdf'))
    plt.close()

def hind_leg_muscles_graph(
        muscle_: str,
        side_: str = 'RHS',
        n_hops: int = 2,
        label_nodes: bool = True,
        ):
    '''
    Show the graph of the neurons contacting MDNs -> motor neurons within n hops
    for the front leg muscles.
    '''
    target = {
        'class:string': 'motor neuron',
        'somaSide:string': side_,
        'subclass:string': 'hl',
        'target:string': muscle_,
        }
    title, axs = paper_funcs.graph_from_mdn_to_muscle(
        target,
        n_hops=n_hops,
        label_nodes=label_nodes,
        )
    plt.savefig(os.path.join(FOLDER, title+'.pdf'))
    plt.close()

def cluster_t3_graph(n_hops: int = 2):
    '''
    Show the graph of the neurons contacting MDNs -> motor neurons within n hops
    for the hind leg muscles.
    '''
    subconnections = get_length_n_t3_connections(n_hops=n_hops)
    mdns = mdn_helper.get_subdivided_mdns(
        subconnections,
        neuropil = 'hl',
        side = 'RHS'
        )
    target_neurons = mns_helper.get_leg_motor_neurons(
        subconnections,
        leg = 'h',
        side = 'RHS'
    )

    # Cluster based on connectivity similarity
    t3_cmatrix = subconnections.get_cmatrix(type_ = 'unnorm')
    t3_sim, uid_clusters, index_clusters = t3_cmatrix.detect_clusters(
        distance = 'cosine',
        method = 'hierarchical',
        cutoff = 0.5,
        cluster_size_cutoff = 2,
        cluster_data_type = 'uid',
    )

    # draw a matrix where all entries are white except for the boundaries
    # between clusters as defined by the number of elements in each list in
    # the clusters list.
    clustered_mat = t3_sim.get_matrix()
    mat = np.zeros((clustered_mat.shape[0], clustered_mat.shape[1]))
    for cluster in index_clusters:
        mat[cluster[0]:cluster[-1]+1, cluster[0]:cluster[-1]+1] = 1
    ax, _ = t3_sim.imshow(savefig=False, title='Clustered T3 matrix')
    ax.imshow(mat, cmap='binary', alpha=0.5)
    plt.savefig(os.path.join(FOLDER, 'clustered_t3_matrix.pdf'))
    plt.close()
    
    # Plot the different graphs
    fig, axs = plt.subplots(
        1,
        len(uid_clusters),
        figsize=(len(uid_clusters)*params.FIGSIZE[0], params.FIGSIZE[1])
        )
    for i, cluster in enumerate(uid_clusters):
        nodes = cluster + list(mdns) + list(target_neurons)
        try:
            graph = subconnections.subgraph(nodes)
            _ = graph.draw_graph_concentric_by_attribute(
                title=f'Cluster {i+1}',
                ax=axs[i],
                center_nodes=mdns,
                target_nodes=target_neurons,
                save=False,
                label_nodes=True,
                attribute='target:string',
                )
        except:
            pass
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, 'clusters_as_graph.pdf'))
    plt.close()     

    # Save the clusters as csv file
    cluster_file = os.path.join(FOLDER, 'clusters.csv')
    with open(cluster_file, 'w') as f:
        for i, cluster in enumerate(uid_clusters):
            for uid in cluster:
                f.write(f'{uid}\n')
            f.write('\n')
    return cluster_file
       
def display_l2_t3_graph_with_clusters(cluster_file: str):
    '''
    Represent the graph of the neurons contacting MDNs -> motor neurons within
    2 hops. Nodes are colored based on the cluster they belong to.
    '''
    subconnections = get_length_n_t3_connections(n_hops=2)
    # read the clusters from the file
    uid_clusters = read_clusters_from_file(cluster_file)
    mdns = mdn_helper.get_subdivided_mdns(
        subconnections,
        neuropil = 'hl',
        side = 'RHS'
        )
    target_neurons = mns_helper.get_leg_motor_neurons(
        subconnections,
        leg = 'h',
        side = 'RHS'
    )

    # get the graph
    graph = subconnections.get_graph()
    _, pos = subconnections.draw_graph_concentric_by_attribute(
        center_nodes=mdns,
        target_nodes=target_neurons,
        save=False,
        return_pos=True,
        attribute='target:string',
    ) # get the positions of the nodes according to their nature

    # create a color map for the clusters
    colors = plt.cm.tab20.colors
    colors = [colors[i] for i in range(len(uid_clusters))]

    # add the 'color' attribute to the nodes as the cluster they belong to
    for i, cluster in enumerate(uid_clusters):
        for node in cluster:
            graph.nodes[node]['color'] = colors[i]
    # give -1 to the nodes that are not in a cluster
    remaining_nodes = set(
        graph.nodes()
        ) - set([node for cluster in uid_clusters for node in cluster])
    for node in remaining_nodes:
        graph.nodes[node]['color'] = params.DARKGREY

    # draw the graph
    ax = nx_design.draw_graph(
        graph,
        pos,
        label_nodes=True,
    )
    plt.savefig(os.path.join(FOLDER, 'l2_t3_graph_with_clusters.pdf'))
    plt.close()

def display_l2_t3_to_motor_neuron_clusters(cluster_file):
    '''
    Explore potential motor synergies through motor neurons that are clustered
    together based on their connectivity.
    '''
    clusters = read_clusters_from_file(cluster_file)
    subconnections = get_length_n_t3_connections(n_hops=2)
    motor_neurons = mns_helper.get_leg_motor_neurons(
        subconnections,
        leg = 'h',
        side = 'RHS'
    )
    mdns = mdn_helper.get_subdivided_mdns(
        subconnections,
        neuropil = 'hl',
        side = 'RHS'
        )

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
        figsize=(len(motor_clusters)*params.FIGSIZE[0], params.FIGSIZE[1])
        )
    positions = []
    
    for i, cluster in enumerate(motor_clusters):
        graph = subconnections.paths_length_n(2, mdns, cluster)
        specific_connections = subconnections.subgraph(
            nodes = graph.nodes(),
            edges = graph.edges(), # only the edges directly involved in the paths
            )
        _, pos = specific_connections.draw_graph_in_out_center_circle(
            title=f'Motor cluster {i+1}',
            ax=axs[i],
            input_nodes=mdns,
            output_nodes=cluster,
            save=False,
            label_nodes=True,
            return_pos=True,
            )
        positions.append(pos)
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, 'motor_clusters.pdf'))
    plt.close()

    # draw the complete graphs
    fig, axs = plt.subplots(
        1,
        len(motor_clusters),
        figsize=(len(motor_clusters)*params.FIGSIZE[0], params.FIGSIZE[1])
        )
    for i, cluster in enumerate(motor_clusters):
        graph = subconnections.paths_length_n(2, mdns, cluster)
        specific_connections = subconnections.subgraph(
            nodes = graph.nodes(),
            )
        _ = specific_connections.display_graph(
            title=f'Motor cluster {i+1}',
            ax=axs[i],
            save=False,
            label_nodes=True,
            pos = positions[i],
            )
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, 'motor_clusters_complete.pdf'))
    plt.close()

    # save the motor clusters to a file
    motor_cluster_file = os.path.join(FOLDER, 'motor_clusters.csv')
    _ = subconnections.include_node_attributes(['target:string'])
    df = subconnections.list_neuron_properties(
        neurons = motor_neurons,
        input_type = 'uid',
        )

    # add a 'cluster' column to the dataframe based on bodyid indexing
    df['cluster'] = -1
    print(df)
    for i, cluster in enumerate(motor_clusters):
        for c in cluster:
            df.loc[df['uid'] == c, 'cluster'] = i 
    df = df.sort_values(by=['cluster'])
    df.to_csv(motor_cluster_file, index=False)
    return motor_cluster_file

def intersection_upstream_motor_neuron_pools(
        motor_neuron_file,
        syn_threshold: int = None,
        ):
    '''
    Plot the overlap of the upstream motor neuron pools for the different
    clusters of motor neurons as a matrix.
    '''
    cluster_data = pd.read_csv(motor_neuron_file)
    cluster_ids = list(set(cluster_data['cluster']) - {-1})
    nb_clusters = len(cluster_ids)    

    # load connectivity data
    subconnections = get_length_n_t3_connections(n_hops=2)
    motor_neurons = mns_helper.get_leg_motor_neurons(
        subconnections,
        leg = 'h',
        side = 'RHS'
    )
    mdns = mdn_helper.get_subdivided_mdns(
        subconnections,
        neuropil = 'hl',
        side = 'RHS'
        )

    # draw matrix to show how much the interneurons upstream overlap
    neurons_up = {}
    for i, c in enumerate(cluster_ids):
        cluster = list(cluster_data[cluster_data['cluster'] == c]['uid'])
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
            nb_interneurons[i, j] = len(
                set(neurons_up[i]) & set(neurons_up[j])
                )
    ax = matrix_design.imshow(
        nb_interneurons,
        title='Overlap of upstream interneurons',
        xlabel='Cluster',
        ylabel='Cluster',
        save=False,
        vmin=0,
        cmap=params.grey_heatmap,
        )
    title = 'motor_neuron_clusters_upstream_overlap'
    if syn_threshold:
        title += f'_syn-threshold={syn_threshold}'
    plt.savefig(os.path.join(FOLDER, title+'.pdf'))
    plt.close()

def motor_groups_upstream_interactions(
        motor_neuron_file,
        syn_threshold: int = None
        ):
    '''
    Plot the connections between the pools of interneurons upstream of the 
    clusters of motor neurons.
    This is done by looking at the interneurons uniquely upstream of each 
    cluster.
    '''
    cluster_data = pd.read_csv(motor_neuron_file)
    cluster_ids = list(set(cluster_data['cluster']) - {-1})
    nb_clusters = len(cluster_ids)    

    # load connectivity data
    subconnections = get_length_n_t3_connections(n_hops=2)
    motor_neurons = mns_helper.get_leg_motor_neurons(
        subconnections,
        leg = 'h',
        side = 'RHS'
    )
    mdns = mdn_helper.get_subdivided_mdns(
        subconnections,
        neuropil = 'hl',
        side = 'RHS'
        )

    # Find upstream neurons for each cluster
    neurons_up = {}
    for i, c in enumerate(cluster_ids):
        cluster = list(cluster_data[cluster_data['cluster'] == c]['uid'])
        graph = subconnections.paths_length_n(
            n=2,
            source=mdns,
            target=cluster,
            syn_threshold=syn_threshold,
            )
        interneurons = set(graph.nodes()) - set(motor_neurons) - set(mdns)
        neurons_up[i] = interneurons



if __name__ == '__main__':
    #show_length2_t3()
    #hind_leg_muscles_graph(muscle_ = 'Tr flexor')
    #cluster_file = cluster_t3_graph(n_hops=2) # to run only once
    #cluster_file = os.path.join(FOLDER, 'clusters.csv')
    #display_l2_t3_graph_with_clusters(cluster_file)
    #motor_neuron_clusters_file = display_l2_t3_to_motor_neuron_clusters(
    #    cluster_file
    #    )
    motor_neuron_clusters_file = os.path.join(FOLDER, 'motor_clusters.csv')
    #intersection_upstream_motor_neuron_pools(motor_neuron_clusters_file)
    motor_groups_upstream_interactions(motor_neuron_clusters_file)
    pass
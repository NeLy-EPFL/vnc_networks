import matplotlib.pyplot as plt
import numpy as np
from vnc_networks.vnc_networks import params
from vnc_networks.vnc_networks.connections import Connections
import pandas


def plot_cluster_similarity_matrix(vnc : Connections,subset : list[params.UID] = None , clustering_method : str = 'markov'
                                   ,distance_metric : str = 'cosine_in'
                                   ,cutoff : float = 0.5 ,c_min : int = 4):
    m = vnc.get_cmatrix()
    premotor_neurons = m.list_upstream_neurons(subset)
    nodes = list(set(subset).union(premotor_neurons))
    m.restrict_nodes(nodes)
    (
        clustered_cmatrix,  # clustered similarity matrix as cmatrix object
        uid_clusters,  # list of lists of uids in each cluster
        index_clusters,  # list of lists of indices in each cluster matching the clustered cmatrix
    ) = m.detect_clusters(
        distance=distance_metric,
        method=clustering_method,
        cutoff=cutoff,
        cluster_size_cutoff=c_min,
        cluster_data_type="uid",
        cluster_on_subset=subset,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    # Visualise the similarity matrix and its clusters
    clustered_sim_mat = clustered_cmatrix.get_matrix().todense()
    # create a matrix of zeros
    mat = np.zeros((clustered_sim_mat.shape[0], clustered_sim_mat.shape[1]))
    # draw the boundaries between clusters
    for cluster in index_clusters:
        mat[cluster[0]: cluster[-1] + 1, cluster[0]: cluster[-1] + 1] = 1
    _ = clustered_cmatrix.imshow(savefig=False, ax=ax, title="Clustered similarity matrix")
    ax.imshow(mat, cmap="binary", alpha=0.2)
    plt.show()


def show_data_connections(subnetwork : Connections) :

    def get_attributes_list (body_id : int) :
        uid = subnetwork.get_uids_from_bodyid(body_id)
        name = subnetwork.get_node_attribute(uid,"name")
        class_1 = subnetwork.get_node_attribute(uid,"class_1")
        return name,class_1

    def add_new_row(data_frame: pandas.DataFrame, source: str,source_name : str, source_class : str,
                    target: str,target_name : str, target_class : str, syn_count: int, eff_weights: int):
        new_row = {'source_body_ids': source,'source_name' : source_name, 'source_class' : source_class
            , 'target_body_id': target, 'target_name' : target_name, 'target_class' : target_class,
                   'syn_count': syn_count, 'eff_weights': eff_weights}
        data_frame.loc[len(data_frame)] = new_row

    data = subnetwork.get_dataframe()
    data_frame = pandas.DataFrame(columns=["source_body_ids","source_name","source_class", "target_body_id",
                                           "target_name","target_class", "syn_count", "eff_weights"])
    for index, row in data.iterrows():
        source_name, source_class = get_attributes_list(row["start_bid"])
        target_name, target_class = get_attributes_list(row["end_bid"])
        add_new_row(data_frame, row["start_bid"],source_name,source_class, row["end_bid"],target_name,target_class,
                    row["syn_count"], row["eff_weight"])

    display(data_frame)


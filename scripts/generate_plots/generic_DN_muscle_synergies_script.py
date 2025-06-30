import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnc_networks import MANC, UID, CMatrix, params
from vnc_networks.specific_neurons import dn_helper
from vnc_networks.utils import matrix_design

SAVEFILE = True


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Return the path for bash shells",
    )
    parser.add_argument(
        "--dn",
        type=str,
        default=None,
        help="which DN type to use?",
    )
    parser.add_argument(
        "--leg",
        type=str,
        default="hl",
        help="which leg: fl, ml or hl?",
    )
    return parser.parse_args()


def get_uids_from_cluster(cluster_number, df):
    return df[df["cluster"] == cluster_number]["uid"].values


def m1(
    cmat: CMatrix,  # connectivity matrix
    source: UID,  # premotor neuron
    targets: list[UID],  # neurons in cluster
    motor_neurons: list[UID],  # all motor neurons
):
    """
    M1 metric as defined above.
    """
    m1 = 0

    # Positive contributions to the targets
    direct_mat = cmat.get_matrix(
        row_ids=[source],
        column_ids=targets,
        input_type="uid",
    )
    direct_mat = abs(direct_mat)
    m1 += direct_mat.sum()

    # Negative contributions to the targets
    other_targets = [mn for mn in motor_neurons if mn not in targets]
    other_connections = cmat.get_matrix(
        row_ids=[source],
        column_ids=other_targets,
        input_type="uid",
    )
    other_connections = abs(other_connections)

    # alpha: #targets/#non targets
    alpha = len(targets) / len(other_targets)
    m1 -= alpha * other_connections.sum()

    return m1


def comp_delta_i(
    cmat: CMatrix,
    source: list[UID],
    targets: list[UID],
    motor_neurons: list[UID],
):
    """
    Compute the normalisation coefficient delta_i for the cluster i
    Normalisation per cluster such that the sum of the absolute values of the
    contributions is 1.
    """
    delta_inv = 0
    # Sum the absolute values of the m metrics (no simplification due to the abs)
    for s in source:
        delta_inv += np.abs(m1(cmat, s, targets, motor_neurons))
    return 1 / delta_inv


def digest_clustering_table(df):
    """
    Save a more digestable version of the clustering table:
    [cluster ; target ; count ; list_bodyids ; list_node_labels]
    """
    summary = (
        df.groupby(["cluster", "target"])
        .agg(
            count=("body_id", "count"),
            list_bodyids=("body_id", lambda x: list(x)),
            list_node_labels=("node_label", lambda x: list(x)),
        )
        .reset_index()
    )
    return summary


def draw_clustering(
    clustered_cmatrix: CMatrix,
    index_clusters: list,
    leg: str,
    side: str,
    DN_name: str,
    DN_FIGS: str,
    title: str = "Clustered similarity matrix",
):
    # Visualise the clusters

    _, ax = plt.subplots(figsize=(6, 6))
    # Visualise the similarity matrix and its clusters
    clustered_sim_mat = clustered_cmatrix.get_matrix().todense()
    # Save the ordering of the neurons for future use
    ordered_neurons_agnostic = clustered_cmatrix.get_uids()
    # create a matrix of zeros
    mat = np.zeros((clustered_sim_mat.shape[0], clustered_sim_mat.shape[1]))
    # draw the boundaries between clusters
    for cluster in index_clusters:
        mat[cluster[0] : cluster[-1] + 1, cluster[0] : cluster[-1] + 1] = 1
    ax, _ = clustered_cmatrix.imshow(
        savefig=False, ax=ax, title="Clustered similarity matrix"
    )

    # ax.imshow(mat, cmap="binary", alpha=0.4)
    # Instead of shading the clusters, draw the boundaries as black boxes
    for cluster in index_clusters:
        ax.add_patch(
            plt.Rectangle(
                (cluster[0] - 0.5, cluster[0] - 0.5),
                len(cluster),
                len(cluster),
                edgecolor="black",
                facecolor="none",
                lw=1,
            )
        )
    # force matrix to be square visually
    ax.set_aspect("equal")

    plt.savefig(
        os.path.join(DN_FIGS, title),
        dpi=params.DPI,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

    return ordered_neurons_agnostic


def main(DN_name: str, leg: str = "hl"):
    """
    Identify the muscle synergies induced by the DN and the responsible
    premotor neurons.
    """
    # Parameters
    manc_version = "v1.2"
    side = "RHS"

    # Where to save
    DN_DIR = f"{DN_name}_plots"
    FIG_DIR = MANC(manc_version).get_fig_dir()
    DN_FIGS = os.path.join(FIG_DIR, DN_DIR, leg)
    os.makedirs(DN_FIGS, exist_ok=True)

    # Load the data
    CR = MANC(manc_version)
    split_dn_vnc = dn_helper.get_vnc_split_DNs_by_neuropil(
        not_connected=dn_helper.get_dn_bodyids(name=DN_name),
        name=DN_name,
    )

    # Method choices
    # Method for clustering
    clustering_method = "markov"  # "markov"  #'hierarchical'
    # distance metric
    distance_metric = "cosine_in"
    # cutoff to define a cluster
    cutoff = 0.4  # for hierarchical
    inflation = 5  # for markov
    # minimum number of neurons in a cluster
    c_min = 4

    # Preprocessing
    motor_neurons = split_dn_vnc.get_neuron_ids(
        {
            "class_1": "motor",
            "class_2": leg,
            "side": side,
        }
    )

    # ----------- CONTROL ------------- #
    vnc_matrix = split_dn_vnc.get_cmatrix(type_="unnorm")
    # Cut it down to the motor and premotor connections
    premotor_neurons = vnc_matrix.list_upstream_neurons(motor_neurons)
    nodes = list(set(motor_neurons).union(premotor_neurons))
    vnc_matrix.restrict_nodes(nodes)
    # cluster the motor neurons using cosine similarity on the inputs
    (
        clustered_cmatrix,  # clustered similarity matrix as cmatrix object
        uid_clusters,  # list of lists of uids in each cluster
        index_clusters,  # list of lists of indices in each cluster matching the clustered cmatrix
    ) = vnc_matrix.detect_clusters(
        distance=distance_metric,
        method=clustering_method,
        cluster_size_cutoff=c_min,
        cluster_data_type="uid",
        cluster_on_subset=motor_neurons,
        cutoff=cutoff,  # kwargs for hierarchical
        inflation=inflation,  # kwargs for markov
    )

    # Visualise the clustering
    ordered_neurons_agnostic = draw_clustering(
        clustered_cmatrix,
        index_clusters,
        leg,
        side,
        DN_name,
        DN_FIGS,
        title=f"clustered_similarity_matrix_{leg}_{side}_{DN_name}_control.pdf",
    )

    # Save the clusters
    # create a df with a column for the cluster number, one for the neuron uid, one
    # for the neuron bodyid, and the rest for the defined neuron properties.
    neurons_in_clusters = [uid for cluster in uid_clusters for uid in cluster]
    # include 'target' and 'hemilineage' as these are useful for identifying the neurons
    _ = split_dn_vnc.get_node_attribute(uid=neurons_in_clusters, attribute="target")
    _ = split_dn_vnc.get_node_attribute(
        uid=neurons_in_clusters, attribute="hemilineage"
    )
    # retrieve the properties of the neurons in the clusters
    info_df = split_dn_vnc.list_neuron_properties(
        neurons=neurons_in_clusters,
        input_type="uid",
    )
    info_df["cluster"] = -1
    for i, cluster in enumerate(uid_clusters):
        info_df.loc[info_df["uid"].isin(cluster), "cluster"] = i
    info_df.sort_values(by=["cluster", "uid"], inplace=True)
    if SAVEFILE:
        info_df.to_csv(
            os.path.join(DN_FIGS, f"motor_clusters_{leg}_{side}_{DN_name}_control.csv"),
            index=False,
        )
    # save the summary of the clusters
    info_df_summary = digest_clustering_table(info_df)
    if SAVEFILE:
        info_df_summary.to_csv(
            os.path.join(
                DN_FIGS, f"motor_clusters_{leg}_{side}_{DN_name}_control_summary.csv"
            ),
            index=False,
        )

    # ------------- DN input bias ---------------- #
    # Select the MDN subdivisions that have synapses in the hind right leg
    input_neurons = dn_helper.get_subdivided_dns(
        VNC=split_dn_vnc,
        neuropil=leg,
        # side=side,
        name=DN_name,
    )

    # Keep only the connections that create a path from source to target
    subnetwork = split_dn_vnc.subgraph_from_paths(  # copy operation
        source=input_neurons,
        target=motor_neurons,
        n_hops=2,  # within 2 hops, i.e. only 1 interneuron
        keep_edges="intermediate",  # keep the connections between the interneurons
        # as well, but not between source neurons or between target neurons
        # can also be 'direct' (only direct paths) or 'all' (all connections between
        # recruited nodes)
    )

    # Get the connectivity matrix
    subnetwork_matrix = subnetwork.get_cmatrix(type_="unnorm")
    # Cut it down to the motor and premotor connections
    premotor_neurons = subnetwork_matrix.list_upstream_neurons(motor_neurons)
    nodes = list(set(motor_neurons).union(premotor_neurons))
    subnetwork_matrix.restrict_nodes(nodes)

    # cluster the motor neurons using cosine similarity on the inputs
    (
        sub_clustered_cmatrix,  # clustered similarity matrix as cmatrix object
        sub_uid_clusters,  # list of lists of uids in each cluster
        sub_index_clusters,  # list of lists of indices in each cluster matching the clustered cmatrix
    ) = subnetwork_matrix.detect_clusters(
        distance=distance_metric,
        method=clustering_method,
        cluster_size_cutoff=c_min,
        cluster_data_type="uid",
        cluster_on_subset=motor_neurons,
        cutoff=cutoff,  # kwargs for hierarchical
        inflation=inflation,  # kwargs for markov
    )

    # Visualise the clustering
    _ = draw_clustering(
        sub_clustered_cmatrix,
        sub_index_clusters,
        leg,
        side,
        DN_name,
        DN_FIGS,
        title=f"clustered_similarity_matrix_{leg}_{side}_{DN_name}_input.pdf",
    )

    # Plot the same matrix (MDN-induced similarity), but with the ordering from the original clustering
    fig, ax = plt.subplots(figsize=(6, 6))
    sub_clustered_cmatrix.restrict_from_to(
        row_ids=ordered_neurons_agnostic,  # same set of neurons, will not subset, only reorder
        column_ids=ordered_neurons_agnostic,  # same set of neurons, will not subset, only reorder
        keep_initial_order=False,  # we want the new ordering
    )
    _ = sub_clustered_cmatrix.imshow(
        savefig=False, ax=ax, title="Clustered similarity matrix"
    )
    ax.set_aspect("equal")
    fig.savefig(
        os.path.join(
            DN_FIGS,
            f"similarity_matrix_{leg}_{side}_MNs_{DN_name}_input_agnostic_ordering.pdf",
        ),
        dpi=params.DPI,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

    # Save the clusters
    # create a df with a column for the cluster number, one for the neuron uid, one
    # for the neuron bodyid, and the rest for the defined neuron properties.
    neurons_in_subclusters = [uid for cluster in sub_uid_clusters for uid in cluster]
    sub_info_df = split_dn_vnc.list_neuron_properties(
        neurons=neurons_in_subclusters,
        input_type="uid",
    )
    sub_info_df["cluster"] = -1
    for i, cluster in enumerate(sub_uid_clusters):
        sub_info_df.loc[sub_info_df["uid"].isin(cluster), "cluster"] = i
    sub_info_df.sort_values(by=["cluster", "uid"], inplace=True)
    sub_info_df.to_csv(
        os.path.join(DN_FIGS, f"motor_clusters_{leg}_{side}_{DN_name}_input.csv"),
        index=False,
    )
    # save the summary of the clusters
    sub_info_df_summary = digest_clustering_table(sub_info_df)
    if SAVEFILE:
        sub_info_df_summary.to_csv(
            os.path.join(
                DN_FIGS, f"motor_clusters_{leg}_{side}_{DN_name}_input_summary.csv"
            ),
            index=False,
        )

    # ------------- clustering comparison ---------------- #
    # info_df: agnostic
    agnostic_clusters = np.unique(info_df["cluster"])
    # sub_info_df: MDN-induced
    induced_clusters = np.unique(sub_info_df["cluster"])

    meta_mat = np.zeros((len(agnostic_clusters) + 1, len(induced_clusters) + 1))
    for j in induced_clusters:
        candidates = sub_info_df.loc[sub_info_df["cluster"] == j]["uid"].values

        for i in agnostic_clusters:
            selected = info_df.loc[
                (info_df["cluster"] == i) & (info_df["uid"].isin(candidates))
            ]
            meta_mat[i, j] = len(selected) / len(
                candidates
            )  # column-wise normalisation
        # the last row is the non clustered neurons in the agnostic clustering
        left_over = [
            uid
            for uid in motor_neurons
            if uid not in info_df["uid"].values and uid in candidates
        ]
        meta_mat[-1, j] = len(left_over) / len(candidates)
    # the last column is the non clustered neurons in the induced clustering
    candidates = [uid for uid in motor_neurons if not uid in sub_info_df["uid"].values]
    for i in agnostic_clusters:
        selected = info_df.loc[
            (info_df["cluster"] == i) & (info_df["uid"].isin(candidates))
        ]
        meta_mat[i, -1] = len(selected) / len(candidates)  # column-wise normalisation
    # the last element is the non clustered neurons in both clustering
    left_over = [
        uid
        for uid in motor_neurons
        if uid not in info_df["uid"].values and uid in candidates
    ]
    meta_mat[-1, -1] = len(left_over) / len(candidates)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    matrix_design.imshow(
        meta_mat,
        title="",
        ylabel="agnostic_cluster",
        xlabel="induced cluster",
        vmin=0,
        vmax=1,
        cmap=params.grey_heatmap,
        ax=ax,
    )
    # Add a dashed lines to separate the clusters from the last row/column
    ax.axhline(len(agnostic_clusters) - 0.5, color="gray", linestyle="--")
    ax.axvline(len(induced_clusters) - 0.5, color="gray", linestyle="--")

    fig.savefig(
        os.path.join(
            DN_FIGS,
            f"clustering_comparison_{leg}_{side}_agnostic_vs_{DN_name}_induced.eps",
        ),
        dpi=params.DPI,
        bbox_inches="tight",
    )
    plt.close()

    # ------------- PREMOTOR NEURONS ---------------- #
    dn_based_clustering_df = sub_info_df.copy()
    _ = split_dn_vnc.get_node_attribute(uid=neurons_in_clusters, attribute="nt_type")

    # visualise contribution for each cluster
    clusters = np.unique(dn_based_clustering_df["cluster"].values)
    nb_clusters = len(clusters)

    pmn_df = pd.DataFrame()

    fig, axs = plt.subplots(1, nb_clusters, figsize=(30, 6))

    for i, c in enumerate(clusters):
        c_i = get_uids_from_cluster(c, dn_based_clustering_df)
        delta_i = comp_delta_i(
            cmat=subnetwork_matrix,
            source=premotor_neurons,
            targets=c_i,
            motor_neurons=motor_neurons,
        )
        c1_m1 = [
            m1(
                cmat=subnetwork_matrix,
                source=pmn,
                targets=c_i,
                motor_neurons=motor_neurons,
            )
            * delta_i
            for pmn in premotor_neurons
        ]

        # assert_almost_equal(np.sum(c1_m1), 1), f"Cluster {c} does not sum to 1, but to {np.sum(c1_m1)}"
        x = np.linspace(0, len(c1_m1), len(c1_m1))
        nts = split_dn_vnc.get_node_attribute(uid=premotor_neurons, attribute="nt_type")
        colors = [params.NT_TYPES[nt]["color"] for nt in nts]
        axs[i].scatter(
            x, c1_m1, color=colors, marker="o", linestyle="None", label=f"cluster {c}"
        )
        # get error shading
        std_i = 3 * np.std(c1_m1)  # 99th percentile
        mean_i = np.mean(c1_m1)
        axs[i].axhline(y=mean_i, linewidth=4, color="k")
        x = np.linspace(0, len(c1_m1), len(c1_m1))
        axs[i].fill_between(x, mean_i - std_i, mean_i + std_i, alpha=0.1, color="grey")
        axs[i].legend(loc="best")
        axs[i].set_ylim([-0.25, 0.5])

        # save the neurons that contribute significantly
        significant_neurons = [
            node for i, node in enumerate(premotor_neurons) if c1_m1[i] > mean_i + std_i
        ]
        info = split_dn_vnc.list_neuron_properties(
            neurons=significant_neurons,
            input_type="uid",
        )
        info["m1 score"] = info["uid"].apply(
            lambda x: m1(
                cmat=subnetwork_matrix,
                source=x,
                targets=c_i,
                motor_neurons=motor_neurons,
            )
        )
        info["cluster connecting to"] = c
        info.sort_values(by="m1 score", ascending=False, inplace=True)
        pmn_df = pd.concat([pmn_df, info])

    plt.savefig(
        os.path.join(
            DN_FIGS, f"motor_clusters_{leg}_{side}_{DN_name}_premotor_hubs.pdf"
        ),
        dpi=params.DPI,
        bbox_inches="tight",
    )
    plt.close()

    if SAVEFILE:
        notes_file = os.path.join(DN_FIGS, f"motor_neuron_notes_{DN_name}.csv")
        with open(notes_file, "w") as f:
            f.write("nb of motor neurons: ")
            f.write(f"{len(motor_neurons)}\n")
            f.write("nb of premotor neurons: ")
            f.write(f"{len(premotor_neurons)}\n")
            f.write("\n")
            f.write("motor_neurons\n")
            for mn in motor_neurons:
                f.write(f"{mn}\n")
            f.write("\n")
            f.write("premotor_neurons\n")
            for pmn in premotor_neurons:
                f.write(f"{pmn}\n")

    if SAVEFILE:
        pmn_df.to_csv(
            os.path.join(
                DN_FIGS, f"motor_clusters_{leg}_{side}_{DN_name}_premotor_hubs.csv"
            ),
            index=False,
        )

    # make sur that the premotor neurons are in the same order as the clusters
    pmn_df.sort_values(by="cluster connecting to", inplace=True)
    premns = pmn_df["uid"].values
    # get the motor neurons in the same order as the clusters
    sub_info_df.sort_values(by="cluster", inplace=True)
    mns_in_clusters = sub_info_df["uid"].values

    cmat = split_dn_vnc.get_cmatrix(type_="unnorm")
    cmat.restrict_from_to(premns, mns_in_clusters, keep_initial_order=False)

    _, axs = plt.subplots(1, 2, figsize=(12, 6))
    _ = cmat.imshow(ax=axs[0], savefig=False, title="lin scale")
    _ = cmat.imshow(ax=axs[1], savefig=False, log_scale=True, title="log scale")

    for ax in axs:
        ax.set_yticks(np.arange(len(premns)))
        ax.set_yticklabels(pmn_df["cluster connecting to"].values)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            DN_FIGS,
            f"special_premotor_to_motor_connections_{leg}_{side}_{DN_name}.pdf",
        ),
        dpi=params.DPI,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    # ARGS = parse_args()
    # main(DN_name=ARGS.dn, leg=ARGS.leg)
    DNa = ["DNa01", "DNa02", "DNa06", "DNa09", "DNa11", "DNa13"]
    DNb = ["DNb02", "DNb05", "DNb01", "DNb06"]
    DNg = [
        "DNg10",
        "DNg11",
        "DNg12",
        "DNg13",
        "DNg14",
        "DNg34",
        "DNg35",
        "DNg69",
        "DNg74",
        "DNg88",
        "DNg97",
        "DNg100",
        "DNg103",
        "DNg108",
    ]
    DNp = ["DNp07", "DNp09", "DNp10", "DNp42"]
    other_DNs = ["DNfl015", "DNxn164", "MDN"]

    DNs = DNa + DNb + DNg + DNp + other_DNs
    legs = ["fl", "ml", "hl"]

    for dn in ["DNg15"]:
        for leg in legs:
            try:
                main(DN_name=dn, leg=leg)
            except Exception as e:
                print(f"Error for {dn} on {leg}: {e}")
                continue

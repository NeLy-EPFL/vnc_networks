"""
Figure 1

Show that the branches of MDN do recruit circuits that are somewhat independent
in the VNC. This is done by splitting the MDNs according to their synapse
locations.
"""

import copy
import os
from typing import Optional

import numpy as np
import pandas as pd
import params
import specific_neurons.mdn_helper as mdn_helper
import specific_neurons.motor_neurons_helper as mns_helper
from connections import Connections
from get_nodes_data import get_neuron_bodyids
from matplotlib import pyplot as plt
from neuron import Neuron
from params import NeuronAttribute
from tqdm import tqdm
from utils import matrix_design, plots_design

FOLDER_NAME = "Figure_1_independent"
FOLDER = os.path.join(params.FIG_DIR, FOLDER_NAME)
os.makedirs(FOLDER, exist_ok=True)


# -------------------------- Helper functions -------------------------- #
def write_neuron_dataframe(
    neurons: list[int], VNC: Connections, filename: str = "test_df"
):
    """
    Write infromation about the neurons in the list in a csv file.
    """
    # get the boyids from the uids
    bodyids = VNC.get_bodyids_from_uids(neurons)
    # get neuron-specific information
    df = pd.DataFrame()
    for bid in tqdm(bodyids, desc="writing information about neurons to file"):
        neuron = Neuron(bid)  # intialises information in subfields
        neuron_data = neuron.get_data()  # df
        print(neuron_data)
        df = pd.concat([df, neuron_data], axis=0, ignore_index=True)
    df.to_csv(os.path.join(FOLDER, filename + ".csv"))


# -------------------------- Main functions -------------------------- #
def venn_mdn_branches_neuropil_direct(attribute: NeuronAttribute = "class:string"):
    """
    Draw a Venn diagram of the neurons directly downstream of MDN split by where
    the synapses are.
    Draw a bar plot of the attribute of the neurons in the intersection of the
    3 groups, as well as the 3 groups separately.
    """
    # Loading the connectivity data
    full_VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
    )  # exclude connections from MDNs to MDNs
    VNC = (
        full_VNC.get_connections_with_only_traced_neurons()
    )  # exclude untraced neurons for statistics
    # Working with matrix representation, get n-th order connections
    cmatrix = VNC.get_cmatrix(type_="norm")

    # Get the uids of neurons split by MDN synapses in leg neuropils
    mdn_uids = VNC.get_neuron_ids({"type:string": "MDN"})
    list_down_neurons = []  # will become a list with 3 sets
    for i in range(3):
        neuropil = "LegNp(T" + str(i + 1) + ")"  # synapses of MDNs in T1/T2/T3
        mdn_neuropil = [uid for uid in mdn_uids if neuropil in VNC.get_node_label(uid)]
        down_partners = cmatrix.list_downstream_neurons(mdn_neuropil)  # uids
        list_down_neurons.append(set(down_partners))

    # === I. Plot the Venn diagram
    _ = plots_design.venn_3(
        sets=list_down_neurons,
        set_labels=["MDNs|T1", "MDNs|T2", "MDNs|T3"],
        title="Neurons directly downstream of MDNs",
    )

    # saving
    plt.savefig(os.path.join(FOLDER, "venn_mdn_branches_neuropil_direct.pdf"))
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
    ax.set_title("Intersection of all 3 groups")
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "venn_mdn_branches_neuropil_bar_intersection.pdf"))
    plt.close()

    # save the Connection data for the neurons in the intersection
    write_neuron_dataframe(
        full_intersection, VNC, filename="venn_mdn_branches_neuropil_intersection"
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
        set_neurons = (
            list_down_neurons[i]
            - list_down_neurons[(i + 1) % 3]
            - list_down_neurons[(i + 2) % 3]
        )
        VNC.draw_bar_plot(set_neurons, attribute, ax=axs[i])
        axs[i].set_title(f"unique downstream of MDNs|T{i+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "venn_mdn_branches_neuropil_bar_exclusion.pdf"))
    plt.close()


def venn_t3_subbranches(n_clusters: int = 3):
    """
    Split the T3 part in subbranches and look at the isolated circuits.
    Add the previous figure the spatial visualisation of the synapses.
    """
    VNC = mdn_helper.get_connectome_with_MDN_t3_branches(
        n_clusters=n_clusters
    )  # already pruned of non Traced neurons
    mdn_uids = VNC.get_neuron_ids({"type:string": "MDN"})

    # Get the direct downstream partners for each subdivision
    cmatrix = VNC.get_cmatrix(type_="norm")
    down_neurons = {}
    for uid in mdn_uids:
        down_partners = cmatrix.list_downstream_neurons(uid)
        down_neurons[uid] = down_partners

    # Draw a Venn diagram for the branches of each neuron
    mnd_bodyids = get_neuron_bodyids({"type:string": "MDN"})
    for mdn_bid in mnd_bodyids:
        single_mdn_uids = VNC.get_uids_from_bodyid(mdn_bid)
        single_mdn_uids = [
            uid for uid in single_mdn_uids if "-1" not in VNC.get_node_label(uid)
        ]  # keep only the T3 clusters, not the rest named as '-1'
        list_down_neurons = []
        for uid in single_mdn_uids:
            list_down_neurons.append(set(down_neurons[uid]))

        # Plot the Venn diagram
        _ = plots_design.venn_3(
            sets=list_down_neurons,
            set_labels=["0", "1", "2"],
            title=f"Neurons directly downstream of MDN {mdn_bid} T3 branches",
        )

        # saving
        plt.savefig(os.path.join(FOLDER, f"venn_mdn-{mdn_bid}_branches_t3_direct.pdf"))
        plt.close()


def confusion_matrix_mdn_to_mn(n_hops: int = 2):
    """
    Create a confusion matrix of the number of connections from MDN to motor
    neurons within n hops, where the rows are the MDNs split by neuropil and
    the columns are the motor neurons split by leg.
    Focus on the right side.
    """
    # Loading the connectivity data
    full_VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
    )  # exclude connections from MDNs to MDNs
    VNC = (
        full_VNC.get_connections_with_only_traced_neurons()
    )  # exclude untraced neurons for statistics

    # Get the uids of neurons split by MDN synapses in leg neuropils
    mdn_uids = mdn_helper.get_mdn_uids(VNC, side="R")
    mdn_neuropil = []
    for i in range(3):
        neuropil = "LegNp(T" + str(i + 1) + ")"
        mdn_neuropil.append(
            [uid for uid in mdn_uids if neuropil in VNC.get_node_label(uid)]
        )  # 2 left and 2 right MDNs

    # Get the uids of motor neurons split by leg
    list_motor_neurons = [[], [], []]
    for i, leg in enumerate(["f", "m", "h"]):
        leg_motor_neurons = list(
            mns_helper.get_leg_motor_neurons(VNC, leg=leg, side="RHS")
        )
        list_motor_neurons[i] = leg_motor_neurons

    # Get the summed connection strength up to n hops
    eff_weight_abs = VNC.get_cmatrix(type_="norm")
    eff_weight_abs.absolute()
    eff_weight_abs.within_power_n(n_hops)

    # Get the confusion matrix
    confusion_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            mat = copy.deepcopy(eff_weight_abs)
            mat.restrict_from_to(
                row_ids=mdn_neuropil[i],
                column_ids=list_motor_neurons[j],
                input_type="uid",
            )
            matrix = mat.get_matrix()
            confusion_matrix[i, j] = matrix.sum()

    # Plot the confusion matrix
    matrix_design.imshow(
        confusion_matrix,
        ylabel="MDN subdivision",
        row_labels=["T1", "T2", "T3"],
        xlabel="leg motor neuron",
        col_labels=["f", "m", "h"],
        title="Confusion matrix of MDN to MN connections",
        cmap=params.grey_heatmap,
        vmin=0,
    )
    plt.savefig(os.path.join(FOLDER, f"Fig1_{n_hops}_hops_total_weight.pdf"))
    plt.close()


def confusion_matrix_sign_assymetry_mdn_to_mn(
    mdn_side_: Optional[str] = None,
    mn_side_: Optional[str] = None,
):
    """
    Create a confusion matrix of the relative number of excitatory and
    inhibitory of connections from MDN to motor neurons within 2 hops,
    where the rows are the MDNs split by neuropil and the columns are the motor
    neurons split by leg.
    Focus on the right side only.
    """
    # Loading the connectivity data
    full_VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
    )  # exclude connections from MDNs to MDNs
    VNC = (
        full_VNC.get_connections_with_only_traced_neurons()
    )  # exclude untraced neurons for statistics

    # Get the uids of neurons split by MDN synapses in leg neuropils
    mdn_uids = mdn_helper.get_mdn_uids(VNC, side=mdn_side_)
    mdn_neuropil = []
    for i in range(3):
        neuropil = "LegNp(T" + str(i + 1) + ")"
        mdn_neuropil.append(
            [uid for uid in mdn_uids if neuropil in VNC.get_node_label(uid)]
        )  # 2 right MDNs

    # Get the uids of motor neurons split by leg
    list_motor_neurons = [[], [], []]
    for i, leg in enumerate(["f", "m", "h"]):
        if mn_side_ is None:
            leg_motor_neurons = list(mns_helper.get_leg_motor_neurons(VNC, leg=leg))
        else:
            leg_motor_neurons = list(
                mns_helper.get_leg_motor_neurons(VNC, leg=leg, side=mn_side_ + "HS")
            )
        list_motor_neurons[i] = leg_motor_neurons

    # Get the summed connection strength up to n hops
    pos_matrix = VNC.get_cmatrix(type_="norm")
    pos_matrix.square_positive_paths_only()
    neg_matrix = VNC.get_cmatrix(type_="norm")
    neg_matrix.square_negative_paths_only()

    # Show the connections T1-T1, T2-T2, T3-T3
    for k in range(3):
        pmat = copy.deepcopy(pos_matrix)
        pmat.restrict_from_to(
            row_ids=mdn_neuropil[k], column_ids=list_motor_neurons[k], input_type="uid"
        )
        nmat = copy.deepcopy(neg_matrix)
        nmat.restrict_from_to(
            row_ids=mdn_neuropil[k], column_ids=list_motor_neurons[k], input_type="uid"
        )
        # plot both matrices on top of each other
        fig, axs = plt.subplots(2, 1, figsize=(params.FIG_WIDTH, 2 * params.FIG_HEIGHT))
        matrix_design.imshow(
            pmat.get_matrix(),
            ylabel=f"MDN|T{k+1}",
            xlabel="leg motor neuron",
            title="Effective excitatory connections",
            cmap=params.diverging_heatmap,
            ax=axs[0],
        )
        matrix_design.imshow(
            nmat.get_matrix(),
            ylabel=f"MDN|T{k+1}",
            xlabel="leg motor neuron",
            title="Effective inhibitory connections",
            cmap=params.diverging_heatmap,
            ax=axs[1],
        )
        plt.tight_layout()
        title = f"Fig1_T{k+1}-T{k+1}_connections"
        if mdn_side_ is not None:
            title += f"_MDN-{mdn_side_}"
        if mn_side_ is not None:
            title += f"_MN-{mn_side_}"
        title += ".pdf"
        plt.savefig(os.path.join(FOLDER, title))
        plt.close()

    # Get the confusion matrix
    confusion_matrix = np.zeros((3, 3))
    confusion_matrix_tot = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            pmat = copy.deepcopy(pos_matrix)
            pmat.restrict_from_to(
                row_ids=mdn_neuropil[i],
                column_ids=list_motor_neurons[j],
                input_type="uid",
            )
            nmat = copy.deepcopy(neg_matrix)
            nmat.restrict_from_to(
                row_ids=mdn_neuropil[i],
                column_ids=list_motor_neurons[j],
                input_type="uid",
            )
            pos = pmat.get_matrix()
            neg = -1 * nmat.get_matrix()
            # elementwise sign imbalance computation
            imbalance = (pos - neg) / (pos + neg)
            imbalance[np.isnan(imbalance)] = 0
            tot = np.abs(pos - neg) / (pos + neg)
            tot[np.isnan(tot)] = 0
            confusion_matrix[i, j] = imbalance.sum() / imbalance.size
            confusion_matrix_tot[i, j] = tot.sum() / tot.size
    # Plot the confusion matrix
    ax = matrix_design.imshow(
        confusion_matrix,
        ylabel="MDN subdivision",
        row_labels=["T1", "T2", "T3"],
        xlabel="leg motor neuron",
        col_labels=["f", "m", "h"],
        title="Signed confusion matrix of MDN to MN connections",
        cmap=params.diverging_heatmap,
    )
    title = "Fig1_sign_imbalance_MDN"
    if mdn_side_ is not None:
        title += f"_MDN-{mdn_side_}"
    if mn_side_ is not None:
        title += f"_MN-{mn_side_}"
    plt.savefig(os.path.join(FOLDER, f"{title}.pdf"))
    plt.close()

    # Plot the confusion matrix
    ax = matrix_design.imshow(
        confusion_matrix_tot,
        ylabel="MDN subdivision",
        row_labels=["T1", "T2", "T3"],
        xlabel="leg motor neuron",
        col_labels=["f", "m", "h"],
        title="confusion matrix of MDN to each MN connection",
        cmap=params.grey_heatmap,
        vmin=0,
    )
    title = "Fig1_total_imbalance_MDN"
    if mdn_side_ is not None:
        title += f"_MDN-{mdn_side_}"
    if mn_side_ is not None:
        title += f"_MN-{mn_side_}"
    plt.savefig(os.path.join(FOLDER, f"{title}.pdf"))
    plt.close()


def sign_assymetry_mdn_to_mn(
    mdn_side_: Optional[str] = None, mn_side_: Optional[str] = None
):
    """
    Create a plot of the relative number of excitatory and
    inhibitory of connections from MDN to motor neurons within 2 hops.
    """
    # Loading the connectivity data
    full_VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
    )  # exclude connections from MDNs to MDNs
    VNC = (
        full_VNC.get_connections_with_only_traced_neurons()
    )  # exclude untraced neurons for statistics

    # Get the uids of neurons split by MDN synapses in leg neuropils
    mdn_uids = mdn_helper.get_mdn_uids(VNC, side=mdn_side_)
    mdn_neuropil = []
    for i in range(3):
        neuropil = "LegNp(T" + str(i + 1) + ")"
        mdn_neuropil.append(
            [uid for uid in mdn_uids if neuropil in VNC.get_node_label(uid)]
        )  # 2 right MDNs

    # Get the uids of motor neurons split by leg
    list_motor_neurons = [[], [], []]
    for i, leg in enumerate(["f", "m", "h"]):
        if mn_side_ is None:
            leg_motor_neurons = list(mns_helper.get_leg_motor_neurons(VNC, leg=leg))
        else:
            leg_motor_neurons = list(
                mns_helper.get_leg_motor_neurons(VNC, leg=leg, side=mn_side_ + "HS")
            )
        list_motor_neurons[i] = leg_motor_neurons

    # Get the summed connection strength up to n hops
    pos_matrix = VNC.get_cmatrix(type_="norm")
    pos_matrix.square_positive_paths_only()
    neg_matrix = VNC.get_cmatrix(type_="norm")
    neg_matrix.square_negative_paths_only()

    # Get the confusion matrix
    imbalance_metric = []

    for i in range(3):
        pmat = copy.deepcopy(pos_matrix)
        pmat.restrict_from_to(
            row_ids=mdn_neuropil[i], column_ids=list_motor_neurons[i], input_type="uid"
        )
        nmat = copy.deepcopy(neg_matrix)
        nmat.restrict_from_to(
            row_ids=mdn_neuropil[i], column_ids=list_motor_neurons[i], input_type="uid"
        )
        pos = pmat.get_matrix()
        neg = -1 * nmat.get_matrix()  # positive values
        # elementwise sign imbalance computation
        imbalance = np.abs(pos - neg) / (pos + neg)
        imbalance[np.isnan(imbalance)] = 0
        imbalance_metric.append(imbalance.sum() / imbalance.size)

    # Plot the imbalance metric
    _ = plots_design.scatter_xy(
        imbalance_metric,
        xlabel="MDN subdivision",
        ylabel="imbalance metric",
    )
    title = "Fig1_imbalance_metric_MDN"
    if mdn_side_ is not None:
        title += f"_MDN-{mdn_side_}"
    if mn_side_ is not None:
        title += f"_MN-{mn_side_}"
    plt.savefig(os.path.join(FOLDER, f"{title}.pdf"))
    plt.close()


if __name__ == "__main__":
    # venn_mdn_branches_neuropil_direct()
    # n_branches = 3
    # venn_t3_subbranches(n_clusters=n_branches)
    # confusion_matrix_mdn_to_mn(n_hops=2)
    # confusion_matrix_mdn_to_mn(n_hops=4)
    # confusion_matrix_sign_assymetry_mdn_to_mn(mn_side_='R', mdn_side_='R')
    # confusion_matrix_sign_assymetry_mdn_to_mn(mn_side_='R')
    sign_assymetry_mdn_to_mn(mn_side_="R", mdn_side_="R")
    pass

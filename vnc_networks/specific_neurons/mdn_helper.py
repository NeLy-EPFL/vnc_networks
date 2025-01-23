"""
Functions specific to working with MDNs to avoid copying code.
"""

import os
import typing
from typing import Optional

import matplotlib.pyplot as plt
import neuron
import params
from connections import Connections
from connectome_reader import ConnectomeReader
from neuron import Neuron
from params import BodyId

FOLDER_NAME = "MDN_specific"
FOLDER = os.path.join(params.FIG_DIR, FOLDER_NAME)
os.makedirs(FOLDER, exist_ok=True)


def get_mdn_bodyids(CR: ConnectomeReader = ConnectomeReader("MANC", "v1.0")):
    bids = CR.get_neuron_bodyids({"type": "MDN"})
    return bids


def get_mdn_uids(
    data: Connections,
    side: Optional[
        typing.Literal[
            "L", "Left", "l", "left", "LHS", "R", "Right", "r", "right", "RHS"
        ]
    ] = None,
):
    if side is None:
        return data.get_neuron_ids({"type": "MDN"})

    if side in ["L", "Left", "l", "left", "LHS"]:
        side_ = "L"
    elif side in ["R", "Right", "r", "right", "RHS"]:
        side_ = "R"
    else:
        raise ValueError("Side not recognized.")

    mdns = data.get_neuron_ids({"type": "MDN"})
    specific_mdns = [
        mdn
        for mdn in mdns
        if side_ == data.get_node_label(mdn)[-2]
        # names finishing with (L|R)
        # soma side not given for MDNs, but exists in the name
    ]
    return specific_mdns


def get_subdivided_mdns(
    VNC: Connections,
    neuropil: typing.Literal[
        "LegNp(T1)",
        "T1",
        "f",
        "fl",
        "LegNp(T2)",
        "T2",
        "m",
        "ml",
        "LegNp(T3)",
        "T3",
        "h",
        "hl",
    ],
    side: typing.Literal[
        "L", "Left", "l", "left", "LHS", "R", "Right", "r", "right", "RHS"
    ],
):
    """
    Get the uids of MDNs split by neuropil and side.
    neuropil format and side format are flexible to account for different
    naming conventions across the dataset.
    """
    if neuropil in ["LegNp(T1)", "T1", "f", "fl"]:
        neuropil_ = "LegNp(T1)"
    elif neuropil in ["LegNp(T2)", "T2", "m", "ml"]:
        neuropil_ = "LegNp(T2)"
    elif neuropil in ["LegNp(T3)", "T3", "h", "hl"]:
        neuropil_ = "LegNp(T3)"
    else:
        raise ValueError("Neuropil not recognized.")
    if side in ["L", "Left", "l", "left", "LHS"]:
        side_ = "L"
    elif side in ["R", "Right", "r", "right", "RHS"]:
        side_ = "R"

    mdns = VNC.get_neuron_ids({"type": "MDN"})
    specific_mdns = [
        mdn
        for mdn in mdns
        if (
            (neuropil_ in VNC.get_node_label(mdn))
            & (side_ == VNC.get_node_label(mdn)[-2])  # names finishing with (L|R)
        )  # soma side not given for MDNs, but exists in the name
    ]
    return specific_mdns


def get_vnc_split_MDNs_by_neuropil(
    not_connected: Optional[list[BodyId] | list[int]] = None,
):
    """
    Get the VNC Connections object with MDNs split by neuropil.
    """
    try:
        VNC = Connections(from_file="VNC_split_MDNs_by_neuropil")
        print("Loaded VNC Connections object with MDNs split by neuropil.")
    except FileNotFoundError:
        print("Creating VNC Connections object with MDNs split by neuropil...")
        MDNs = []
        for neuron_id in get_mdn_bodyids():
            neuron_name = neuron.split_neuron_by_neuropil(neuron_id)
            MDN = Neuron(from_file=neuron_name)
            MDNs.append(MDN)
        VNC = Connections(
            split_neurons=MDNs,
            not_connected=not_connected,
            )  # full VNC, split MDNs according to the synapse data
        VNC.save(name="VNC_split_MDNs_by_neuropil")
    return VNC


def mdn_synapse_distribution(
        n_clusters: int = 3,
        CR: ConnectomeReader = ConnectomeReader("MANC", "v1.0")
        ):
    """
    show for each MDN the distribution of synapses in the neuropils.
    Each row is an MDN.
    The first figure is depth-colour coded.
    The second in neuropil-colored.
    The third is clustering-colored, specific to T3.

    Save the data for later use.
    """
    mdn_bodyids = CR.get_neuron_bodyids({"type": "MDN"})
    for i in range(4):  # each MDN
        MDN = Neuron(mdn_bodyids[i], CR=CR)
        _ = MDN.get_synapse_distribution()

        # Column 1: depth-color coded
        MDN.plot_synapse_distribution(
            cmap=params.r_red_colorscale,
            savefig=False,
        )  # default is depth-color coded
        plt.tight_layout()
        plt.savefig(
            os.path.join(FOLDER, f"mdn_{mdn_bodyids[i]}_synapse_distribution_depth.pdf")
        )
        plt.close()

        # Column 2: neuropil-colored
        MDN.add_neuropil_information()
        MDN.plot_synapse_distribution(
            color_by="neuropil", discrete_coloring=True, savefig=False
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FOLDER, f"mdn_{mdn_bodyids[i]}_synapse_distribution_neuropil.pdf"
            )
        )
        plt.close()

        # Column 3: clustering-colored
        MDN.remove_defined_subdivisions()
        MDN.cluster_synapses_spatially(
            n_clusters=n_clusters, on_attribute={"neuropil": "T3"}
        )
        MDN.create_synapse_groups(attribute="KMeans_cluster")
        MDN.plot_synapse_distribution(
            color_by="KMeans_cluster", discrete_coloring=True, savefig=False
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FOLDER, f"mdn_{mdn_bodyids[i]}_synapse_distribution_clustering.pdf"
            )
        )
        plt.close()
        MDN.save(name=f"mdn_{mdn_bodyids[i]}_T3_synapses_split")
        del MDN


def get_connectome_with_MDN_t3_branches(
        n_clusters: int = 3,
        CR: ConnectomeReader = ConnectomeReader("MANC", "v1.0")
        ):
    """
    Build the connectome with the T3 neuropil branches of MDN split.
    """
    connectome_name = "VNC_split_MDNs_by_T3_synapses"
    try:
        VNC = Connections(from_file=connectome_name)
        print("Loaded the connectome with T3 branches of MDN split.")
    except FileNotFoundError:
        print("Creating the connectome with T3 branches of MDN split...")

        # === creating the split neurons
        mdn_bodyids = CR.get_neuron_bodyids({"type": "MDN"})
        MDNs = []
        # if not already defined (tested on the first), define the split neurons
        filename = os.path.join(
            params.NEURON_DIR, f"mdn_{mdn_bodyids[0]}_T3_synapses_split.txt"
        )
        if not os.path.isfile(filename):
            mdn_synapse_distribution(n_clusters=n_clusters, CR=CR)
        # load
        for i in range(4):
            MDN = Neuron(from_file=f"mdn_{mdn_bodyids[i]}_T3_synapses_split")
            MDNs.append(MDN)
        # === create the connections
        VNC_full = Connections(
            split_neurons=MDNs,  # split the MDNs according to the synapse data
            not_connected=mdn_bodyids,  # exclude connections from MDNs to MDNs
        )
        VNC_full.save(name=connectome_name)  # for later use
    return VNC_full

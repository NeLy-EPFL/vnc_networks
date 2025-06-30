#!/usr/bin/env python3
"""
Functions specific to working with MDNs to avoid copying code.
"""

import os
import typing
from typing import Optional

import matplotlib.pyplot as plt

from .. import neuron, params
from ..connections import Connections
from ..connectome_reader import MANC, ConnectomeReader
from ..neuron import Neuron
from ..params import BodyId

manc_version = "v1.2"  # used to be "v1.0"

FOLDER_NAME = "DN_preprocessing"
FIG_DIR = MANC(manc_version).get_fig_dir()
FOLDER = os.path.join(FIG_DIR, FOLDER_NAME)
os.makedirs(FOLDER, exist_ok=True)


def get_dn_bodyids(CR: ConnectomeReader = MANC(manc_version), name: str = "MDN"):
    bids = CR.get_neuron_bodyids({"type": name})
    return bids


def get_dn_uids(
    data: Connections,
    name: str = "MDN",
    side: Optional[
        typing.Literal[
            "L", "Left", "l", "left", "LHS", "R", "Right", "r", "right", "RHS"
        ]
    ] = None,
):
    if side is None:
        return data.get_neuron_ids({"type": name})

    if side in ["L", "Left", "l", "left", "LHS"]:
        side_ = "L"
    elif side in ["R", "Right", "r", "right", "RHS"]:
        side_ = "R"
    else:
        raise ValueError("Side not recognized.")

    dns = data.get_neuron_ids({"type": name})
    specific_dns = [
        dn
        for dn in dns
        if side_ == data.get_node_label(dn)[-2]
        # names finishing with (L|R)
        # soma side not given for MDNs, but exists in the name
    ]
    return specific_dns


def get_subdivided_dns(
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
    name: str = "MDN",
    side: Optional[
        typing.Literal[
            "L", "Left", "l", "left", "LHS", "R", "Right", "r", "right", "RHS"
        ]
    ] = None,
):
    """
    Get the uids of DNs split by neuropil and side.
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

    dns = VNC.get_neuron_ids({"type": name})
    if side is None:
        specific_dns = [dn for dn in dns if neuropil_ in VNC.get_node_label(dn)]
    else:
        specific_dns = [
            dn
            for dn in dns
            if (
                (neuropil_ in VNC.get_node_label(dn))
                & (side_ == VNC.get_node_label(dn)[-2])  # names finishing with (L|R)
            )  # soma side not given for MDNs, but exists in the name
        ]
    return specific_dns


def get_vnc_split_DNs_by_neuropil(
    not_connected: Optional[list[BodyId] | list[int]] = None,
    CR: ConnectomeReader = MANC(manc_version),
    name: str = "MDN",
):
    """
    Get the VNC Connections object with the specific DNs split by neuropil.
    """
    CR = MANC(manc_version)
    try:
        VNC = Connections(
            from_file=f"VNC_split_{name}s_by_neuropil",
            CR=CR,
        )
        print(f"Loaded VNC Connections object with {name}s split by neuropil.")
    except FileNotFoundError:
        print(f"Creating VNC Connections object with {name}s split by neuropil...")
        DNs = []
        for neuron_id in get_dn_bodyids(name=name):
            neuron_name = neuron.split_neuron_by_neuropil(neuron_id, CR=CR)
            DN = Neuron(from_file=neuron_name, CR=CR)
            DNs.append(DN)
        VNC = Connections(
            CR=CR,
            split_neurons=DNs,
            not_connected=not_connected,
        )  # full VNC, split MDNs according to the synapse data
        VNC.save(name=f"VNC_split_{name}s_by_neuropil")
    return VNC


def dn_synapse_distribution(
    n_clusters: int = 3, CR: ConnectomeReader = MANC(manc_version), name: str = "MDN"
):
    """
    show for each DN the distribution of synapses in the neuropils.
    Each row is a DN.
    The first figure is depth-colour coded.
    The second in neuropil-colored.
    The third is clustering-colored, specific to T3.

    Save the data for later use.
    """
    dn_bodyids = CR.get_neuron_bodyids({"type": name})
    for i in range(4):  # each MDN
        DN = Neuron(dn_bodyids[i], CR=CR)
        _ = DN.get_synapse_distribution()

        # Column 1: depth-color coded
        DN.plot_synapse_distribution(
            cmap=params.r_red_colorscale,
            savefig=False,
        )  # default is depth-color coded
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FOLDER, f"{name}_{dn_bodyids[i]}_synapse_distribution_depth.pdf"
            )
        )
        plt.close()

        # Column 2: neuropil-colored
        DN.add_neuropil_information()
        DN.plot_synapse_distribution(
            color_by="neuropil", discrete_coloring=True, savefig=False
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FOLDER, f"{name}_{dn_bodyids[i]}_synapse_distribution_neuropil.pdf"
            )
        )
        plt.close()

        # Column 3: clustering-colored
        DN.remove_defined_subdivisions()
        DN.cluster_synapses_spatially(
            n_clusters=n_clusters, on_attribute={"neuropil": "T3"}
        )
        DN.create_synapse_groups(attribute="KMeans_cluster")
        DN.plot_synapse_distribution(
            color_by="KMeans_cluster", discrete_coloring=True, savefig=False
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FOLDER, f"{name}_{dn_bodyids[i]}_synapse_distribution_clustering.pdf"
            )
        )
        plt.close()
        DN.save(name=f"{name}_{dn_bodyids[i]}_T3_synapses_split")
        del MD


def get_connectome_with_DN_t3_branches(
    n_clusters: int = 3, CR: ConnectomeReader = MANC(manc_version), name: str = "MDN"
):
    """
    Build the connectome with the T3 neuropil branches of the DN split.
    """
    connectome_name = f"VNC_split_{name}s_by_T3_synapses"
    try:
        VNC = Connections(from_file=connectome_name)
        print(f"Loaded the connectome with T3 branches of {name} split.")
    except FileNotFoundError:
        print(f"Creating the connectome with T3 branches of {name} split...")

        # === creating the split neurons
        dn_bodyids = CR.get_neuron_bodyids({"type": name})
        DNs = []
        # if not already defined (tested on the first), define the split neurons
        filename = os.path.join(
            params.NEURON_DIR, f"{name}_{dn_bodyids[0]}_T3_synapses_split.txt"
        )
        if not os.path.isfile(filename):
            dn_synapse_distribution(n_clusters=n_clusters, CR=CR, name=name)
        # load
        for i in range(4):
            DN = Neuron(from_file=f"{name}_{dn_bodyids[i]}_T3_synapses_split")
            DNs.append(DN)
        # === create the connections
        VNC_full = Connections(
            split_neurons=DNs,  # split the MDNs according to the synapse data
            not_connected=dn_bodyids,  # exclude connections from MDNs to MDNs
        )
        VNC_full.save(name=connectome_name)  # for later use
    return VNC_full

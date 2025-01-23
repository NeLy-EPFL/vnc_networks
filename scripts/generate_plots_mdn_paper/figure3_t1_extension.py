"""
Figure 3: T1 extension

Show that exciting MDN favors leg extension in T1.
"""

import copy
import os

import matplotlib.pyplot as plt
import mdn_paper_helper_functions as paper_funcs
import numpy as np
import params
import specific_neurons.mdn_helper as mdn_helper
import specific_neurons.motor_neurons_helper as mns_helper
from utils import matrix_design

FOLDER_NAME = "Figure_3_t1_extension"
FOLDER = os.path.join(params.FIG_DIR, FOLDER_NAME)
os.makedirs(FOLDER, exist_ok=True)

# -------------------------- Helper functions -------------------------- #

# -------------------------- Main functions -------------------------- #


def front_leg_muscles_graph(
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
        "class_1": "motor",
        "side": side_,
        "class_2": "fl",
        "target": muscle_,
    }
    title, axs = paper_funcs.graph_from_mdn_to_muscle(
        target,
        n_hops=n_hops,
        label_nodes=label_nodes,
    )
    plt.savefig(os.path.join(FOLDER, title + ".pdf"))
    plt.close()


if __name__ == "__main__":
    # front_leg_muscles_graph(muscle_ = 'Tr flexor')
    pass

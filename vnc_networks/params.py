#!/usr/bin/env python3
"""
2024.03.14
author: femke.hurtak@epfl.ch
File containing parameters for the project.
"""

import os
import typing
from pathlib import Path

import numpy as np
import seaborn as sns

# --- Where to find the raw data sets --- #
# Need to add the raw data dumps manually given the file sizes
# RAW_DATA_DIR = '/home/hurtak/multi_connectomes/vnc_networks/data_dump'
# tmp fix when using as an external package

RAW_DATA_DIR = os.path.join(Path(__file__).absolute().parent.parent, "data_dump")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# --- Where to save the preprocessed data sets --- #
# Used to store a preprocessed version of the data, to avoid re-running the
# preprocessing steps each time
PREPROCESSED_DATA_DIR = os.path.join(
    Path(__file__).absolute().parent.parent, "preprocessed"
)
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

# --- Where to save the processed data --- #
# Used to store the processed data, i.e. the data that is generated during
# the analysis
PROCESSED_DATA_DIR = os.path.join(
    Path(__file__).absolute().parent.parent, "data_processed"
)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- Where to save the figures --- #
# Used to store the figures generated during the analysis
FIG_DIR = os.path.join(Path(__file__).absolute().parent.parent, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --- Parameters for the analysis --- #
# nb: GLUT being inhibitory is still unclear, you can change it here before
# running the data preparation
SYNAPSE_CUTOFF = 5  # number of synapses to consider a connection

# --- Parameters for the visualization --- #
DARKBLUE = "#0173b2"  # colors from the colorblind friendly palette sns
LIGHTBLUE = "#56b4e9"
GREEN = "#029e73"
LIGHTORANGE = "#de8f05"
DARKORANGE = "#d55e00"

LIGHTGREY = "#c3c3c3"
DARKGREY = "#949494"
WHITE = "#FFFFFF"

GLUT_COLOR = "#C20C6D"
INHIB_COLOR = "#0f0b87"
EXCIT_COLOR = "#9c0d0b"

custom_palette = [
    DARKBLUE,
    GREEN,
    LIGHTORANGE,
    LIGHTBLUE,
    DARKGREY,
    DARKORANGE,
    LIGHTGREY,
]
colorblind_palette = sns.color_palette("colorblind")
blue_colorscale = (
    "crest"  # Perceptually uniform palettes, can be used in the categorical setting
)
r_blue_colorscale = "crest_r"
red_colorscale = "flare"
r_red_colorscale = "flare_r"
blue_heatmap = "mako_r"
red_heatmap = "rocket_r"
diverging_heatmap = "vlag"
grey_heatmap = "Greys"

# Figures
MAX_EDGE_WIDTH = 5
FIG_WIDTH = 8
FIG_HEIGHT = 8
FIGSIZE = (FIG_WIDTH, FIG_HEIGHT)
DPI = 300
NODE_SIZE = 100
TITLE_FONTSIZE = 12
FONT_SIZE = 5
FONT_COLOR = "black"
LABEL_SIZE = 16
AXIS_OFFSET = 2
LINEWIDTH = 2

# --- New types used in the code --- #
BodyId = typing.NewType("BodyId", int)  # always associated to a field 'body_id'
"""ID of a neuron body in the connectome. Ranges from 10000 to 841369768457 (MANC v1.0)"""

UID = typing.NewType("UID", int)
"""Unique ID of a neuron in the connections object. Note that this will not match a neuron's `BodyId`"""

NeuronAttribute = typing.Literal[
    "body_id",  # common to all
    "start_bid",
    "end_bid",
    "synapse_id",
    # connectivity
    "syn_count",
    "nb_pre_synapses",
    "nb_post_synapses",
    "nb_pre_neurons",
    "nb_post_neurons",
    "target",
    "nerve",
    "entry_nerve",
    "exit_nerve",
    # function
    "nt_type",  # common to all
    "nt_proba",  # common to all
    # classification
    "class_1",  # upper classification level, common to all
    "class_2",  # lower classification level, common to all
    "name",  # common to all
    "type",
    # morphology
    "side",  # common to all
    "neuropil",  # common to all
    "size",  # common to all
    "area",
    "length",
    "position",  # for neurons
    "location",  # for synapses
    # genetics
    "hemilineage",  # common to all
    # tracing related
    "tracing_status",
]
""" All the neuron attributes that can exit in any of the connectomes. 
Specific entries might not have a value for all attributes. Translations are
defined in the ConnectomeReader class for each instance of connectome. """

NeuronClass = typing.Literal[
    # common types
    "ascending",
    "descending",
    "motor",
    "sensory",
    # brain specific
    "central",
    "endocrine",
    "optic",
    "visual_centrifugal",
    "visual_projection",
    "other",
    # VNC specific
    "intrinsic",
    "glia",
    "sensory_ascending",
    "efferent",
    "efferent_ascending",
    "unknown",
    "sensory_unknown",
    "interneuron_unknown",
]
"""Possible values for neuron class. These are the 'class:string' values in MANC v1.0 or the 'superclass' values in the FAFB v630 connectome"""

SelectionDict = dict[NeuronAttribute, str | int | float | bool | BodyId]
"""Dictionary for selecting subsets of neurons based on different `NeuronAttribute`s."""


# --- Parameters for the network representation --- #
NT_TYPES = {
    "gaba": {"color": INHIB_COLOR, "linestyle": "-"},  # ":"
    "GABA": {"color": INHIB_COLOR, "linestyle": "-"},  # ":"
    "acetylcholine": {"color": EXCIT_COLOR, "linestyle": "-"},  # "--"
    "ACH": {"color": EXCIT_COLOR, "linestyle": "-"},  # "--"
    "glutamate": {"color": GLUT_COLOR, "linestyle": "-"},
    "GLUT": {"color": GLUT_COLOR, "linestyle": "-"},
    "DA": {"color": LIGHTGREY, "linestyle": "-"},
    "OCT": {"color": LIGHTGREY, "linestyle": "-"},
    "SER": {"color": LIGHTGREY, "linestyle": "-"},
    "unknown": {"color": LIGHTGREY, "linestyle": "-"},
    None: {"color": LIGHTGREY, "linestyle": "-"},
    np.nan: {"color": LIGHTGREY, "linestyle": "-"},
}

NEURON_CLASSES: dict[NeuronClass, dict[typing.Literal["color"], str]] = {
    # orange: input related
    "ascending": {"color": DARKORANGE},
    "sensory": {"color": LIGHTORANGE},
    "sensory ascending": {"color": LIGHTORANGE},
    "optic": {"color": LIGHTORANGE},
    "visual projection": {"color": LIGHTORANGE},
    # blue: output related
    "descending": {"color": DARKBLUE},
    "motor": {"color": LIGHTBLUE},
    "efferent": {"color": LIGHTBLUE},
    "efferent ascending": {"color": LIGHTBLUE},
    "visual centrifugal": {"color": LIGHTBLUE},
    # green: interneurons
    "intrinsic": {"color": GREEN},
    "central": {"color": GREEN},
    # grey: other
    "interneuron_unknown": {"color": DARKGREY},
    "sensory_unknown": {"color": DARKGREY},
    "endocrine": {"color": DARKGREY},
    "unknown": {"color": LIGHTGREY},
    "other": {"color": LIGHTGREY},
    "glia": {"color": LIGHTGREY},
}

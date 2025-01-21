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
# MANC traced
RAW_DATA_DIR = os.path.join(Path(__file__).absolute().parent.parent, "data_dump")

# --- Where to save the processed data sets --- #
PROCESSED_DATA_DIR = os.path.join(Path(__file__).absolute().parent.parent, "data")

# --- Where to save the preprocessed data sets --- #
PREPROCESSING_DIR = os.path.join(
    Path(__file__).absolute().parent.parent, "preprocessing"
)
NEURON_DIR = os.path.join(PREPROCESSING_DIR, "neurons")
os.makedirs(NEURON_DIR, exist_ok=True)
CONNECTION_DIR = os.path.join(PREPROCESSING_DIR, "connections")
os.makedirs(CONNECTION_DIR, exist_ok=True)

# --- Where to save the figures --- #
FIG_DIR = os.path.join(Path(__file__).absolute().parent.parent, "plots")
os.makedirs(FIG_DIR, exist_ok=True)
PLOT_DIR = os.path.join(FIG_DIR, "tmp")
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Parameters for the analysis --- #
NT_WEIGHTS = {
    "acetylcholine": +1,
    "gaba": -1,
    "glutamate": -1,
    "unknown": 0,
    None: 0,
    np.nan: 0,
}
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
FONT_SIZE = 5
FONT_COLOR = "black"
LABEL_SIZE = 16
AXIS_OFFSET = 2
LINEWIDTH = 2

# --- New types used in the code --- #
BodyId = typing.NewType("BodyId", int) # always associated to a field 'body_id'
"""ID of a neuron body in the connectome. Ranges from 10000 to 841369768457 (MANC v1.0)"""

UID = typing.NewType("UID", int)
"""Unique ID of a neuron in the connections object. Note that this will not match a neuron's `BodyId`"""

NeuronAttribute = typing.Literal[
    "body_id", # common to all
    # connectivity
    "nb_pre_synapses",
    "nb_post_synapses",
    "nb_pre_neurons",
    "nb_post_neurons",
    "target",
    "nerve",
    "entry_nerve",
    "exit_nerve",
    # function
    "nt_type", # common to all
    "nt_proba", # common to all
    # classification
    "class_1", # upper classification level, common to all
    "class_2", # lower classification level, common to all
    "name", # common to all
    # morphology
    "side", # common to all
    "neuropil", # common to all
    "size", # common to all
    "area",
    "length",
    "position",
    # genetics
    "hemilineage", # common to all
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
    "visual centrifugal",
    "visual projection",
    "other",
    # VNC specific
    "intrinsic",
    "glia",
    "sensory ascending",
    "efferent",
    "efferent ascending",
    "unknown",
    "sensory_unknown",
    "interneuron_unknown",
]

SelectionDict = dict[NeuronAttribute, str | int | float | bool | BodyId]


# --- Parameters for the network representation --- #
NT_TYPES = {
    "gaba": {"color": INHIB_COLOR, "linestyle": "-"},  # ":"
    "acetylcholine": {"color": EXCIT_COLOR, "linestyle": "-"},  # "--"
    "glutamate": {"color": GLUT_COLOR, "linestyle": "-"},
    "unknown": {"color": LIGHTGREY, "linestyle": "-"},
    None: {"color": LIGHTGREY, "linestyle": "-"},
    np.nan: {"color": LIGHTGREY, "linestyle": "-"},
}
'''
NEURON_CLASSES: dict[NeuronClass, dict[typing.Literal["color"], str]] = {
    "sensory neuron": {"color": LIGHTORANGE},
    "sensory ascending": {"color": LIGHTORANGE},
    "ascending neuron": {"color": DARKORANGE},
    "Sensory TBD": {"color": LIGHTORANGE},
    "motor neuron": {"color": LIGHTBLUE},
    "descending neuron": {"color": DARKBLUE},
    "efferent neuron": {"color": DARKGREY},
    "efferent ascending": {"color": DARKGREY},
    "intrinsic neuron": {"color": GREEN},
    "Interneuron TBD": {"color": DARKGREY},
    "Glia": {"color": LIGHTGREY},
    "TBD": {"color": LIGHTGREY},
}
'''

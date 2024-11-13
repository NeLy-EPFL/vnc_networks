"""
2024.03.14
author: femke.hurtak@epfl.ch
File containing parameters for the project.
"""
import os
from pathlib import Path
import numpy as np
import seaborn as sns

# --- Where to find the raw data sets --- #
# MANC traced
RAW_DATA_DIR = os.path.join(Path(__file__).absolute().parent.parent, "data_dump")
MANC_RAW_DIR = os.path.join(RAW_DATA_DIR, "manc", "v1.0", "manc-traced-adjacencies-v1.0")
NODES_FILE = os.path.join(MANC_RAW_DIR, "traced-neurons.csv")
CONNECTIONS_FILE = os.path.join(MANC_RAW_DIR, "traced-connections.csv")
# MANC neuprint
NEUPRINT_RAW_DIR = os.path.join(RAW_DATA_DIR, "manc", "v1.0", "neuprint_manc_v1.0", "neuprint_manc_v1.0_ftr")
NEUPRINT_NODES_FILE = os.path.join(NEUPRINT_RAW_DIR, "Neuprint_Neurons_manc_v1.ftr")
NEUPRINT_CONNECTIONS_FILE = os.path.join(NEUPRINT_RAW_DIR, "Neuprint_Neuron_Connections_manc_v1.ftr")
NEUPRINT_NEURON_SYNAPSESSET_FILE = os.path.join(NEUPRINT_RAW_DIR, "Neuprint_Neuron_to_SynapseSet_manc_v1.ftr")
NEUPRINT_SYNAPSSET_FILE = os.path.join(NEUPRINT_RAW_DIR, "Neuprint_SynapseSet_to_Synapses_manc_v1.ftr")
NEUPRINT_SYNAPSE_FILE = os.path.join(NEUPRINT_RAW_DIR, "Neuprint_Synapses_manc_v1.ftr")

# --- Where to save the processed data sets --- #
PROCESSED_DATA_DIR = os.path.join(Path(__file__).absolute().parent.parent, "data")

# --- Where to save the preprocessed data sets --- #
PREPROCESSING_DIR = os.path.join(Path(__file__).absolute().parent.parent, "preprocessing")
NEURON_DIR = os.path.join(PREPROCESSING_DIR, "neurons")
os.makedirs(NEURON_DIR, exist_ok=True)
CONNECTION_DIR = os.path.join(PREPROCESSING_DIR, "connections")
os.makedirs(CONNECTION_DIR, exist_ok=True)

# --- Where to save the figures --- #
FIG_DIR = os.path.join(Path(__file__).absolute().parent.parent, "plots")
os.makedirs(FIG_DIR, exist_ok=True)
PLOT_DIR = os.path.join(FIG_DIR,"tmp")
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Parameters for the analysis --- #
NT_WEIGHTS = {"acetylcholine": +1, "gaba": -1, "glutamate": -1, "unknown": 0, None: 0, np.nan: 0}
# nb: GLUT being inhibitory is still unclear, you can change it here before
# running the data preparation
SYNAPSE_CUTOFF = 5 # number of synapses to consider a connection

# --- Parameters for the visualization --- #
DARKBLUE = '#0173b2' # colors from the colorblind friendly palette sns
LIGHTBLUE = '#56b4e9'
GREEN = '#029e73'
LIGHTORANGE = '#de8f05'
DARKORANGE = '#d55e00'

LIGHTGREY = "#c3c3c3"
DARKGREY = '#949494'
WHITE = "#FFFFFF"

GLUT_COLOR = "#C20C6D"
INHIB_COLOR = "#0f0b87"
EXCIT_COLOR = "#9c0d0b"

custom_palette = [DARKBLUE, GREEN,LIGHTORANGE,LIGHTBLUE,DARKGREY,DARKORANGE,LIGHTGREY]
colorblind_palette = sns.color_palette("colorblind")
blue_colorscale = 'crest' # Perceptually uniform palettes, can be used in the categorical setting
r_blue_colorscale = 'crest_r'
red_colorscale = 'flare'
r_red_colorscale = 'flare_r'
blue_heatmap = 'mako_r'
red_heatmap = 'rocket_r'
diverging_heatmap = 'vlag'
grey_heatmap = 'Greys'

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

# --- Parameters for the network representation --- #
NT_TYPES = { 
    "gaba": {"color": INHIB_COLOR, "linestyle": "-"},  # ":"
    "acetylcholine": {"color": EXCIT_COLOR, "linestyle": "-"},  # "--"
    "glutamate": {"color": GLUT_COLOR, "linestyle": "-"},
    "unknown": {"color": LIGHTGREY, "linestyle": "-"},
    None: {"color": LIGHTGREY, "linestyle": "-"},
    np.nan: {"color": LIGHTGREY, "linestyle": "-"},
}
NEURON_CLASSES = {
    'sensory neuron': {"color": LIGHTORANGE},
    'sensory ascending': {"color": LIGHTORANGE},
    'ascending neuron': {"color": DARKORANGE},
    'Sensory TBD': {"color": LIGHTORANGE},
    'motor neuron': {"color":LIGHTBLUE},
    'descending neuron': {"color":DARKBLUE},
    'efferent neuron': {"color": DARKGREY},
    'efferent ascending': {"color": DARKGREY},
    'intrinsic neuron': {"color": GREEN},
    'Interneuron TBD': {"color": DARKGREY},
    'Glia': {"color": LIGHTGREY},
    'TBD': {"color": LIGHTGREY},
}



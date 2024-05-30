"""
2024.03.14
author: femke.hurtak@epfl.ch
File containing parameters for the project.
"""
import os
from pathlib import Path

# --- Where to find the raw data sets --- #
# MANC traced
RAW_DATA_DIR = os.path.join(Path(__file__).absolute().parent.parent, "data_dump")
MANC_RAW_DIR = os.path.join(RAW_DATA_DIR, "manc", "v1.0", "manc-traced-adjacencies-v1.0")
NODES_FILE = os.path.join(MANC_RAW_DIR, "traced-neurons.csv")
CONNECTIONS_FILE = os.path.join(MANC_RAW_DIR, "traced-connections-per-roi.csv")
# MANC neuprint
NEUPRINT_RAW_DIR = os.path.join(RAW_DATA_DIR, "manc", "v1.0", "neuprint_manc_v1.0", "neuprint_manc_v1.0_ftr")
NEUPRINT_NODES_FILE = os.path.join(NEUPRINT_RAW_DIR, "Neuprint_Neurons_manc_v1.ftr")
NEUPRINT_CONNECTIONS_FILE = os.path.join(NEUPRINT_RAW_DIR, "Neuprint_Neuron_Connections_manc_v1.ftr")

# --- Where to save the processed data sets --- #

# --- Where to save the figures --- #
PLOT_DIR = os.path.join(Path(__file__).absolute().parent.parent, "plots")

# --- Parameters for the analysis --- #
NT_WEIGHTS = {"acetylcholine": +1, "gaba": -1, "glutamate": -1, "unknown": 0, None: 0}
# nb: GLUT being inhibitory is still unclear, you can change it here before
# running the data preparation
SYNAPSE_CUTOFF = 5 # number of synapses to consider a connection

# --- Parameters for the visualization --- #



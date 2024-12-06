"""
2024.03.14
author: femke.hurtak@epfl.ch
File containing parameters for the project.
"""
import os
from pathlib import Path
import numpy as np
import seaborn as sns
import typing

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
NEUPRINT_NEURON_SYNAPSESET_FILE = os.path.join(NEUPRINT_RAW_DIR, "Neuprint_Neuron_to_SynapseSet_manc_v1.ftr")
NEUPRINT_SYNAPSESET_FILE = os.path.join(NEUPRINT_RAW_DIR, "Neuprint_SynapseSet_to_Synapses_manc_v1.ftr")
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
DPI = 75
NODE_SIZE = 100
FONT_SIZE = 5
FONT_COLOR = "black"
LABEL_SIZE = 16
AXIS_OFFSET = 2
LINEWIDTH = 2

# --- New types used in the code --- #
BodyId = typing.NewType('BodyId', int)
"""ID of a neuron body in the connectome. Ranges from 10000 to 841369768457 (MANC v1.0)"""

UID = typing.NewType('UID', int)
"""Unique ID of a neuron in the connections object. Note that this will not match a neuron's `BodyId`"""

NeuronAttribute = typing.Literal[":ID(Body-ID)", "bodyId:long", "pre:int", "post:int", "upstream:int", 
								 "downstream:int", "synweight:int", "status:string", "statusLabel:string", 
								 "cropped:boolean", "instance:string", "synonyms:string", "type:string", 
								 "systematicType:string", "hemilineage:string", "somaSide:string", "class:string", 
								 "subclass:string", "group:int", "serial:int", "rootSide:string", "entryNerve:string", 
								 "exitNerve:string", "position:point{srid:9157}", "somaNeuromere:string", 
								 "longTract:string", "birthtime:string", "cellBodyFiber:string", 
								 "somaLocation:point{srid:9157}", "rootLocation:point{srid:9157}", 
								 "tosomaLocation:point{srid:9157}", "size:long", "ntGabaProb:float", 
								 "ntAcetylcholineProb:float", "ntGlutamateProb:float", "ntUnknownProb:float", 
								 "predictedNtProb:float", "predictedNt:string", "origin:string", "target:string", 
								 "subcluster:int", "positionType:string", "tag:string", "modality:string", 
								 "serialMotif:string", "transmission:string", "roiInfo:string", ":LABEL", 
								 "ADMN(L):boolean", "ADMN(R):boolean", "Ov(L):boolean", "Ov(R):boolean", 
								 "ANm:boolean", "AbN1(L):boolean", "AbN1(R):boolean", "AbN2(L):boolean", 
								 "AbN2(R):boolean", "AbN3(L):boolean", "AbN3(R):boolean", "AbN4(L):boolean", 
								 "AbN4(R):boolean", "AbNT:boolean", "CV:boolean", "CvN(L):boolean", "CvN(R):boolean", 
								 "DMetaN(L):boolean", "DMetaN(R):boolean", "DProN(L):boolean", "DProN(R):boolean", 
								 "GF(L):boolean", "GF(R):boolean", "HTct(UTct-T3)(L):boolean", "HTct(UTct-T3)(R):boolean", 
								 "LegNp(T1)(L):boolean", "LegNp(T1)(R):boolean", "LegNp(T2)(L):boolean", 
								 "LegNp(T2)(R):boolean", "LegNp(T3)(L):boolean", "LegNp(T3)(R):boolean", "IntTct:boolean", 
								 "LTct:boolean", "MesoAN(L):boolean", "MesoAN(R):boolean", "MesoLN(L):boolean", 
								 "MesoLN(R):boolean", "MetaLN(L):boolean", "MetaLN(R):boolean", "NTct(UTct-T1)(L):boolean", 
								 "NTct(UTct-T1)(R):boolean", "PDMN(L):boolean", "PDMN(R):boolean", "PrN(L):boolean", 
								 "PrN(R):boolean", "ProCN(L):boolean", "ProCN(R):boolean", "ProAN(L):boolean", "ProAN(R):boolean", 
								 "ProLN(L):boolean", "ProLN(R):boolean", "VProN(L):boolean", "VProN(R):boolean", 
								 "WTct(UTct-T2)(L):boolean", "WTct(UTct-T2)(R):boolean", "mVAC(T1)(L):boolean", "mVAC(T1)(R):boolean", 
								 "mVAC(T2)(L):boolean", "mVAC(T2)(R):boolean", "mVAC(T3)(L):boolean", "mVAC(T3)(R):boolean"]
"""All attributes for a neuron in the MANC v1.0 data, in `"name:type"` format. Note that parameters of type `point{srid:9157}` will be loaded as strings (eg. `"{x:24406,y:11337,z:26827}"`)"""

NeuronClass = typing.Literal['sensory neuron', 'motor neuron', 'efferent neuron', 'sensory ascending', 'TBD', 
							 'intrinsic neuron', 'ascending neuron', 'descending neuron', 'Glia', 'Sensory TBD', 'Interneuron TBD', 'efferent ascending']
"""Possible values for neuron class (the `"class:string"` parameter) in the MANC v1.0 connectome."""


# --- Parameters for the network representation --- #
NT_TYPES = { 
    "gaba": {"color": INHIB_COLOR, "linestyle": "-"},  # ":"
    "acetylcholine": {"color": EXCIT_COLOR, "linestyle": "-"},  # "--"
    "glutamate": {"color": GLUT_COLOR, "linestyle": "-"},
    "unknown": {"color": LIGHTGREY, "linestyle": "-"},
    None: {"color": LIGHTGREY, "linestyle": "-"},
    np.nan: {"color": LIGHTGREY, "linestyle": "-"},
}
NEURON_CLASSES: dict[NeuronClass, dict[typing.Literal["color"], str]] = {
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
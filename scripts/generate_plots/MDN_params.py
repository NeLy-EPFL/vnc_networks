"""
Parameters for the MDN paper
"""

import os

from vnc_networks import MANC

# ----- Directories
MDN_DIR = "MDN_project"
FIG_DIR = MANC("v1.2").get_fig_dir()
MDN_FIGS = os.path.join(FIG_DIR, MDN_DIR)
os.makedirs(MDN_FIGS, exist_ok=True)

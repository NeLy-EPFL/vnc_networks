#!/usr/bin/env python3
"""
Helper functions regarding the ensemble of all neurons in the VNC.
"""

from typing import Optional

from ..connections import Connections
from ..params import BodyId


def get_full_vnc(not_connected: Optional[list[BodyId] | list[int]] = None):
    try:
        VNC = Connections(from_file="full_VNC")
        print("Loaded full VNC Connections object.")
    except FileNotFoundError:
        print("Creating full VNC Connections object...")
        VNC = Connections(not_connected=not_connected)
        VNC.save(name="full_VNC")
    return VNC

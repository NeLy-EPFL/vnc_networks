#!/usr/bin/env python3
"""
Helper functions regarding the sensory neurons in the VNC.
"""

import typing
from typing import Optional

from ..connections import Connections
from ..params import UID


def get_leg_sensory_neurons(
    data: Connections,
    leg: Optional[typing.Literal["f", "m", "h"]] = None,
    side: Optional[
        typing.Literal["LHS", "L", "left", "Left", "RHS", "R", "right", "Right"]
    ] = None,
) -> set[UID]:
    """
    Get the uids of leg sensory neurons.
    f: front leg
    m: middle leg
    h: hind leg
    side: LHS or RHS
    """
    if data.CR.connectome_name != "manc":
        print(data.CR.connectome_name)
        raise ValueError("This function is only implemented for MANC.")
    match leg:  # Sleection based on figure 5 panel C, https://elifesciences.org/reviewed-preprints/96084
        case "f":
            entry_nerves = ["ProLN", "ProAN", "VProN", "DProN"]
        case "m":
            entry_nerves = ["MesoLN"]
        case "h":
            entry_nerves = ["AbN1", "MetaLN"]
        case None:
            entry_nerves = [
                "ProLN",
                "ProAN",
                "VProN",
                "DProN",
                "MesoLN",
                "AbN1",
                "MetaLN",
            ]
    if side is not None:
        if side in ["LHS", "L", "left", "Left"]:
            side = "_L"
        elif side in ["RHS", "R", "right", "Right"]:
            side = "_R"
        else:
            raise ValueError("Side not recognized.")
        # duplicate all elements in entry_nerves by adding the side
        all_nerves = [en + side for en in entry_nerves]
        entry_nerves = all_nerves
    else:
        all_nerves = [en + "_L" for en in entry_nerves] + [
            en + "_R" for en in entry_nerves
        ]
        entry_nerves = all_nerves

    leg_sensory_neurons = []
    for en in entry_nerves:
        neurons_post = data.get_neuron_ids(
            {
                "entry_nerve": en,
                "class_1": "sensory",
            }
        )
        leg_sensory_neurons.extend(neurons_post)
    return set(leg_sensory_neurons)

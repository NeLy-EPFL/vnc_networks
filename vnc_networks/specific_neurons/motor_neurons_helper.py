#!/usr/bin/env python3
"""
Helper functions regarding the ensemble of all motor neurons in the VNC.
"""

import typing
from typing import Optional

from ..connections import Connections
from ..params import UID


def get_leg_motor_neurons(
    data: Connections,
    leg: Optional[typing.Literal["f", "m", "h"]] = None,
    side: Optional[typing.Literal["LHS", "RHS"]] = None,
) -> set[UID]:
    """
    Get the uids of leg motor neurons.
    f: front leg
    m: middle leg
    h: hind leg
    side: LHS or RHS
    """
    if side is not None:
        if side not in ["LHS", "RHS"]:
            raise ValueError("side must be either LHS or RHS")
    match leg:
        case "f":
            target = ["fl"]
        case "m":
            target = ["ml"]
        case "h":
            target = ["hl"]
        case None:
            target = ["fl", "ml", "hl"]
    leg_motor_neurons = []
    for t in target:
        if side is not None:
            neurons_post = data.get_neuron_ids(
                {
                    "class_2": t,
                    "class_1": "motor",
                    "side": side,
                }
            )
        else:
            neurons_post = data.get_neuron_ids(
                {"class_2": t, "class_1": "motor"}
            )
        leg_motor_neurons.extend(neurons_post)
    return set(leg_motor_neurons)


if __name__ == "__main__":
    VNC = Connections()
    leg_motor_neurons = get_leg_motor_neurons(VNC, leg="f", side="LHS")
    bids = VNC.get_bodyids_from_uids(leg_motor_neurons)
    print(bids)

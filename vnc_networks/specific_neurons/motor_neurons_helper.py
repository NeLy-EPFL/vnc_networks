"""
Helper functions regarding the ensemble of all motor neurons in the VNC.
"""

from connections import Connections
import typing
from typing import Optional

from params import UID


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
                    "subclass:string": t,
                    "class:string": "motor neuron",
                    "somaSide:string": side,
                }
            )
        else:
            neurons_post = data.get_neuron_ids(
                {"subclass:string": t, "class:string": "motor neuron"}
            )
        leg_motor_neurons.extend(neurons_post)
    return set(leg_motor_neurons)

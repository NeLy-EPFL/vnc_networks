"""
This module provides functions to get the data of the neurons from the dataset.
"""

import typing
from typing import Optional

import pandas as pd
import params
from params import BodyId, NeuronAttribute, NeuronClass, SelectionDict



def get_neuron_bodyids(selection_dict: SelectionDict) -> list[BodyId]:
    raise NotImplementedError("This function has moved to the ConnectomeReader class.")

def get_neurons_from_class(class_: NeuronClass) -> list[BodyId]:
    raise NotImplementedError("This function has moved to the ConnectomeReader class.")

def load_data_neuron(
    id_: BodyId | int, attributes: Optional[list[NeuronAttribute]] = None
) -> pd.DataFrame:
    raise NotImplementedError("This function has moved to the ConnectomeReader class.")


def load_data_neuron_set(
    ids: list[BodyId] | list[int], attributes: Optional[list[NeuronAttribute]] = None
) -> pd.DataFrame:
    raise NotImplementedError("This function has moved to the ConnectomeReader class.")


def get_possible_columns() -> list[str]:
    raise NotImplementedError("This function has moved to the ConnectomeReader class.")

'''
This module provides functions to get the data of the neurons from the dataset.
'''

from typing import Optional
import typing
import pandas as pd
import params
from params import BodyId, NeuronAttribute, NeuronClass

def get_neuron_bodyids(
        selection_dict: Optional[dict[NeuronAttribute, str | int | float | bool | BodyId]] = None,
        nodes: Optional[list[int]] = None
    ) -> list[BodyId]:
    """
    Get the Ids of the neurons in the dataset.
    Select (keep) according to the selection_dict.
    Different criteria are treated as 'and' conditions.
    """
    columns_to_read = {':ID(Body-ID)'}.union(selection_dict.keys()) if selection_dict is not None else {':ID(Body-ID)'}
    neurons = pd.read_feather(params.NEUPRINT_NODES_FILE, columns=list(columns_to_read))
    if selection_dict is not None:
        for key in selection_dict:
            neurons = neurons[neurons[key] == selection_dict[key]]
    if nodes is not None:
        neurons = neurons[neurons[':ID(Body-ID)'].isin(nodes)]
    return list(neurons[':ID(Body-ID)'].values)

def get_neurons_from_class(class_: NeuronClass) -> list[BodyId]:
    """
    Get the bodyids of neurons of a certain class.
    Existing classes are: 
    'sensory neuron', 'motor neuron', 'efferent neuron',
    'sensory ascending', 'TBD', 'intrinsic neuron', 'ascending neuron',
    'descending neuron', 'Glia', 'Sensory TBD', 'Interneuron TBD',
    'efferent ascending'
    """
    return get_neuron_bodyids({'class:string': class_})

def load_data_neuron(id_: BodyId | int, attributes: Optional[list[NeuronAttribute]] = None) -> pd.DataFrame:
    """
    Load the data of a neuron with a certain id.
    includes (not exhaustive): [predictedNt:string, predictedNtProb:float, 
    upstream:int, downstream:int, synweight:int, status:string, instance:string,
    type:string, systematicType:string, hemilineage:string, somaSide:string, 
    subclass:string, rootSide:string, entryNerve:string, exitNerve:string,
    size:long]


    Parameters
    ----------
    id : int
        The id of the neuron.

    Returns
    -------
    pandas.DataFrame
        The data of the neuron.
    """
    columns_to_read = {':ID(Body-ID)'}.union(attributes) if attributes is not None else {':ID(Body-ID)'}
    neurons = pd.read_feather(params.NEUPRINT_NODES_FILE, columns=list(columns_to_read))
    if attributes is not None:
        for att in attributes:
            if att not in neurons.columns:
                raise ValueError(f'The attribute {att} is not in the dataset.')
        if ':ID(Body-ID)' not in attributes:
            attributes.append(':ID(Body-ID)')
        return neurons[neurons[':ID(Body-ID)'] == id_][attributes]
    else:
        return neurons[neurons[':ID(Body-ID)'] == id_]

def load_data_neuron_set(ids: list[BodyId] | list[int], attributes: Optional[list[NeuronAttribute]] = None) -> pd.DataFrame:
    """
    Load the data of a set of neurons with certain ids.
    includes (not exhaustive): [predictedNt:string, predictedNtProb:float, 
    upstream:int, downstream:int, synweight:int, status:string, instance:string,
    somaNeuromere:string,type:string, systematicType:string, hemilineage:string,
    somaSide:string, subclass:string, rootSide:string, entryNerve:string,
    exitNerve:string, size:long]

    Parameters
    ----------
    ids : list
        The bodyids of the neurons.

    Returns
    -------
    pandas.DataFrame
        The data of the neurons.
    """
    columns_to_read = {':ID(Body-ID)'}.union(attributes) if attributes is not None else {':ID(Body-ID)'}
    neurons = pd.read_feather(params.NEUPRINT_NODES_FILE, columns=list(columns_to_read))
    if attributes is not None:
        # verify if all elements of 'attributes' are columns in the dataset
        for att in attributes:
            if att not in neurons.columns:
                raise ValueError(f'The attribute {att} is not in the dataset.')
        if ':ID(Body-ID)' not in attributes:
            attributes.append(':ID(Body-ID)')
        if len(ids) == 1:
            return neurons[neurons[':ID(Body-ID)'] == ids[0]][attributes]
        return neurons[neurons[':ID(Body-ID)'].isin(ids)][attributes]
    else:
        return neurons[neurons[':ID(Body-ID)'].isin(ids)]
    
def get_possible_columns() -> list[str]:
    """
    Get the possible columns of the dataset.
    """
    return list(typing.get_args(NeuronAttribute))
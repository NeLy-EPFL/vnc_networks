'''
This module provides functions to get the data of the neurons from the dataset.
'''

import pandas as pd
import params


def get_neuron_bodyids(
        selection_dict: dict = None,
        nodes: list = None
    ) -> list[int]:
    """
    Get the Ids of the neurons in the dataset.
    Select (keep) according to the selection_dict.
    Different criteria are treated as 'and' conditions.
    """
    neurons = pd.read_feather(params.NEUPRINT_NODES_FILE)
    if selection_dict is not None:
        for key in selection_dict:
            neurons = neurons[neurons[key] == selection_dict[key]]
    if nodes is not None:
        neurons = neurons[neurons[':ID(Body-ID)'].isin(nodes)]
    return neurons[':ID(Body-ID)'].values


def get_neurons_from_class(class_: str) -> list[int]:
    """
    Get the bodyids of neurons of a certain class.
    Existing classes are: 
    'sensory neuron', 'motor neuron', 'efferent neuron',
    'sensory ascending', 'TBD', 'intrinsic neuron', 'ascending neuron',
    'descending neuron', 'Glia', 'Sensory TBD', 'Interneuron TBD',
    'efferent ascending'
    """
    selection_dict = {'class:string': class_}
    return get_neuron_bodyids(selection_dict)


def load_data_neuron(id_: int, attributes: list = None) -> pd.DataFrame:
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
    neurons = pd.read_feather(params.NEUPRINT_NODES_FILE)
    if attributes is not None:
        if attributes not in neurons.columns:
            raise ValueError(
                f'The attribute {attributes} is not in the dataset.'
                )
        attributes.append(':ID(Body-ID)')
        return neurons[neurons[':ID(Body-ID)'] == id_][attributes]
    else:
        return neurons[neurons[':ID(Body-ID)'] == id_]


def load_data_neuron_set(ids: list, attributes: list = None) -> pd.DataFrame:
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
    neurons = pd.read_feather(params.NEUPRINT_NODES_FILE)
    if attributes is not None:
        # verify if all elements of 'attributes' are columns in the dataset
        for att in attributes:
            if att not in neurons.columns:
                raise ValueError(f'The attribute {att} is not in the dataset.')
        attributes.append(':ID(Body-ID)')
        if len(ids) == 1:
            return neurons[neurons[':ID(Body-ID)'] == ids[0]][attributes]
        return neurons[neurons[':ID(Body-ID)'].isin(ids)][attributes]
    else:
        return neurons[neurons[':ID(Body-ID)'].isin(ids)]


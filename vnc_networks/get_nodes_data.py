'''
This module provides functions to get the data of the neurons from the dataset.
'''

import pandas as pd
import params

def get_neuron_ids(selection_dict:dict=None) -> list[int]:
    """
    Get the Ids of the neurons in the dataset.
    Select (keep) according to the selection_dict.
    """
    neurons = pd.read_feather(params.NEUPRINT_NODES_FILE)
    if selection_dict is not None:
        for key in selection_dict:
            neurons = neurons[neurons[key] == selection_dict[key]]
    return neurons['bodyId:long'].values

def get_neurons_from_class(class_:str) -> list[int]:
    """
    Get the neurons of a certain class.
    Existing classes are: 
    'sensory neuron', 'motor neuron', 'efferent neuron',
    'sensory ascending', 'TBD', 'intrinsic neuron', 'ascending neuron',
    'descending neuron', 'Glia', 'Sensory TBD', 'Interneuron TBD',
    'efferent ascending'
    """
    selection_dict = {'class:string': class_}
    return get_neuron_ids(selection_dict)

def load_data_neuron(id:int, attributes:list=None) -> pd.DataFrame:
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
        attributes.append(':ID(Body-ID)')
        return neurons[neurons[':ID(Body-ID)'] == id][attributes]
    return neurons[neurons[':ID(Body-ID)'] == id]

def load_data_neuron_set(ids:list, attributes:list=None) -> pd.DataFrame:
    """
    Load the data of a set of neurons with certain ids.
    includes (not exhaustive): [predictedNt:string, predictedNtProb:float, 
    upstream:int, downstream:int, synweight:int, status:string, instance:string,
    type:string, systematicType:string, hemilineage:string, somaSide:string, 
    subclass:string, rootSide:string, entryNerve:string, exitNerve:string,
    size:long]

    Parameters
    ----------
    ids : list
        The ids of the neurons.

    Returns
    -------
    pandas.DataFrame
        The data of the neurons.
    """
    neurons = pd.read_feather(params.NEUPRINT_NODES_FILE)
    if attributes is not None:
        attributes.append(':ID(Body-ID)')
        return neurons[neurons[':ID(Body-ID)'].isin(ids)][attributes]
    return neurons[neurons[':ID(Body-ID)'].isin(ids)]

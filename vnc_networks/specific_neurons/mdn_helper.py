'''
Functions specific to working with MDNs to avoid copying code.
'''
import neuron
from neuron import Neuron
from connections import Connections
import get_nodes_data

def get_mdn_bodyids():
    return get_nodes_data.get_neuron_bodyids({'type:string': 'MDN'})

def get_subdivided_mdns(VNC, neuropil, side):
    '''
    Get the uids of MDNs split by neuropil and side.
    neuropil format and side format are flexible to account for different
    naming conventions accross the dataset.
    '''
    if neuropil in ['LegNp(T1)', 'T1', 'f', 'fl']:
        neuropil_ = 'LegNp(T1)'
    elif neuropil in ['LegNp(T2)', 'T2', 'm', 'ml']:
        neuropil_ = 'LegNp(T2)'
    elif neuropil in ['LegNp(T3)', 'T3', 'h', 'hl']:
        neuropil_ = 'LegNp(T3)'
    else:
        raise ValueError('Neuropil not recognized.')
    if side in ['L', 'Left', 'l', 'left', 'LHS']:
        side_ = 'L'
    elif side in ['R', 'Right', 'r', 'right', 'RHS']:
        side_ = 'R'

    mdns = VNC.get_neuron_ids({'type:string': 'MDN'})
    specific_mdns = [
            mdn for mdn in mdns if (
                (neuropil_ in VNC.get_node_label(mdn))
                & (side_ == VNC.get_node_label(mdn)[-2]) # names finishing with (L|R)
                ) # soma side not given for MDNs, but exists in the name
        ]
    return specific_mdns

def get_vnc_split_MDNs_by_neuropil(not_connected: list[int] = None):
    '''
    Get the VNC Connections object with MDNs split by neuropil.
    '''
    try:
        VNC = Connections(from_file='VNC_split_MDNs_by_neuropil')
        print('Loaded VNC Connections object with MDNs split by neuropil.')
    except:
        print('Creating VNC Connections object with MDNs split by neuropil...')
        MDNs = []
        for neuron_id in get_mdn_bodyids():
            neuron_name = neuron.split_neuron_by_neuropil(neuron_id)
            MDN = Neuron(from_file=neuron_name)
            MDNs.append(MDN)
        VNC = Connections()  # full VNC
        VNC.initialize(
            split_neurons=MDNs,
            not_connected=not_connected,
            )  # split MDNs according to the synapse data
        VNC.save(name='VNC_split_MDNs_by_neuropil')
    return VNC
'''
Helper functions regarding the ensemble of all motor neurons in the VNC.
'''
from connections import Connections

def get_leg_motor_neurons(data: Connections, leg: str = None):
    '''
    Get the uids of leg motor neurons.
    f: front leg
    m: middle leg
    h: hind leg
    '''
    match leg:
        case 'f':
            target = ['fl']
        case 'm':
            target = ['ml']
        case 'h':
            target = ['hl']
        case None:
            target = ['fl', 'ml', 'hl']
    leg_motor_neurons = []
    for t in target:
        selection_dict = {
            'subclass:string': t,
            'class:string': 'motor neuron'
            }
        neurons_post = data.get_neuron_ids(selection_dict) # uids
        leg_motor_neurons.extend(neurons_post)
    return set(leg_motor_neurons)
'''
Helper functions regarding the ensemble of all motor neurons in the VNC.
'''
from connections import Connections

def get_leg_motor_neurons(
    data: Connections,
    leg: str = None,
    side: str = None):
    '''
    Get the uids of leg motor neurons.
    f: front leg
    m: middle leg
    h: hind leg
    side: LHS or RHS
    '''
    if side is not None:
        if side not in ['LHS', 'RHS']:
            raise ValueError('side must be either LHS or RHS')
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
        if side is not None:
            selection_dict = {
                'subclass:string': t,
                'class:string': 'motor neuron',
                'somaSide:string': side
                }
        else:
            selection_dict = {
                'subclass:string': t,
                'class:string': 'motor neuron'
                }
        neurons_post = data.get_neuron_ids(selection_dict) # uids
        leg_motor_neurons.extend(neurons_post)
    return set(leg_motor_neurons)
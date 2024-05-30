'''
Initialises the Neuron class.
Meant to be used to load data for a single neuron.
Possible use cases are visualisation, or looking at synapse distributions.
'''
from  vnc_networks.get_nodes_data import load_data_neuron

class Neuron:
    def __init__(self, bodyId: int):
        self.bodyId = bodyId
        self.data = load_data_neuron(bodyId)
        self.__initialise_base_attributes()

    # private methods
    def __initialise_base_attributes(self):
        self.type = self.data['systematicType:string'].values[0]
        self.hemilineage = self.data['hemilineage:string'].values[0]
        self.soma_side = self.data['somaSide:string'].values[0]
        self.class_ = self.data['class:string'].values[0]
        self.subclass = self.data['subclass:string'].values[0]
        self.group = self.data['group:int'].values[0]
        self.cell_body_fiber = self.data['cellBodyFiber:string'].values[0]
        self.size = self.data['size:long'].values[0]
        self.target = self.data['target:string'].values[0]
        self.predicted_nt_prob = self.data['predictedNtProb:float'].values[0]
        self.predicted_nt = self.data['predictedNt:string'].values[0]
        self.tag = self.data['tag:string'].values[0]
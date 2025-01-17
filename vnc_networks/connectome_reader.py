"""
Definition of the ConnectomeReader class.
Class defining the methods and labels to read different connectomes.
"""

import os
import typing
from typing import Optional

import pandas as pd
import params
from params import BodyId


# --- Parent class --- #
class ConnectomeReader:
    def __new__(
            cls,
            connectome_name: typing.Literal['MANCv1.0','FAFBv630'],
        ):
        if cls is ConnectomeReader:  # Only redirect if creating an instance directly
            match connectome_name:
                case 'MANCv1.0':
                    return MANC_v_1_0('MANCv1.0')
                case 'FAFBv630':
                    return FAFB_v_630('FAFBv630')
                case _:
                    raise ValueError("Connectome not recognized.")
        return super().__new__(cls)

    def __init__(
            self,
            connectome_name: typing.Literal['MANCv1.0','FAFBv630'],
            ):

        self.connectome_name = connectome_name
        self.raw_data_dir = params.RAW_DATA_DIR
        # specific namefields
        self._load_specific_namefields()

        self._load_specific_directories()
        # assert all is defined, defined globally
        self.__assert_all_is_defined()
        self.__define_data_types()

    # ----- virtual methods -----
    def _load_specific_namefields(self):
        raise NotImplementedError('This should only be called on child instances.')
    
    def _load_specific_directories(self):
        raise NotImplementedError('This should only be called on child instances.')


    # ----- private methods -----
    def __assert_all_is_defined(self):
        """
        Verify that all the base elements are defined.
        """
        try:
            _ = self.node_base_attributes()
        except AttributeError:
            raise AttributeError("Some base attributes are not defined.")
        return
        
    def __define_data_types(self):
        """
        Define the data types of the attributes.
        """
        self.SelectionDict = dict[self.NeuronAttribute, str | int | float | bool | BodyId]
        #Dictionary for selecting subsets of neurons based on different `NeuronAttribute`s.


    # ----- public methods -----
    def exists_tracing_status(self):
        """
        Identify if 'tracing_status' is a field in the connectome
        """
        if hasattr(self, 'tracing_status'):
            return True
        return False
    
    def node_base_attributes(self):
        """
        Returns a list of the base attributes that all nodes have, used
        to initialise a neuron::Neuron() object for instance.
        """
        list_node_attributes: list[self.NeuronAttribute] = [
                self.nt_type,
                self.nt_proba,
                self.class_1,
                self.class_2,
                self.name,
                self.side,
                self.neuropil,
                self.nerve,
                self.hemilineage,
                self.size,
            ]
        return list_node_attributes
    
    def get_neuron_bodyids(
        self,
        selection_dict: Optional[dict] = None,
        nodes: Optional[list[BodyId] | list[int]] = None,
    ) -> list[BodyId]:
        """
        Get the Ids of the neurons in the dataset.
        Select (keep) according to the selection_dict.
        Different criteria are treated as 'and' conditions.
        """
        columns_to_read = (
            {self.BodyId}.union(selection_dict.keys())
            if selection_dict is not None
            else {self.BodyId}
        )
        # verify if the naming is fine: all elements of 'columns_to_read' are NeuronAttributes
        if not all([col in self.NeuronAttribute for col in columns_to_read]):
            raise ValueError(
                "ConnectomeReader::get_neuron_bodyids() \
                Some attributes of the selction_dict are not in the dataset."
                )
        
        neurons = pd.read_feather(self.nodes_file, columns=list(columns_to_read))
        if selection_dict is not None:
            for key in selection_dict:
                neurons = neurons[neurons[key] == selection_dict[key]]
        if nodes is not None:
            neurons = neurons[neurons[self.BodyId].isin(nodes)]
        return list(neurons[self.BodyId].values)

    def get_neurons_from_class(self, class_: str) -> list[BodyId]:
        """
        Get the bodyids of neurons of a certain class (e.g. sensory).
        """
        # verify if the class is indeed a neuron class for this dataset
        if class_ not in self.NeuronClass:
            raise ValueError(
                f"ConnectomeReader::get_neurons_from_class().\
                The neuron class [{class_}] is not in the dataset."
                )
        return self.get_neuron_bodyids({self.class_1: class_})

    def load_data_neuron(
        self, id_: BodyId | int, attributes: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Load the data of a neuron with a certain id.
        
        Parameters
        ----------
        id : BodyId | int
            The id of the neuron.

        Returns
        -------
        pandas.DataFrame
            The data of the neuron.
        """
        # Verify naming
        if not all([col in self.NeuronAttribute for col in attributes]):
            raise ValueError(
                "ConnectomeReader::load_data_neuron() \
                Some attributes are not in the dataset."
                )
        
        # Load data
        columns_to_read = list(
            {self.BodyId}.union(attributes)
            if attributes is not None
            else {self.BodyId}
        )
        neurons = pd.read_feather(self.nodes_file, columns=columns_to_read)
        if attributes is not None:
            return neurons[neurons[self.BodyId] == id_][columns_to_read]
        else:
            return neurons[neurons[self.BodyId] == id_]

    def load_data_neuron_set(
        self, ids: list[BodyId] | list[int], attributes: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Load the data of a set of neurons with certain ids.

        Parameters
        ----------
        ids : list
            The bodyids of the neurons.

        Returns
        -------
        pandas.DataFrame
            The data of the neurons.
        """
        # Verify naming
        if not all([col in self.NeuronAttribute for col in attributes]):
            raise ValueError(
                "ConnectomeReader::load_data_neuron() \
                Some attributes are not in the dataset."
                )
        
        # Load data
        columns_to_read = list(
            {self.BodyId}.union(attributes)
            if attributes is not None
            else {self.BodyId}
        )
        neurons = pd.read_feather(self.nodes_file, columns=columns_to_read)
        
        if attributes is not None:
            return neurons[neurons[self.BodyId].isin(ids)][columns_to_read]
        return neurons[neurons[self.BodyId].isin(ids)]

    def get_possible_columns(self) -> list[str]:
        """
        Get the possible columns of the dataset.
        """
        return list(typing.get_args(self.NeuronAttribute))

# --- Specific classes --- #
class MANC_v_1_0(ConnectomeReader):
    def __init__(self, connectome_name): # 2nd argument is useless, only for compatibility
        super().__init__('MANCv1.0')
        
    # ----- overwritten methods -----
    # --- additions
    def __define_data_types(self):
        """
        Add the specific typing for Neuron classes.
        """
        super().__define_data_types()
        self.NeuronClass = typing.Literal[
            "sensory neuron",
            "motor neuron",
            "efferent neuron",
            "sensory ascending",
            "TBD",
            "intrinsic neuron",
            "ascending neuron",
            "descending neuron",
            "Glia",
            "Sensory TBD",
            "Interneuron TBD",
            "efferent ascending",
        ]

    # --- replacements
    def _load_specific_namefields(self):
        """
        Need to define the fields that are common to all connectomes.
        BodyId, start_bid, end_bid, syn_count, nt_type, class_1, class_2, neuron_attributes
        """

        self.BodyId = ":ID(Body-ID)"
        self.start_bid = "START_ID(Body-ID)"
        self.end_bid = "END_ID(Body-ID)"
        self.syn_count = "weightHR:int"
        self.nt_type = "predictedNt:string"
        self.nt_proba = "predictedNtProb:float"
        self.class_1 = "class:string"
        self.class_2 = "subclass:string"
        self.name = "systematicType:string"
        self.tracing_status = "status:string"
        self.side = "somaSide:string"
        self.neuropil = "somaNeuromere:string"
        self.nerve = "exitNerve:string"
        self.hemilineage = "hemilineage:string"
        self.size = "size:long"

        self.NeuronAttribute = typing.Literal[
            ":ID(Body-ID)",
            "bodyId:long",
            "pre:int",
            "post:int",
            "upstream:int",
            "downstream:int",
            "synweight:int",
            "status:string",
            "statusLabel:string",
            "cropped:boolean",
            "instance:string",
            "synonyms:string",
            "type:string",
            "systematicType:string",
            "hemilineage:string",
            "somaSide:string",
            "class:string",
            "subclass:string",
            "group:int",
            "serial:int",
            "rootSide:string",
            "entryNerve:string",
            "exitNerve:string",
            "position:point{srid:9157}",
            "somaNeuromere:string",
            "longTract:string",
            "birthtime:string",
            "cellBodyFiber:string",
            "somaLocation:point{srid:9157}",
            "rootLocation:point{srid:9157}",
            "tosomaLocation:point{srid:9157}",
            "size:long",
            "ntGabaProb:float",
            "ntAcetylcholineProb:float",
            "ntGlutamateProb:float",
            "ntUnknownProb:float",
            "predictedNtProb:float",
            "predictedNt:string",
            "origin:string",
            "target:string",
            "subcluster:int",
            "positionType:string",
            "tag:string",
            "modality:string",
            "serialMotif:string",
            "transmission:string",
            "roiInfo:string",
            ":LABEL",
            "ADMN(L):boolean",
            "ADMN(R):boolean",
            "Ov(L):boolean",
            "Ov(R):boolean",
            "ANm:boolean",
            "AbN1(L):boolean",
            "AbN1(R):boolean",
            "AbN2(L):boolean",
            "AbN2(R):boolean",
            "AbN3(L):boolean",
            "AbN3(R):boolean",
            "AbN4(L):boolean",
            "AbN4(R):boolean",
            "AbNT:boolean",
            "CV:boolean",
            "CvN(L):boolean",
            "CvN(R):boolean",
            "DMetaN(L):boolean",
            "DMetaN(R):boolean",
            "DProN(L):boolean",
            "DProN(R):boolean",
            "GF(L):boolean",
            "GF(R):boolean",
            "HTct(UTct-T3)(L):boolean",
            "HTct(UTct-T3)(R):boolean",
            "LegNp(T1)(L):boolean",
            "LegNp(T1)(R):boolean",
            "LegNp(T2)(L):boolean",
            "LegNp(T2)(R):boolean",
            "LegNp(T3)(L):boolean",
            "LegNp(T3)(R):boolean",
            "IntTct:boolean",
            "LTct:boolean",
            "MesoAN(L):boolean",
            "MesoAN(R):boolean",
            "MesoLN(L):boolean",
            "MesoLN(R):boolean",
            "MetaLN(L):boolean",
            "MetaLN(R):boolean",
            "NTct(UTct-T1)(L):boolean",
            "NTct(UTct-T1)(R):boolean",
            "PDMN(L):boolean",
            "PDMN(R):boolean",
            "PrN(L):boolean",
            "PrN(R):boolean",
            "ProCN(L):boolean",
            "ProCN(R):boolean",
            "ProAN(L):boolean",
            "ProAN(R):boolean",
            "ProLN(L):boolean",
            "ProLN(R):boolean",
            "VProN(L):boolean",
            "VProN(R):boolean",
            "WTct(UTct-T2)(L):boolean",
            "WTct(UTct-T2)(R):boolean",
            "mVAC(T1)(L):boolean",
            "mVAC(T1)(R):boolean",
            "mVAC(T2)(L):boolean",
            "mVAC(T2)(R):boolean",
            "mVAC(T3)(L):boolean",
            "mVAC(T3)(R):boolean",
        ]
            
    def _load_specific_directories(self):
        """
        Need to define the directories that are common to all connectomes.
        """
        
        self.connectome_dir = os.path.join(
            self.raw_data_dir,
            "manc",
            "v1.0",
            "neuprint_manc_v1.0",
            "neuprint_manc_v1.0_ftr",
        )
        self.nodes_file = os.path.join(
            self.raw_data_dir,
            "Neuprint_Neurons_manc_v1.ftr"
            )
        self.connections_file = os.path.join(
            self.connectome_dir,
            "Neuprint_Neuron_Connections_manc_v1.ftr"
            )



class FAFB_v_630(ConnectomeReader):
    def __init__(self, connectome_name):
        super().__init__('FAFBv630') # 2nd argument is useless, only for compatibility
        
    # ----- overwritten methods -----
    # --- additions
    def __define_data_types(self):
        """
        Add the specific typing for Neuron classes.
        """
        super().__define_data_types()
        self.NeuronClass = typing.Literal[ # TODO: fill
            "",
        ]
        raise NotImplementedError("Define the NeuronClass for FAFBv630.")
    
    # --- replacements
    def _load_specific_namefields(self):
        """
        Need to define the fields that are common to all connectomes.
        BodyId, start_bid, end_bid, syn_count, nt_type, class_1, class_2, neuron_attributes
        """
        
        self.BodyId = "root_id"
        self.start_bid = "pre_root_id"
        self.end_bid = "post_root_id"
        self.syn_count = "syn_count"
        self.nt_type = "nt_type"
        self.nt_proba = "nt_type_score"
        self.class_1 = "super_class"
        self.class_2 = "class"
        self.side = "side"
        self.neuropil = "cell_type" # TODO: check, probably wrong
        self.nerve = "nerve"
        self.hemilineage = "hemilineage"
        self.size = "size_nm"

        self.NeuronAttribute = typing.Literal[
            "group",
            "nt_type",
            "nt_type_score",
            "morphology_cluster",
            "connectivity_cluster",
            "flow",
            "super_class",
            "class",
            "sub_class",
            "cell_type",
            "hemibrain_type",
            "hemilineage",
            "side",
            "nerve",
            "length_nm",
            "area_nm",
            "size_nm",
        ]
            
    def _load_specific_directories(self):
        """
        Need to define the directories that are common to all connectomes.
        """
        
        self.connectome_dir = os.path.join(
            self.raw_data_dir,
            "",
        )
        self.nodes_file = os.path.join(
            self.raw_data_dir,
            ""
            )
        self.connections_file = os.path.join(
            self.connectome_dir,
            ""
            )
        


if __name__ == "__main__":
    # Test the class
    manc = ConnectomeReader('MANCv1.0')
    if manc.exists_tracing_status():
        print("MANC has tracing status.")
                
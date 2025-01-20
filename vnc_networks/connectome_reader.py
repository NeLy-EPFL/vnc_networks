"""
Definition of the ConnectomeReader class.
Class defining the methods and labels to read different connectomes.

Each instance also redefines the generic NeuronAttribute and NeuronClass
data types to the specific ones of the connectome.
"""

import os
import typing
from typing import Optional

import pandas as pd
import params
from params import BodyId, NeuronAttribute, NeuronClass, SelectionDict


# --- Parent class --- #
class ConnectomeReader:
    def __new__(
            cls,
            connectome_name: typing.Literal['MANC','FAFB'],
            connectome_version: typing.Literal['v1.0', 'v1.2','v630','v783'],
        ):
        if cls is ConnectomeReader:  # Only redirect if creating an instance directly
            match connectome_name:
                case 'MANC':
                    return MANC('MANC', connectome_version)
                case 'FAFB':
                    return FAFB('FAFB', connectome_version)
                case _:
                    raise ValueError("Connectome not recognized.")
        return super().__new__(cls)

    def __init__(
            self,
            connectome_name: typing.Literal['MANC','FAFB'],
            connectome_version: typing.Literal['v1.0', 'v1.2','v630','v783'],
            ):

        self.connectome_name = connectome_name
        self.connectome_version = connectome_version
        self.raw_data_dir = params.RAW_DATA_DIR
        # specific namefields
        self._load_specific_namefields()
        self._load_specific_neuron_classes()
        self._load_specific_directories()
        # assert all is defined, defined globally
        self.__assert_all_is_defined()
        self.__define_data_types()

    # ----- virtual private methods -----
    def _load_specific_namefields(self):
        raise NotImplementedError('This should only be called on child instances.')
    
    def _load_specific_neuron_classes(self):
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
        self.SpecificSelectionDict = dict[
            self.SpecificNeuronAttribute,
            str | int | float | bool | BodyId
            ]
        #Dictionary for selecting subsets of neurons based on different `NeuronAttribute`s.

    # ----- public methods -----
    def exists_tracing_status(self):
        """
        Identify if '_tracing_status' is a field in the connectome
        """
        if hasattr(self, '_tracing_status'):
            self.traced_entry = "Traced"
            return True
        return False
    
    def node_base_attributes(self):
        """
        Returns a list of the base attributes that all nodes have, used
        to initialise a neuron::Neuron() object for instance.
        """
        list_node_attributes: list[self.SpecificNeuronAttribute] = [
                self.BodyId,
                # function
                self.nt_type,
                self._nt_proba,
                # classification
                self._class_1,
                self._class_2,
                self._name,
                # morphology
                self._side,
                self._neuropil,
                self._size,
                # genetics
                self._hemilineage,
            ]
        return list_node_attributes
    
    def get_neuron_bodyids(
        self,
        selection_dict: Optional[SelectionDict] = None,
        nodes: Optional[list[BodyId] | list[int]] = None,
    ) -> list[BodyId]:
        """
        Get the Ids of the neurons in the dataset.
        Select (keep) according to the selection_dict.
        Different criteria are treated as 'and' conditions.
        """
        s_dict = self.specific_selection_dict(selection_dict)
        columns_to_read = (
            {self.BodyId}.union(s_dict.keys())
            if s_dict is not None
            else {self.BodyId}
        )
        
        neurons = pd.read_feather(self.nodes_file, columns=list(columns_to_read))
        if s_dict is not None:
            for key in s_dict:
                neurons = neurons[neurons[key] == s_dict[key]]
        if nodes is not None:
            neurons = neurons[neurons[self.BodyId].isin(nodes)]
        return list(neurons[self.BodyId].values)

    def get_neurons_from_class(self, class_: NeuronClass) -> list[BodyId]:
        """
        Get the bodyids of neurons of a certain class (e.g. sensory).
        """
        # verify if the class is indeed a neuron class for this dataset
        specific_class = self.specific_neuron_class(class_)
        return self.get_neuron_bodyids({self.class_1: specific_class})

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
        self,
        ids: list[BodyId] | list[int],
        attributes: Optional[list[NeuronAttribute]] = None,
    ) -> pd.DataFrame:
        """
        Load the data of a set of neurons with certain ids.

        Parameters
        ----------
        ids : list
            The bodyids of the neurons.
        attributes : list
            The attributes to load.

        Returns
        -------
        pandas.DataFrame
            The data of the neurons.
        """
        # Identify the names of columns to read
        columns_to_read = [
            self.specific_neuron_attribute(a) for a in attributes
            ]
        columns_to_write = attributes
        if self.BodyId not in columns_to_read:
            columns_to_read.append(self.BodyId)
            columns_to_write.append(self.BodyId) # to keep the order, and only field that keeps its name
        
        # Load data
        neurons = pd.read_feather(self.nodes_file, columns=columns_to_read)
        # rename the columns to the generic names
        neurons.columns = columns_to_write
        
        if attributes is not None:
            return neurons[neurons[self.BodyId].isin(ids)][columns_to_write]
        return neurons[neurons[self.BodyId].isin(ids)]

    def get_possible_columns(self) -> list[str]:
        """
        Get the possible columns of the dataset.
        """
        return list(typing.get_args(NeuronAttribute))

    def specific_selection_dict(
            self,
            selection_dict: SelectionDict
            ): # returns a self.SpecificSelectionDict
        """
        Returns the specific selection_dict for the connectome.
        Example: the generic key 'class_1' input will be replaced with
        'class:string' for MANC and 'super_class' for FAFB.
        """
        s_dict = self.SpecificSelectionDict() 
        for k, v in selection_dict.items():
            s_dict[self.specific_neuron_attribute(k)] = v
        return s_dict
    
    def specific_neuron_attribute(
            self,
            generic_n_a: NeuronAttribute
            ) -> str:
        """
        Returns the specific Neuron Attribute defined for the connectome.
        Example: the generic 'class_1' input will return 'class:string' for MANC
        and 'super_class' for FAFB.
        The mapped attributes are the ones common to all connectomes. 
        Specific mappings can be added through overloaded methods.
        """
        mapping = {
            self.BodyId: self.BodyId, # field that can be called directly
            "bodyId": self.BodyId,
            # connectivity
            # function
            self.nt_type: self.nt_type, # field that can be called directly
            "nt_type": self.nt_type,
            "nt_proba": self._nt_proba,
            # classification
            "class_1": self._class_1, 
            "class_2": self._class_2,
            "name": self._name,
            # morphology
            "side": self._side,
            "neuropil": self._neuropil,
            "size": self._size,
            # genetics
            "hemilineage": self._hemilineage,
        }
        equivalent_name = mapping.get(generic_n_a)
        return self.SpecificNeuronAttribute(equivalent_name)

    def specific_neuron_class(
            self,
            generic_n_c: NeuronClass
            ):
        """
        Returns the specific Neuron Class defined for the connectome.
        This method only defines the mapping for the common classes. Specific
        classes can be added through overloaded methods.
        """
        mapping = {
            "sensory": self.sensory,
            "sensory ascending": self.sensory_ascending,
            "ascending": self.ascending,
            "motor": self.motor,
            "descending": self.descending,
            "efferent": self.efferent,
            "unknown": self.unknown,
        }
        equivalent_name = mapping.get(generic_n_c)
        return self.SpecificNeuronClass(equivalent_name)

    def list_possible_attributes(self):
        raise NotImplementedError('This should only be called on child instances.')
    
# --- Specific classes --- #

# === MANC: Male Adult Neuronal Connectome
class MANC(ConnectomeReader):
    def __new__(
            cls,
            connectome_name: typing.Literal['MANC'],
            connectome_version: typing.Literal['v1.0','v1.2'],
        ):
        if cls is MANC:  # Only redirect if creating an instance directly
            match connectome_version:
                case 'v1.0':
                    return MANC_v_1_0('MANC', 'v1.0')
                case 'v1.2':
                    return MANC_v_1_2('MANC', 'v1.2')
                case _:
                    raise ValueError("Connectome not recognized.")
        return super().__new__(cls, connectome_name, connectome_version)
    
    def __init__(
            self,
            connectome_name: typing.Literal['MANC'],
            connectome_version: typing.Literal['v1.0','v1.2']
            ): # 2nd argument is useless, only for compatibility
        super().__init__('MANC', connectome_version)
        
    # ----- overwritten methods -----
    def _load_specific_namefields(self):
        """
        Need to define the attribute naming conventions.
        """

        # common to all
        # public (necessary to initialise the connections object)
        self.BodyId = ":ID(Body-ID)"
        self.start_bid = "START_ID(Body-ID)"
        self.end_bid = "END_ID(Body-ID)"
        self.syn_count = "weightHR:int"
        self.nt_type = "predictedNt:string"
        # private (used for processing)
        self._nt_proba = "predictedNtProb:float"
        self._class_1 = "class:string"
        self._class_2 = "subclass:string"
        self._name = "systematicType:string"
        self._side = "somaSide:string"
        self._neuropil = "somaNeuromere:string"
        self._hemilineage = "hemilineage:string"
        self._size = "size:long"
        # specific to MANC -> need to add to ::specific_neuron_attribute()
        self._tracing_status = "status:string"
        self._entry_nerve = "entryNerve:string"
        self._exit_nerve = "exitNerve:string"
        self._position = "position:point{srid:9157}"

        self.SpecificNeuronAttribute = typing.Literal[
            self.BodyId,
            self.nt_type,
            self._nt_proba,
            self._class_1,
            self._class_2,
            self._name,
            self._side,
            self._neuropil,
            self._hemilineage,
            self._size,
            self._tracing_status,
            self._entry_nerve,
            self._exit_nerve,
            self._position,
        ]


        # Existing fields in the MANC v1.0 for future reference
        '''
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
        '''
            
    def _load_specific_neuron_classes(self):
        """
        Name the neuron classes.
        """
        # common types
        self._sensory = "sensory neuron"
        self._motor = "motor neuron"
        self._ascending = "ascending neuron"
        self._descending = "descending neuron"
        # specific types
        self._intrinsic = "intrinsic neuron"
        self._glia = "Glia"
        self._sensory_ascending = "sensory ascending"
        self._efferent = "efferent neuron"
        self._efferent_ascending = "efferent ascending"
        self._unknown = "TBD"
        self._sensory_unknown = "Sensory TBD"
        self._interneuron_unknown = "Interneuron TBD"

        self.SpecificNeuronClass = typing.Literal[
            self._sensory,
            self._motor,
            self._ascending,
            self._descending,
            self._intrinsic,
            self._glia,
            self._sensory_ascending,
            self._efferent,
            self._efferent_ascending,
            self._unknown,
            self._sensory_unknown,
            self._interneuron_unknown,
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

    # public methods
    # --- additions
    def specific_neuron_attribute(
            self,
            generic_n_a: NeuronAttribute
            ) -> str:
        """
        Converts the generic Neuron Attribute to the specific one.
        It first tries to map the attribute defined for all connectomes. If it
        fails, it looks for specific attributes only defined in this connectome.
        """
        try:
            converted_type = super().__specific_neuron_attribute(generic_n_a)
        except KeyError:
            # look for specific attributes only defined in this connectome
            mapping = {
                "tracing_status": self._tracing_status,
                "position": self._position,
                "entry_nerve": self._entry_nerve,
                "exit_nerve": self._exit_nerve,
            }
            try:
                new_attribute = mapping.get(generic_n_a)
                converted_type = self.SpecificNeuronAttribute(new_attribute)
            except KeyError:
                raise ValueError(
                    f"ConnectomeReader::specific_neuron_attribute().\
                    The attribute {generic_n_a} is not defined in {self.connectome_name}."
                    )
        return converted_type
    
    def specific_neuron_class(
            self,
            generic_n_c: NeuronClass
            ):
        """
        Converts the generic Neuron Class to the specific one.
        """
        try:
            converted_type = super().__specific_neuron_class(generic_n_c)
        except KeyError:
            # look for specific classes only defined in this connectome
            mapping = {
                "intrinsic": self._intrinsic_neuron,
                "glia": self._glia,
                "sensory ascending": self._sensory_ascending,
                "efferent": self._efferent,
                "efferent ascending": self._efferent_ascending,
                "unknown": self._unknown,
                "sensory_unknown": self._sensory_unknown,
                "interneuron_unknown": self._interneuron_unknown,
            }
            try:
                new_class = mapping.get(generic_n_c)
                converted_type = self.SpecificNeuronClass(new_class)
            except KeyError:
                raise ValueError(
                    f"ConnectomeReader::specific_neuron_class().\
                    The class {generic_n_c} is not defined in {self.connectome_name}."
                    )
        return converted_type

    def list_possible_attributes(self):
        """
        List the possible attributes of the dataset.
        """
        all_attr = [
            "nt_type", "nt_proba", "class_1", "class_2", "name", "side",
            "neuropil", "hemilineage", "size", "tracing_status", "entry_nerve",
            "exit_nerve", "position"
        ]
        return all_attr

# Specific versions of MANC
class MANC_v_1_0(MANC):
    def __init__(
            self,
            connectome_name: typing.Literal['MANC'],
            connectome_version: typing.Literal['v1.0'],
            ):
        super().__init__('MANC', 'v1.0')

class MANC_v_1_2(MANC):
    def __init__(self, connectome_name, connectome_version):
        super().__init__('MANC', 'v1.2')

# === FAFB: Female Adult Fly Brain
class FAFB(ConnectomeReader):
    def __new__(
            cls,
            connectome_name: typing.Literal['FAFB'],
            connectome_version: typing.Literal['v630','v783'],
        ):
        if cls is FAFB:
            match connectome_version:
                case 'v630':
                    return FAFB_v630('FAFB', 'v630')
                case 'v783':
                    return FAFB_v783('FAFB', 'v783')
                case _:
                    raise ValueError("Connectome not recognized.")
        return super().__new__(cls, connectome_name, connectome_version)
        
    def __init__(
            self,
            connectome_name: typing.Literal['FAFB'],
            connectome_version: typing.Literal['v630','v783'],
            ):
        super().__init__('FAFB', connectome_version) # 2nd argument is useless, only for compatibility
        
    # ----- overwritten methods -----
    def _load_specific_namefields(self):
        """
        Need to define the fields that are common to all connectomes.
        BodyId, start_bid, end_bid, syn_count, nt_type, class_1, class_2, neuron_attributes
        """
        
        # common to all
        self.BodyId = "root_id"
        self.start_bid = "pre_root_id"
        self.end_bid = "post_root_id"
        self.syn_count = "syn_count"
        self.nt_type = "nt_type"
        self._nt_proba = "nt_type_score"
        self._class_1 = "super_class"
        self._class_2 = "class"
        self._side = "side"
        self._neuropil = "cell_type" # TODO: check, probably wrong
        self._hemilineage = "hemilineage"
        self._size = "size_nm"
        # specific to FAFB -> need to add to ::specific_neuron_attribute()
        self._nerve = "nerve"
        self._area = "area_nm"
        self._length = "length_nm"
        self._flow = "flow",


        self.NeuronAttribute = typing.Literal[
            self.BodyId,
            self.nt_type,
            self._nt_proba,
            self._class_1,
            self._class_2,
            self._side,
            self._neuropil,
            self._hemilineage,
            self._size,
            self._nerve,
            self._area,
            self._length,
            self._flow,
        ]
            
    def _load_specific_neuron_classes(self):
        """
        Name the neuron classes.
        """
        # common types
        self._sensory = "sensory"
        self._motor = "motor"
        self._ascending = "ascending"
        self._descending = "descending"
        # specific types
        self._central = "central"
        self._endocrine = "endocrine"
        self._optic = "optic"
        self._visual_centrifugal = "visual centrifugal"
        self._visual_projection = "visual projection"
        self._other = "other"

        self.SpecificNeuronClass = typing.Literal[
            self._sensory,
            self._motor,
            self._ascending,
            self._descending,
            self._central,
            self._endocrine,
            self._optic,
            self._visual_centrifugal,
            self._visual_projection,
            self._other,
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
        
    # public methods
        # --- additions
    def specific_neuron_attribute(
            self,
            generic_n_a: NeuronAttribute
            ) -> str:
        """
        Converts the generic Neuron Attribute to the specific one.
        It first tries to map the attribute defined for all connectomes. If it
        fails, it looks for specific attributes only defined in this connectome.
        """
        try:
            converted_type = super().__specific_neuron_attribute(generic_n_a)
        except KeyError:
            # look for specific attributes only defined in this connectome
            mapping = {
                "nerve": self._nerve,
                "area": self._area,
                "length": self._length,
                "flow": self._flow,
            }
            try:
                new_attribute = mapping.get(generic_n_a)
                converted_type = self.SpecificNeuronAttribute(new_attribute)
            except KeyError:
                raise ValueError(
                    f"ConnectomeReader::specific_neuron_attribute().\
                    The attribute {generic_n_a} is not defined in {self.connectome_name}."
                    )
        return converted_type
    
    def specific_neuron_class(
            self,
            generic_n_c: NeuronClass
            ):
        """
        Converts the generic Neuron Class to the specific one.
        """
        try:
            converted_type = super().__specific_neuron_class(generic_n_c)
        except KeyError:
            # look for specific classes only defined in this connectome
            mapping = {
                "central": self._central,
                "endocrine": self._endocrine,
                "optic": self._optic,
                "visual centrifugal": self._visual_centrifugal,
                "visual projection": self._visual_projection,
                "other": self._other,
            }
            try:
                new_class = mapping.get(generic_n_c)
                converted_type = self.SpecificNeuronClass(new_class)
            except KeyError:
                raise ValueError(
                    f"ConnectomeReader::specific_neuron_class().\
                    The class {generic_n_c} is not defined in {self.connectome_name}."
                    )
        return converted_type

    def list_possible_attributes(self):
        """
        List the possible attributes of the dataset.
        """
        all_attr = [
            "nt_type", "nt_proba", "class_1", "class_2", "side",
            "neuropil", "hemilineage", "size", "nerve", "area", "length", "flow"
        ]
        return all_attr

# Specific versions of FAFB
class FAFB_v630(FAFB):
    def __init__(self, connectome_name, connectome_version):
        super().__init__(connectome_name, connectome_version)

class FAFB_v783(FAFB):
    def __init__(self, connectome_name, connectome_version):
        super().__init__(connectome_name, connectome_version)

if __name__ == "__main__":
    # Test the class
    manc = ConnectomeReader('MANC', 'v1.0')
    if manc.exists_tracing_status():
        print("MANC has tracing status.")
                
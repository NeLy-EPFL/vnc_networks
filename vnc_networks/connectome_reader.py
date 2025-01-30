#!/usr/bin/env python3
"""
Definition of the ConnectomeReader class.
Class defining the methods and labels to read different connectomes.

Each instance also redefines the generic NeuronAttribute and NeuronClass
data types to the specific ones of the connectome.
"""

import os
import typing
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from . import params
from .params import BodyId, NeuronAttribute, NeuronClass, SelectionDict


# --- Parent class, abstract --- #
class ConnectomeReader(ABC):

    @abstractmethod
    def __init__(self):

        self.raw_data_dir = params.RAW_DATA_DIR
        # specific namefields
        self._load_specific_namefields()
        self._load_specific_neuron_classes()
        self._load_specific_directories()

    # ----- virtual private methods -----
    @abstractmethod
    def _load_specific_namefields(self):
        pass

    @abstractmethod
    def _load_specific_neuron_classes(self):
        pass   

    @abstractmethod 
    def _load_specific_directories(self):
        pass
        
    # ----- public methods -----
    # --- abstract methods
    @abstractmethod
    def get_synapse_df(self, body_id: BodyId) -> pd.DataFrame:
        """
        Get the synapses of a neuron.
        """
        pass

    @abstractmethod
    def get_synapse_locations(self, synapse_ids: list[int]) -> pd.DataFrame:
        """
        Get the locations of the synapses.
        """
        pass

    @abstractmethod
    def get_synapse_neuropil(self, synapse_ids: list[int]) -> pd.DataFrame:
        """
        Get the neuropil of the synapses.
        """
        pass
 
    @abstractmethod
    def list_possible_attributes(self) -> list[str]:
        """
        List the possible attributes for a neuron.
        """
        pass

    # --- partially reused methods
    def sna(
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
            "body_id": self._body_id,
            "start_bid": self._start_bid,
            "end_bid": self._end_bid,
            "syn_count": self._syn_count,
            # connectivity
            # function
            "nt_type": self._nt_type,
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
        if equivalent_name is None:
            raise KeyError
        return equivalent_name

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
            "sensory": self._sensory,
            "ascending": self._ascending,
            "motor": self._motor,
            "descending": self._descending,
        }
        equivalent_name = mapping.get(generic_n_c)
        if equivalent_name is None:
            raise KeyError # this will be caught and handeled by child instances
        return equivalent_name
    
    def decode_neuron_class(self, specific_class: str) -> NeuronClass:
        """
        Decode the specific class to the generic one.
        """
        mapping = {
            self._sensory: "sensory",
            self._motor: "motor",
            self._ascending: "ascending",
            self._descending: "descending",
        }
        equivalent_name = mapping.get(specific_class)
        if equivalent_name is None:
            raise KeyError
        return equivalent_name
    
    # --- common methods
    def get_connections(
            self,
            columns: list[NeuronAttribute],
            keep_only_traced_neurons: bool = False,
            ):
        """
        Load the connections of the connectome.
        If keep_only_traced_neurons is True, only the neurons that have been traced
        will be kept.

        Parameters
        ----------
        columns : list
            The columns to load.
        """
        # translate to specific names
        columns_to_read = [
            self.sna(a) for a in columns
        ]
        # Identify which neurons are traced
        if keep_only_traced_neurons and self.exists_tracing_status():
            nodes_data = pd.read_feather(
                self._nodes_file,
                columns=[self._body_id, self._tracing_status]
                )
            nodes_data = nodes_data[
                nodes_data[self._tracing_status] == self.traced_entry
                ]
            traced_bids = nodes_data[self._body_id].values

        # Load the connections
        df = pd.read_feather(self._connections_file, columns=columns_to_read)

        # rename to generic names
        df.columns = columns

        # Filter out the untraced neurons
        if keep_only_traced_neurons:
            df = df[df['start_bid'].isin(traced_bids)]
            df = df[df['end_bid'].isin(traced_bids)]

        return df

    def list_all_nodes(self) -> list[BodyId]:
        """
        List all the pre-synaptic neurons existing in the connectome.
        """
        data = pd.read_feather(self._nodes_file, columns=[self._body_id])
        return list(data[self._body_id].values)

    def exists_tracing_status(self):
        """
        Identify if '_tracing_status' is a field in the connectome
        """
        if hasattr(self, '_tracing_status'):
            self.traced_entry = "Traced"
            return True
        return False

    def get_neuron_bodyids(
        self,
        selection_dict: Optional[SelectionDict] = None,
        nodes: Optional[list[BodyId] | list[int]] = None,
    ) -> list[BodyId]:
        """
        Get the Ids of the neurons in the dataset.
        Select (keep) according to the selection_dict.
        Different criteria are treated as 'and' conditions.

        For the specific case of "class_1" that refers to the NeuronClass,
        we need to verify both the generic and the specific names.
        """
        s_dict = self.specific_selection_dict(selection_dict)

        # Identify columns to load
        columns_to_read = [
            self.sna(a) for a in selection_dict.keys()
        ]
        columns_to_write = list(selection_dict.keys())
        if 'body_id' not in selection_dict.keys():
            columns_to_read.append(self._body_id) # specific name field
            columns_to_write.append('body_id') # generic name field
        
        neurons = pd.read_feather(self._nodes_file, columns=list(columns_to_read))

        if s_dict is not None:
            for key in s_dict.keys():
                if key == self.sna('class_1'):
                    try: # will work if a generic NeuronClass is given
                        specific_class = self.specific_neuron_class(s_dict[key])
                    except KeyError: # will work if a specific NeuronClass is given
                        specific_class = s_dict[key]
                    neurons = neurons[
                        neurons[self._class_1] == specific_class
                        ]
                else:
                    neurons = neurons[neurons[key] == s_dict[key]]
        if nodes is not None:
            neurons = neurons[neurons[self._body_id].isin(nodes)]
    
        return list(neurons[self._body_id].values)

    def get_neurons_from_class(self, class_: NeuronClass) -> list[BodyId]:
        """
        Get the bodyids of neurons of a certain class (e.g. sensory).
        """
        # verify if the class is indeed a neuron class for this dataset
        specific_class = self.specific_neuron_class(class_)
        return self.get_neuron_bodyids({self.class_1: specific_class})

    def load_data_neuron(
        self,
        id_: BodyId | int,
        attributes: Optional[list[NeuronAttribute]] = None,
    ) -> pd.DataFrame:
        """
        Load the data of a neuron with a certain id.
        
        Parameters
        ----------
        id : BodyId | int
            The id of the neuron.
        attributes : list
            The attributes to load.

        Returns
        -------
        pandas.DataFrame
            The data of the neuron.
        """
        # Identify columns to load
        columns_to_read = [
            self.sna(a) for a in attributes
        ]
        columns_to_write = attributes
        if 'body_id' not in attributes:
            columns_to_read.append(self._body_id) # specific name field
            columns_to_write.append('body_id') # generic name field

        # Load data
        neurons = pd.read_feather(self._nodes_file, columns=columns_to_read)
        # rename the columns to the generic names
        neurons.columns = columns_to_write
        if attributes is not None:
            return neurons[neurons['body_id'] == id_][columns_to_write]
        return neurons[neurons['body_id'] == id_]

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
        # Identify columns to load
        columns_to_read = [
            self.sna(a) for a in attributes
        ]
        columns_to_write = attributes
        if 'body_id' not in attributes:
            columns_to_read.append(self._body_id) # specific name field
            columns_to_write.append('body_id') # generic name field

        # Load data
        neurons = pd.read_feather(self._nodes_file, columns=columns_to_read)
        # rename the columns to the generic names
        neurons.columns = columns_to_write
        
        if attributes is not None:
            return neurons[neurons['body_id'].isin(ids)][columns_to_write]
        return neurons[neurons['body_id'].isin(ids)]

    def specific_selection_dict(
            self,
            selection_dict: SelectionDict
            ):
        """
        Returns the specific selection_dict for the connectome.
        Example: the generic key 'class_1' input will be replaced with
        'class:string' for MANC and 'super_class' for FAFB.
        """
        s_dict = {} 
        for k, v in selection_dict.items():
            s_dict[self.sna(k)] = v
        return s_dict
    
    @staticmethod
    def node_base_attributes() -> list[NeuronAttribute]:
        """
        Returns a list of the base attributes that all nodes have, used
        to initialise a neuron::Neuron() object for instance.
        """
        list_node_attributes: list[NeuronAttribute] = [
                "body_id",
                # function
                "nt_type",
                "nt_proba",
                # classification
                "class_1",
                "class_2",
                "name",
                # morphology
                "side",
                "neuropil",
                "size",
                # genetics
                "hemilineage",
            ]
        return list_node_attributes

# --- Specific classes --- #

# === MANC: Male Adult Neuronal Connectome
class MANC(ConnectomeReader):
    _versions = {} # Will contain the specific versions of MANC

    def __new__(
            cls,
            connectome_version: str,
            *args,
            **kwargs,
        ):
        """
        Instantiate the child class corresponding to the version of MANC.
        """
        if connectome_version in cls._versions:
            return super().__new__(cls._versions[connectome_version])
        raise ValueError("MANC version not recognized.")
    
    def __init_subclass__(
            cls,
            connectome_version: Optional[str] = None,
            **kwargs,
        ):
        """
        Register the existing versions of MANC.
        """
        super().__init_subclass__(**kwargs)
        if connectome_version is not None:
            MANC._versions[connectome_version] = cls
    
    @abstractmethod
    def __init__(
            self,
            connectome_version: str,
            ): # 2nd argument is useless, only for compatibility
        
        self.connectome_name = 'MANC'
        self.connectome_version = connectome_version
        super().__init__()
        
    # ----- overwritten methods -----
    def _load_specific_namefields(self):
        """
        Need to define the attribute naming conventions.
        """
        # common to all
        self._body_id = ":ID(Body-ID)"
        self._start_bid = ":START_ID(Body-ID)"
        self._end_bid = ":END_ID(Body-ID)"
        self._syn_count = "weightHR:int"
        self._nt_type = "predictedNt:string"
        self._nt_proba = "predictedNtProb:float"
        self._class_1 = "class:string"
        self._class_2 = "subclass:string"
        self._name = "systematicType:string"
        self._side = "somaSide:string"
        self._neuropil = "somaNeuromere:string"
        self._hemilineage = "hemilineage:string"
        self._size = "size:long"
        # specific to MANC -> need to add to ::sna()
        self._type = "type:string"
        self._tracing_status = "status:string"
        self._entry_nerve = "entryNerve:string"
        self._exit_nerve = "exitNerve:string"
        self._position = "position:point{srid:9157}"
        self._nb_pre_synapses = "pre:int",
        self._nb_post_synapses = "post:int",
        self._nb_pre_neurons = "upstream:int",
        self._nb_post_neurons = "downstream:int",
        # Synapse specific
        self._start_synset_id = ":START_ID(SynSet-ID)"
        self._end_synset_id = ":END_ID(SynSet-ID)"
        self._start_syn_id = ":START_ID(Syn-ID)"
        self._end_syn_id = ":END_ID(Syn-ID)"
        self._syn_id = ":ID(Syn-ID)"
        self._syn_location = "location:point{srid:9157}"

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
        
        self._connectome_dir = os.path.join(
            self.raw_data_dir,
            "manc",
            "v1.0",
            "neuprint_manc_v1.0",
            "neuprint_manc_v1.0_ftr",
        )
        self._nodes_file = os.path.join(
            self._connectome_dir,
            "Neuprint_Neurons_manc_v1.ftr"
            )
        self._connections_file = os.path.join(
            self._connectome_dir,
            "Neuprint_Neuron_Connections_manc_v1.ftr"
            )
        
        # unique to MANC
        self._synapseset_file = os.path.join(
            self._connectome_dir,
            "Neuprint_SynapseSet_to_Synapses_manc_v1.ftr"
        )
        self._neuron_synapseset_file = os.path.join(
            self._connectome_dir,
            "Neuprint_Neuron_to_SynapseSet_manc_v1.ftr"
        )
        self._synapse_file = os.path.join(
            self._connectome_dir,
            "Neuprint_Synapses_manc_v1.ftr"
        )

    # public methods
    def get_synapse_df(self, body_id: BodyId) -> pd.DataFrame:
        """
        Load the synapse ids for the neuron.
        should define the columns ['synapse_id','start_id','end_id']
        """
        # neuron to synapse set
        neuron_to_synapse = pd.read_feather(self._neuron_synapseset_file)
        synset_list = neuron_to_synapse.loc[
            neuron_to_synapse[self._start_bid] == body_id
        ][self._end_synset_id].values

        # synapse set to synapse
        synapses = pd.read_feather(self._synapseset_file)
        synapses = synapses.loc[synapses[self._start_synset_id].isin(synset_list)]
        synapses.reset_index(drop=True, inplace=True)

        # build a dataframe with columns 'syn_id', 'synset_id'
        synapse_df = pd.DataFrame(
            {
                "synapse_id": synapses[self._end_syn_id],
                "synset_id": synapses[self._start_synset_id],
            }
        )
        synapse_df["start_id"] = synapse_df["synset_id"].apply(
            lambda x: int(x.split("_")[0])
        )  # body id of the presynaptic neuron
        synapse_df["end_id"] = synapse_df["synset_id"].apply(
            lambda x: int(x.split("_")[1])
        )  # body id of the postsynaptic neuron
        synapse_df["position"] = synapse_df["synset_id"].apply(
            lambda x: x.split("_")[2]
        )  # pre or post

        # remove the synapses that belong to partner neurons
        synapse_df = synapse_df[synapse_df["position"] == "pre"]
        synapse_df.drop(columns=["position"], inplace=True)

        return synapse_df
  
    def get_synapse_locations(self, synapse_ids: list[int]) -> pd.DataFrame:
        """
        Get the locations of the synapses.
        """
        data = pd.read_feather(
            self._synapse_file,
            columns = [self._syn_id, self._syn_location]
            )
        data = data.loc[data[self._syn_id].isin(synapse_ids)]
        data.columns = ['synapse_id', 'location']

        locations = data['location'].values

        # split the location into X, Y, Z
        X, Y, Z = [], [], []
        for loc in locations:
            name = loc.replace('x','"x"').replace('y','"y"').replace('z','"z"')
            try:
                pos = eval(name)  # read loc as a dict, use of x,y,z under the hood
            except TypeError:
                pos = {"x": np.nan, "y": np.nan, "z": np.nan}
                print(f'Type Error in reading location for {name}')
            except NameError:
                pos = {"x": np.nan, "y": np.nan, "z": np.nan}
                print(f'Name Error in reading location for {name}')
            if not isinstance(pos, dict):
                pos = {"x": np.nan, "y": np.nan, "z": np.nan}
            X.append(pos["x"])
            Y.append(pos["y"])
            Z.append(pos["z"])
        data["X"] = X
        data["Y"] = Y
        data["Z"] = Z

        data.drop(columns=["location"], inplace=True)

        return data
    
    def get_synapse_neuropil(self, synapse_ids: list[int]) -> pd.DataFrame:
        """
        Get the neuropil of the synapses.
        In MANC, this means finding the name of the neuropil column for which 
        the entry is True.
        """
        roi_file = os.path.join(self._connectome_dir, "all_ROIs.txt")
        rois = list(pd.read_csv(roi_file, sep="\t").values.flatten())

        # find the subset of the synapses in the dataset that we care about for this neuron
        # then for each possible ROI, check which synapses are in that ROI
        # store each synapse's ROI in the neuropil column
        data = pd.read_feather(self._synapse_file, columns = [self._syn_id])
        data.columns = ['synapse_id']
        data["neuropil"] = "None"

        for roi in rois:
            column_name = roi + ":boolean"
            roi_column = pd.read_feather(
                self._synapse_file,
                columns=[column_name, self._syn_id],
            )[synapse_ids]
            synapses_in_roi = roi_column.loc[
                roi_column[column_name] == True, self._syn_id
            ].values  # type: ignore
            data.loc[
                data["synapse_id"].isin(synapses_in_roi), "neuropil"
            ] = roi
        return data

    def list_possible_attributes(self) -> list[str]:
        """
        List the possible attributes for a neuron.
        """
        all_attr = [
            "nt_type", "nt_proba", "class_1", "class_2", "name", "type", "side",
            "neuropil", "hemilineage", "size", "tracing_status", "entry_nerve",
            "exit_nerve", "position", "nb_pre_synapses", "nb_post_synapses",
            "nb_pre_neurons", "nb_post_neurons",
        ]
        return all_attr
    
    # --- additions
    def sna( # specific_neuron_attribute, abbreviated due to frequent use
            self,
            generic_n_a: NeuronAttribute
            ) -> str:
        """
        Converts the generic Neuron Attribute to the specific one.
        It first tries to map the attribute defined for all connectomes. If it
        fails, it looks for specific attributes only defined in this connectome.
        """
        try:
            converted_type = super().sna(generic_n_a)
        except KeyError:
            # look for specific attributes only defined in this connectome
            mapping = {
                "type": self._type,
                "tracing_status": self._tracing_status,
                "position": self._position,
                "entry_nerve": self._entry_nerve,
                "exit_nerve": self._exit_nerve,
                "nb_pre_synapses": self._nb_pre_synapses,
                "nb_post_synapses": self._nb_post_synapses,
                "nb_pre_neurons": self._nb_pre_neurons,
                "nb_post_neurons": self._nb_post_neurons,
            }
            try:
                converted_type = mapping.get(generic_n_a)
                if converted_type is None:
                    raise KeyError
                # if the converted type is a tuple, get the first element
                if isinstance(converted_type, tuple):
                    converted_type = converted_type[0]
            except KeyError:
                raise ValueError(
                    f"ConnectomeReader::sna().\
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
            converted_type = super().specific_neuron_class(generic_n_c)
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
                converted_type = mapping.get(generic_n_c)
                if converted_type is None:
                    raise KeyError
            except KeyError:
                raise ValueError(
                    f"ConnectomeReader::specific_neuron_class().\
                    The class {generic_n_c} is not defined in {self.connectome_name}."
                    )
        return converted_type
    
    def decode_neuron_class(self, specific_class: str) -> NeuronClass:
        """
        Decode the specific class to the generic one.
        """
        try:
            converted_class = super().decode_neuron_class(specific_class)
        except KeyError:
            # look for specific classes only defined in this connectome
            mapping = {
                self._intrinsic: "intrinsic",
                self._glia: "glia",
                self._sensory_ascending: "sensory ascending",
                self._efferent: "efferent",
                self._efferent_ascending: "efferent ascending",
                self._unknown: "unknown",
                self._sensory_unknown: "sensory_unknown",
                self._interneuron_unknown: "interneuron_unknown",
                None: "unknown",
            }
            try:
                converted_class = mapping.get(specific_class)
                if converted_class is None:
                    raise KeyError
            except KeyError:
                raise ValueError(
                    f"ConnectomeReader::decode_neuron_class().\
                    The class {specific_class} is not defined in {self.connectome_name}."
                    )   
        return converted_class

# Specific versions of MANC
class MANC_v_1_0(MANC, connectome_version='v1.0'):
    def __init__(self, connectome_version: str):
        super().__init__(connectome_version)

class MANC_v_1_2(MANC, connectome_version='v1.2'):
    def __init__(self, connectome_version: str):
        super().__init__(connectome_version)

# === FAFB: Female Adult Fly Brain
class FAFB(ConnectomeReader):
    _versions = {} # Will contain the specific versions of FAFB

    def __new__(
            cls,
            connectome_version: str,
            *args,
            **kwargs,
        ):
        """
        Instantiate the child class corresponding to the version of FAFB.
        """
        if connectome_version in cls._versions:
            return super().__new__(cls._versions[connectome_version])
        raise ValueError("FAFB version not recognized.")
    
    def __init_subclass__(
            cls,
            connectome_version: Optional[str] = None,
            **kwargs,
        ):
        """
        Register the existing versions of FAFB.
        """
        super().__init_subclass__(**kwargs)
        if connectome_version is not None:
            FAFB._versions[connectome_version] = cls

    @abstractmethod    
    def __init__(
            self,
            connectome_version: str,
            ):
        
        self.connectome_name = 'FAFB'
        self.connectome_version = connectome_version
        super().__init__()
        
    # ----- overwritten methods -----
    def _load_specific_namefields(self):
        """
        Need to define the fields that are common to all connectomes.
        BodyId, start_bid, end_bid, syn_count, nt_type, class_1, class_2, neuron_attributes
        """
        
        # common to all
        self._body_id = "root_id"
        self._start_bid = "pre_root_id"
        self._end_bid = "post_root_id"
        self._syn_count = "syn_count"
        self._nt_type = "nt_type"
        self._nt_proba = "nt_type_score"
        self._class_1 = "super_class"
        self._class_2 = "class"
        self._side = "side"
        self._neuropil = "cell_type" # TODO: check, probably wrong
        self._hemilineage = "hemilineage"
        self._size = "size_nm"
        # specific to FAFB -> need to add to ::sna()
        self._nerve = "nerve"
        self._area = "area_nm"
        self._length = "length_nm"
        self._flow = "flow"
            
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
        
        self._connectome_dir = os.path.join(
            self.raw_data_dir,
            "",
        )
        self._nodes_file = os.path.join(
            self._connectome_dir,
            ""
            )
        self._connections_file = os.path.join(
            self._connectome_dir,
            ""
            )
        
    # public methods
    # --- additions
    def get_synapse_df(self, body_id: BodyId) -> pd.DataFrame:
        """
        Load the synapse ids for the neuron.
        should define the columns ['synapse_id','start_id','end_id']
        """
        raise NotImplementedError('Method not implemented yet on FAFB.')
    
    def get_synapse_locations(self, synapse_ids: list[int]) -> pd.DataFrame:
        """
        Get the locations of the synapses.
        """
        raise NotImplementedError('Method not implemented yet on FAFB.')

    def get_synapse_neuropil(self, synapse_ids: list[int]) -> pd.DataFrame:
        raise NotImplementedError('Method not implemented yet on FAFB.')
    
    def sna(
            self,
            generic_n_a: NeuronAttribute
            ) -> str:
        """
        Converts the generic Neuron Attribute to the specific one.
        It first tries to map the attribute defined for all connectomes. If it
        fails, it looks for specific attributes only defined in this connectome.
        """
        try:
            converted_type = super().sna(generic_n_a)
        except KeyError:
            # look for specific attributes only defined in this connectome
            mapping = {
                "nerve": self._nerve,
                "area": self._area,
                "length": self._length,
                "flow": self._flow,
            }
            try:
                converted_type = mapping.get(generic_n_a)
                if converted_type is None:
                    raise KeyError
            except KeyError:
                raise ValueError(
                    f"ConnectomeReader::sna().\
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
            converted_type = super().specific_neuron_class(generic_n_c)
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
                converted_type = mapping.get(generic_n_c)
                if converted_type is None:
                    raise KeyError 
            except KeyError:
                raise ValueError(
                    f"ConnectomeReader::specific_neuron_class().\
                    The class {generic_n_c} is not defined in {self.connectome_name}."
                    )
        return converted_type
    
    def decode_neuron_class(self, specific_class: str) -> NeuronClass:
        """
        Decode the specific class to the generic one.
        """
        try:
            converted_class = super().decode_neuron_class(specific_class)
        except KeyError:
            # look for specific classes only defined in this connectome
            mapping = {
                self._central: "central",
                self._endocrine: "endocrine",
                self._optic: "optic",
                self._visual_centrifugal: "visual centrifugal",
                self._visual_projection: "visual projection",
                self._other: "other",
            }
            try:
                converted_class = mapping.get(specific_class)
                if converted_class is None:
                    raise KeyError
            except KeyError:
                raise ValueError(
                    f"ConnectomeReader::decode_neuron_class().\
                    The class {specific_class} is not defined in {self.connectome_name}."
                    )
        return converted_class

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
class FAFB_v630(FAFB, connectome_version='v630'):
    def __init__(self, connectome_version: str):
        super().__init__(connectome_version)

class FAFB_v783(FAFB, connectome_version='v783'):
    def __init__(self, connectome_version: str):
        super().__init__(connectome_version)


                

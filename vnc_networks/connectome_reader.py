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
from .params import (
    BodyId,
    ConnectomePreprocessingOptions,
    NeuronAttribute,
    NeuronClass,
    SelectionDict,
)


# --- Parent class, abstract --- #
class ConnectomeReader(ABC):
    # Names of common properties
    # these get overwritten by custom connectomes which use different names
    _body_id = "body_id"
    _syn_id = "syn_id"
    _start_bid = "start_bid"
    _end_bid = "end_bid"
    _syn_count = "syn_count"
    _nt_type = "nt_type"
    _nt_proba = "nt_proba"
    _class_1 = "class_1"
    _class_2 = "class_2"
    _name = "name"
    _side = "side"
    _neuropil = "neuropil"
    _hemilineage = "hemilineage"
    _size = "size"
    _position = "position"
    _location = "location"
    _target = "target"

    # Names of neuron types
    # these get overwritten by custom connectomes which use different names
    _sensory = "sensory"
    _motor = "motor"
    _ascending = "ascending"
    _descending = "descending"

    def __init__(
        self,
        connectome_name: str,
        connectome_version: str,
        connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
    ):
        self.connectome_name = connectome_name
        self.connectome_version = connectome_version
        # if no explicit preprocessing options are provided, use the default
        if connectome_preprocessing is None:
            connectome_preprocessing = ConnectomePreprocessingOptions()
        self.connectome_preprocessing = connectome_preprocessing

        # specific namefields
        self._load_specific_namefields()
        self._load_specific_neuron_classes()
        self._load_data_directories()
        self._create_output_directories()

    def _create_output_directories(self):
        """
        Creates the directories where data is stored.
        """
        # preprocessed data saving
        preprocessing_dir = params.PREPROCESSED_DATA_DIR
        self._neuron_save_dir = os.path.join(
            preprocessing_dir, self.connectome_name, self.connectome_version, "neurons"
        )
        os.makedirs(self._neuron_save_dir, exist_ok=True)
        self._connections_save_dir = os.path.join(
            preprocessing_dir,
            self.connectome_name,
            self.connectome_version,
            "connections",
        )
        os.makedirs(self._connections_save_dir, exist_ok=True)

        # data saving directories
        processed_dir = params.PROCESSED_DATA_DIR
        self._data_save_dir = os.path.join(
            processed_dir, self.connectome_name, self.connectome_version
        )
        os.makedirs(self._data_save_dir, exist_ok=True)

        plots_dir = params.FIG_DIR
        self._plot_save_dir = os.path.join(
            plots_dir, self.connectome_name, self.connectome_version
        )
        os.makedirs(self._plot_save_dir, exist_ok=True)

    # ----- virtual private methods -----
    @abstractmethod
    def _load_specific_namefields(self): ...

    @abstractmethod
    def _load_specific_neuron_classes(self): ...

    @abstractmethod
    def _load_data_directories(self): ...

    @abstractmethod
    def _get_traced_bids(self) -> list[BodyId]:
        """
        Get the body ids of the traced neurons.
        """
        ...

    @abstractmethod
    def _load_connections(self) -> pd.DataFrame:
        """
        Load the connections of the connectome.
        """
        ...

    # ----- public methods -----
    # --- abstract methods
    @abstractmethod
    def get_synapse_df(self, body_id: BodyId | int) -> pd.DataFrame:
        """
        Get the locations of the synapses.
        """
        ...

    @abstractmethod
    def get_synapse_neuropil(
        self,
        synapse_ids: list[int],
    ) -> pd.DataFrame:
        """
        Get the neuropil of the synapses.
        """
        ...

    @abstractmethod
    def list_possible_attributes(self) -> list[str]:
        """
        List the possible attributes for a neuron.
        """
        ...

    @abstractmethod
    def list_all_nodes(self) -> list[BodyId]:
        """
        List all the pre-synaptic neurons existing in the connectome.
        """
        ...

    @abstractmethod
    def get_neuron_bodyids(
        self,
        selection_dict: SelectionDict = {},
        nodes: Optional[list[BodyId] | list[int]] = None,
    ) -> list[BodyId]:
        """
        Get the Ids of the neurons in the dataset.
        Select (keep) according to the selection_dict.
        Different criteria are treated as 'and' conditions.

        For the specific case of "class_1" that refers to the NeuronClass,
        we need to verify both the generic and the specific names.
        """
        ...

    @abstractmethod
    def load_data_neuron(
        self,
        id_: BodyId | int,
        attributes: list[NeuronAttribute] = [],
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
        ...

    @abstractmethod
    def load_data_neuron_set(
        self,
        ids: list[BodyId] | list[int],
        attributes: list[NeuronAttribute] = [],
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
        ...

    # --- partially reused methods
    def sna(self, generic_n_a: NeuronAttribute) -> str:
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
            "synapse_id": self._syn_id,
            # connectivity
            # function
            "nt_type": self._nt_type,
            "nt_proba": self._nt_proba,
            # classification
            "class_1": self._class_1,
            "class_2": self._class_2,
            "name": self._name,
            "target": self._target,
            # morphology
            "side": self._side,
            "neuropil": self._neuropil,
            "size": self._size,
            "position": self._position,
            # genetics
            "hemilineage": self._hemilineage,
        }
        equivalent_name = mapping.get(generic_n_a)
        if equivalent_name is None:
            raise KeyError
        return equivalent_name

    def decode_neuron_attribute(self, specific_attribute: str) -> NeuronAttribute:
        """
        Decode the specific attribute to the generic one.
        """
        mapping: dict[str, NeuronAttribute] = {
            self._body_id: "body_id",
            self._start_bid: "start_bid",
            self._end_bid: "end_bid",
            self._syn_count: "syn_count",
            self._syn_id: "synapse_id",
            # connectivity
            # function
            self._nt_type: "nt_type",
            self._nt_proba: "nt_proba",
            # classification
            self._class_1: "class_1",
            self._class_2: "class_2",
            self._name: "name",
            self._target: "target",
            # morphology
            self._side: "side",
            self._neuropil: "neuropil",
            self._size: "size",
            self._position: "position",
            # genetics
            self._hemilineage: "hemilineage",
        }
        equivalent_name = mapping.get(specific_attribute)
        if equivalent_name is None:
            raise KeyError
        return equivalent_name

    def specific_neuron_class(self, generic_n_c: NeuronClass):
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
            raise KeyError  # this will be caught and handled by child instances
        return equivalent_name

    def decode_neuron_class(self, specific_class: str | None) -> NeuronClass:
        """
        Decode the specific class to the generic one.
        """
        mapping: dict[str | None, NeuronClass] = {
            self._sensory: "sensory",
            self._motor: "motor",
            self._ascending: "ascending",
            self._descending: "descending",
            None: "unknown",
        }
        equivalent_name = mapping.get(specific_class)
        if equivalent_name is None:
            raise KeyError
        return equivalent_name

    # --- common methods
    def get_connections(
        self,
        keep_only_traced_neurons: bool = False,
    ):
        """
        Load the connections of the connectome.
        Will read ['start_bid', 'end_bid', 'syn_count', 'nt_type'].
        If keep_only_traced_neurons is True, only the neurons that have been traced
        will be kept.

        Parameters
        ----------
        columns : list
            The columns to load.
        """
        # Load the connections
        df = self._load_connections()

        # Filter out the untraced neurons
        if keep_only_traced_neurons and self.exists_tracing_status():
            traced_bids = self._get_traced_bids()
            df = df[df["start_bid"].isin(traced_bids)]
            df = df[df["end_bid"].isin(traced_bids)]

        return df

    def exists_tracing_status(self):
        """
        Identify if '_tracing_status' is a field in the connectome
        """
        if hasattr(self, "_tracing_status"):
            self.traced_entry = "Traced"
            return True
        return False

    def get_neurons_from_class(self, class_: NeuronClass) -> list[BodyId]:
        """
        Get the bodyids of neurons of a certain class (e.g. sensory).
        """
        # verify if the class is indeed a neuron class for this dataset
        specific_class = self.specific_neuron_class(class_)
        return self.get_neuron_bodyids({self.class_1: specific_class})

    def specific_selection_dict(self, selection_dict: SelectionDict):
        """
        Returns the specific selection_dict for the connectome.
        Example: the generic key 'class_1' input will be replaced with
        'class:string' for MANC and 'super_class' for FAFB.
        """
        s_dict = {self.sna(k): v for k, v in selection_dict.items()}
        return s_dict

    def get_plots_dir(self) -> str:
        """
        Returns the directory where plots are saved.
        """
        return self._plot_save_dir

    def get_connections_save_dir(self) -> str:
        """
        Returns the directory where connections are saved.
        """
        return self._connections_save_dir

    def get_neuron_save_dir(self) -> str:
        """
        Returns the directory where neurons are saved.
        """
        return self._neuron_save_dir

    def get_fig_dir(self) -> str:
        """
        Returns the directory where figures are saved.
        """
        return self._plot_save_dir

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
            "target",
            # morphology
            "side",
            "neuropil",
            "size",
            "position",
            # genetics
            "hemilineage",
        ]
        return list_node_attributes


# --- Specific classes --- #


# === MANC: Male Adult Neuronal Connectome
class MANCReader(ConnectomeReader):
    # these will get implemented by specific MANC versions
    # data files
    _nodes_file: str
    _connections_file: str
    # neuron attributes
    _type: str
    _tracing_status: str
    _entry_nerve: str
    _exit_nerve: str
    _nb_pre_synapses: str
    _nb_post_synapses: str
    _nb_pre_neurons: str
    _nb_post_neurons: str

    def __init__(
        self,
        connectome_version: str,
        connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
    ):  # 2nd argument is useless, only for compatibility
        super().__init__("manc", connectome_version, connectome_preprocessing)

        self.nt_weights = {
            "acetylcholine": +1,
            "gaba": -1,
            "glutamate": -1
            if self.connectome_preprocessing.glutamate_inhibitory
            else +1,
            "unknown": 0,
            None: 0,
            np.nan: 0,
        }

    # ----- overwritten methods -----
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

    def _get_traced_bids(self) -> list[BodyId]:
        """
        Get the body ids of the traced neurons.
        """
        nodes_data = pd.read_feather(
            self._nodes_file, columns=[self._body_id, self._tracing_status]
        )
        nodes_data = nodes_data[nodes_data[self._tracing_status] == self.traced_entry]
        traced_bids = list(nodes_data[self._body_id].values)
        return traced_bids

    def _load_connections(self) -> pd.DataFrame:
        """
        Load the connections of the connectome.
        Needs to gather the columns ['start_bid', 'end_bid', 'syn_count', 'nt_type'].
        """
        # Loading data in the connections file
        columns = ["start_bid", "end_bid", "syn_count"]
        columns_to_read = [self.sna(a) for a in columns]
        connections = pd.read_feather(self._connections_file, columns=columns_to_read)
        read_columns = (
            connections.columns
        )  # ordering can be different than columns_to_read
        connections.columns = [self.decode_neuron_attribute(c) for c in read_columns]

        # add nt_type information
        data = pd.read_feather(self._nodes_file, columns=[self._body_id, self._nt_type])
        read_columns = data.columns
        data.columns = [self.decode_neuron_attribute(c) for c in read_columns]
        connections = pd.merge(
            connections,
            data,
            left_on="start_bid",
            right_on="body_id",
            how="left",
            suffixes=("", ""),
        )
        connections = connections.drop(columns=["body_id"])

        return connections

    # public methods
    def list_possible_attributes(self) -> list[str]:
        """
        List the possible attributes for a neuron.
        """
        all_attr = [
            "nt_type",
            "nt_proba",
            "class_1",
            "class_2",
            "name",
            "type",
            "side",
            "neuropil",
            "hemilineage",
            "size",
            "tracing_status",
            "entry_nerve",
            "exit_nerve",
            "position",
            "nb_pre_synapses",
            "nb_post_synapses",
            "nb_pre_neurons",
            "nb_post_neurons",
        ]
        return all_attr

    def sna(  # specific_neuron_attribute, abbreviated due to frequent use
        self, generic_n_a: NeuronAttribute
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
                "entry_nerve": self._entry_nerve,
                "exit_nerve": self._exit_nerve,
                "nb_pre_synapses": self._nb_pre_synapses,
                "nb_post_synapses": self._nb_post_synapses,
                "nb_pre_neurons": self._nb_pre_neurons,
                "nb_post_neurons": self._nb_post_neurons,
                "location": self._location,  # synapse position
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

    def decode_neuron_attribute(self, specific_attribute: str) -> NeuronAttribute:
        """
        Decode the specific attribute to the generic one.
        """
        try:
            converted_attr = super().decode_neuron_attribute(specific_attribute)
        except KeyError:
            # look for specific attributes only defined in this connectome
            mapping: dict[str, NeuronAttribute] = {
                self._type: "type",
                self._tracing_status: "tracing_status",
                self._entry_nerve: "entry_nerve",
                self._exit_nerve: "exit_nerve",
                self._nb_pre_synapses: "nb_pre_synapses",
                self._nb_post_synapses: "nb_post_synapses",
                self._nb_pre_neurons: "nb_pre_neurons",
                self._nb_post_neurons: "nb_post_neurons",
                self._location: "location",  # synapse position
            }
            try:
                converted_attr = mapping.get(specific_attribute)
                if converted_attr is None:
                    raise KeyError
            except KeyError:
                raise ValueError(
                    f"ConnectomeReader::decode_neuron_attribute().\
                    The attribute {specific_attribute} is not defined in {self.connectome_name}."
                )
        return converted_attr

    def specific_neuron_class(self, generic_n_c: NeuronClass):
        """
        Converts the generic Neuron Class to the specific one.
        """
        try:
            converted_type = super().specific_neuron_class(generic_n_c)
        except KeyError:
            # look for specific classes only defined in this connectome
            mapping = {
                "intrinsic": self._intrinsic,
                "glia": self._glia,
                "sensory_ascending": self._sensory_ascending,
                "efferent": self._efferent,
                "efferent_ascending": self._efferent_ascending,
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

    def decode_neuron_class(self, specific_class: str | None) -> NeuronClass:
        """
        Decode the specific class to the generic one.
        """
        try:
            converted_class = super().decode_neuron_class(specific_class)
        except KeyError:
            # look for specific classes only defined in this connectome
            mapping: dict[str | None, NeuronClass] = {
                self._intrinsic: "intrinsic",
                self._glia: "glia",
                self._sensory_ascending: "sensory_ascending",
                self._efferent: "efferent",
                self._efferent_ascending: "efferent_ascending",
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

    def list_all_nodes(self) -> list[BodyId]:
        """
        List all the neurons existing in the connectome.
        """
        data = pd.read_feather(self._nodes_file, columns=[self._body_id])
        return list(data[self._body_id].values)

    def get_neuron_bodyids(
        self,
        selection_dict: SelectionDict = {},
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
        columns_to_read = [self.sna(a) for a in selection_dict.keys()]
        columns_to_write = list(selection_dict.keys())
        if "body_id" not in selection_dict.keys():
            columns_to_read.append(self._body_id)  # specific name field
            columns_to_write.append("body_id")  # generic name field

        neurons = pd.read_feather(self._nodes_file, columns=list(columns_to_read))

        if s_dict is not None:
            for key in s_dict.keys():
                if key == self.sna("class_1"):
                    requested_value = s_dict[
                        key
                    ]  # can be 'sensory' or 'sensory neuron'
                    try:  # will work if a generic NeuronClass is given
                        specific_value = self.specific_neuron_class(requested_value)
                    except KeyError:  # will work if a specific NeuronClass is given
                        specific_value = requested_value
                    neurons = neurons[neurons[self._class_1] == specific_value]
                else:
                    neurons = neurons[neurons[key] == s_dict[key]]
        if nodes is not None:
            neurons = neurons[neurons[self._body_id].isin(nodes)]

        return list(neurons[self._body_id].values)

    def load_data_neuron(
        self,
        id_: BodyId | int,
        attributes: list[NeuronAttribute] = [],
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
        columns_to_read = [self.sna(a) for a in attributes]
        columns_to_write = attributes
        if "body_id" not in attributes:
            columns_to_read.append(self._body_id)  # specific name field
            columns_to_write.append("body_id")  # generic name field

        # Load data
        neurons = pd.read_feather(self._nodes_file, columns=columns_to_read)
        # rename the columns to the generic names
        read_columns = neurons.columns  # ordering can be different than columns_to_read
        neurons.columns = [self.decode_neuron_attribute(c) for c in read_columns]
        if attributes is not None:
            return neurons[neurons["body_id"] == id_][columns_to_write]
        return neurons[neurons["body_id"] == id_]

    def load_data_neuron_set(
        self,
        ids: list[BodyId] | list[int],
        attributes: list[NeuronAttribute] = [],
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
        columns_to_read = [self.sna(a) for a in attributes]
        columns_to_write = attributes
        if "body_id" not in attributes:
            columns_to_read.append(self._body_id)  # specific name field
            columns_to_write.append("body_id")  # generic name field

        # Load data
        neurons = pd.read_feather(self._nodes_file, columns=columns_to_read)
        # rename the columns to the generic names
        read_columns = neurons.columns  # ordering can be different than columns_to_read
        neurons.columns = [self.decode_neuron_attribute(c) for c in read_columns]

        # return columns in the order they were requested in attributes
        if len(attributes) > 0:
            return neurons[neurons["body_id"].isin(ids)][columns_to_write]
        return neurons[neurons["body_id"].isin(ids)]


# Specific versions of MANC
class MANC_v_1_0(MANCReader):
    def __init__(
        self, connectome_preprocessing: ConnectomePreprocessingOptions | None = None
    ):
        super().__init__("v1.0", connectome_preprocessing)

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
        self._position = "position:point{srid:9157}"  # for neurons
        self._location = "location:point{srid:9157}"  # for synapses
        self._target = "target:string"
        # specific to MANC -> need to add to ::sna()
        self._type = "type:string"
        self._tracing_status = "status:string"
        self._entry_nerve = "entryNerve:string"
        self._exit_nerve = "exitNerve:string"
        self._nb_pre_synapses = "pre:int"
        self._nb_post_synapses = "post:int"
        self._nb_pre_neurons = "upstream:int"
        self._nb_post_neurons = "downstream:int"
        # Synapse specific
        self._start_synset_id = ":START_ID(SynSet-ID)"
        self._end_synset_id = ":END_ID(SynSet-ID)"
        self._start_syn_id = ":START_ID(Syn-ID)"
        self._end_syn_id = ":END_ID(Syn-ID)"
        self._syn_id = ":ID(Syn-ID)"
        self._syn_location = "location:point{srid:9157}"

        # Existing fields in the MANC v1.0 for future reference
        """
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
        """

    def _load_data_directories(self):
        """
        Need to define the directories that are common to all connectomes.
        """
        self._connectome_dir = os.path.join(
            params.RAW_DATA_DIR,
            self.connectome_name,
            self.connectome_version,
            f"neuprint_manc_{self.connectome_version}",
            f"neuprint_manc_{self.connectome_version}_ftr",
        )
        self._connections_file = os.path.join(
            self._connectome_dir, "Neuprint_Neuron_Connections_manc_v1.ftr"
        )

        # specific to MANC
        # all information for a neuron
        self._nodes_file = os.path.join(
            self._connectome_dir, "Neuprint_Neurons_manc_v1.ftr"
        )
        # synapses: neurons to synapse sets -> synapsesets to synapses -> synapses to synapse data
        self._synapseset_file = os.path.join(
            self._connectome_dir, "Neuprint_SynapseSet_to_Synapses_manc_v1.ftr"
        )
        self._neuron_synapseset_file = os.path.join(
            self._connectome_dir, "Neuprint_Neuron_to_SynapseSet_manc_v1.ftr"
        )
        self._synapse_file = os.path.join(
            self._connectome_dir, "Neuprint_Synapses_manc_v1.ftr"
        )

    # --- specific private methods
    def _load_synapse_locations(self, synapse_ids: list[int]) -> pd.DataFrame:
        """
        Get the locations of the synapses.
        """
        data = pd.read_feather(
            self._synapse_file, columns=[self._syn_id, self._syn_location]
        )
        data = data.loc[data[self._syn_id].isin(synapse_ids)]
        read_columns = data.columns
        data.columns = [self.decode_neuron_attribute(c) for c in read_columns]

        locations = data["location"].values

        # split the location into X, Y, Z
        X, Y, Z = [], [], []
        for loc in locations:
            name = loc.replace("x", '"x"').replace("y", '"y"').replace("z", '"z"')
            try:
                pos = eval(name)  # read loc as a dict, use of x,y,z under the hood
            except TypeError:
                pos = {"x": np.nan, "y": np.nan, "z": np.nan}
                print(f"Type Error in reading location for {name}")
            except NameError:
                pos = {"x": np.nan, "y": np.nan, "z": np.nan}
                print(f"Name Error in reading location for {name}")
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

    # public methods
    def get_synapse_df(self, body_id: BodyId | int) -> pd.DataFrame:
        """
        Load the synapse ids for the neuron.
        should define the columns
        ['synapse_id','start_bid','end_bid', 'X', 'Y', 'Z']
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
        synapse_df["start_bid"] = synapse_df["synset_id"].apply(
            lambda x: int(x.split("_")[0])
        )  # body id of the presynaptic neuron
        synapse_df["end_bid"] = synapse_df["synset_id"].apply(
            lambda x: int(x.split("_")[1])
        )  # body id of the postsynaptic neuron
        synapse_df["location"] = synapse_df["synset_id"].apply(
            lambda x: x.split("_")[2]
        )  # pre or post

        # remove the synapses that belong to partner neurons
        synapse_df = synapse_df[synapse_df["location"] == "pre"]
        synapse_df.drop(columns=["location"], inplace=True)

        # add a column with 'tracing_status' of the postsynaptic neuron if it exists
        if self.exists_tracing_status():
            nodes_data = pd.read_feather(
                self._nodes_file,
                columns=[self._body_id, self._tracing_status],
            )
            read_columns = nodes_data.columns
            nodes_data.columns = [self.decode_neuron_attribute(c) for c in read_columns]
            synapse_df = synapse_df.merge(
                nodes_data, left_on="end_bid", right_on="body_id", how="left"
            )
            synapse_df.drop(columns=["body_id"], inplace=True)
            # remove the rows where the postsynaptic neuron is not traced
            synapse_df = synapse_df[synapse_df["tracing_status"] == self.traced_entry]
            synapse_df.drop(columns=["tracing_status"], inplace=True)

        # remove the rows where there are fewer than threshold synapses from
        # a presynaptic neuron to a postsynaptic neuron
        synapse_df = synapse_df.groupby(["start_bid", "end_bid"]).filter(
            lambda x: len(x) >= self.connectome_preprocessing.min_synapse_count_cutoff
        )

        # Load the synapse locations
        syn_ids = list(synapse_df["synapse_id"].values)
        data = self._load_synapse_locations(syn_ids)
        synapse_df = synapse_df.merge(data, on="synapse_id", how="inner")

        return synapse_df

    def get_synapse_neuropil(
        self,
        synapse_ids: list[int],
    ) -> pd.DataFrame:
        """
        Get the neuropil of the synapses.
        In MANC v1.0, this means finding the name of the neuropil column for which
        the entry is True.
        In v1.0, each synapse has a unique identifier, so we can directly
        use the synapse id to find the neuropil.
        """
        roi_file = os.path.join(self._connectome_dir, "all_ROIs.txt")
        rois = list(pd.read_csv(roi_file, sep="\t").values.flatten())

        # find the subset of the synapses in the dataset that we care about for this neuron
        # then for each possible ROI, check which synapses are in that ROI
        # store each synapse's ROI in the neuropil column
        data = pd.read_feather(self._synapse_file, columns=[self._syn_id])
        data.columns = ["synapse_id"]
        data["neuropil"] = "None"

        for roi in rois:
            column_name = roi + ":boolean"
            roi_column = pd.read_feather(
                self._synapse_file,
                columns=[column_name, self._syn_id],
            )
            roi_column = roi_column[roi_column[self._syn_id].isin(synapse_ids)]
            synapses_in_roi = roi_column.loc[
                roi_column[column_name] == True, self._syn_id
            ].values  # type: ignore
            data.loc[data["synapse_id"].isin(synapses_in_roi), "neuropil"] = roi
        return data


class MANC_v_1_2(MANCReader):
    def __init__(
        self, connectome_preprocessing: ConnectomePreprocessingOptions | None = None
    ):
        super().__init__("v1.2", connectome_preprocessing)

    # ----- overwritten methods -----
    def _load_specific_namefields(self):
        self._body_id = "bodyId"
        self._start_bid = "bodyId_pre"
        self._end_bid = "bodyId_post"
        self._syn_count = "weight"
        self._nt_type = "predictedNt"
        self._nt_proba = "predictedNtProb"
        self._class_1 = "class"
        self._class_2 = "subclass"
        self._name = "systematicType"
        self._side = "somaSide"
        self._neuropil = "somaNeuromere"
        self._hemilineage = "hemilineage"
        self._size = "size"
        self._position = "location"  # for neurons
        # self._location = "location:point{srid:9157}"  # for synapses
        self._target = "target"

        # specific to MANC -> need to add to ::sna()
        self._type = "type"
        self._tracing_status = "status"
        self._entry_nerve = "entryNerve"
        self._exit_nerve = "exitNerve"
        self._nb_pre_synapses = "pre"
        self._nb_post_synapses = "post"
        self._nb_pre_neurons = "upstream"
        self._nb_post_neurons = "downstream"

        # Synapse specific
        self._syn_id = "synapse_id"
        self._synapse_x = "x_pre"
        self._synapse_y = "y_pre"
        self._synapse_z = "z_pre"
        self._syn_neuropil = "roi_pre"

    def _load_specific_neuron_classes(self):
        """
        Name the neuron classes.
        """
        MANCReader._load_specific_neuron_classes(self)
        # MANC 1.2 is the same as MANC 1.0 except Glia was renamed to glia
        self._glia = "glia"

    def _load_data_directories(self):
        """
        Need to define the directories that are common to all connectomes.
        """
        self._connectome_dir = os.path.join(
            params.RAW_DATA_DIR,
            self.connectome_name,
            self.connectome_version,
        )

        self._nodes_file = os.path.join(self._connectome_dir, "neurons.ftr")
        self._synapses_file = os.path.join(self._connectome_dir, "synapses.ftr")
        self._connections_file = os.path.join(self._connectome_dir, "connections.ftr")

    # --- specific private methods
    def _load_synapse_locations(self, synapse_ids: list[int]) -> pd.DataFrame:
        """
        Get the locations of the synapses.
        """
        synapses = pd.read_feather(self._synapses_file)

        # create synapse ids from the index
        synapses = synapses.loc[synapses[self._syn_id].isin(synapse_ids)]

        synapses = synapses[
            [
                self._syn_id,
                self._synapse_x,
                self._synapse_y,
                self._synapse_z,
            ]
        ]
        synapses.columns = ["synapse_id", "X", "Y", "Z"]
        return synapses

    # public methods
    def get_synapse_df(self, body_id: BodyId | int) -> pd.DataFrame:
        """
        Load the synapse ids for the neuron.
        should define the columns
        ['synapse_id','start_bid','end_bid', 'X', 'Y', 'Z']
        """
        # synapse set to synapse
        synapses = pd.read_feather(self._synapses_file)

        # filter on the presynaptic neuron
        synapses = synapses.loc[synapses[self._start_bid] == body_id]

        synapses = synapses[
            [
                self._syn_id,
                self._start_bid,
                self._end_bid,
                self._synapse_x,
                self._synapse_y,
                self._synapse_z,
            ]
        ]
        synapses.columns = ["synapse_id", "start_bid", "end_bid", "X", "Y", "Z"]

        # filter out non traced postsynaptic neurons
        # add a column with 'tracing_status' of the postsynaptic neuron if it exists
        if self.exists_tracing_status():
            nodes_data = pd.read_feather(
                self._nodes_file,
                columns=[self._body_id, self._tracing_status],
            )
            read_columns = nodes_data.columns
            nodes_data.columns = [self.decode_neuron_attribute(c) for c in read_columns]
            synapses = synapses.merge(
                nodes_data, left_on="end_bid", right_on="body_id", how="left"
            )
            synapses.drop(columns=["body_id"], inplace=True)
            # remove the rows where the postsynaptic neuron is not traced
            synapses = synapses.loc[synapses["tracing_status"] == self.traced_entry]
            synapses.drop(columns=["tracing_status"], inplace=True)

        # remove the rows where there are fewer than threshold synapses from
        # a presynaptic neuron to a postsynaptic neuron
        synapses = synapses.groupby(["start_bid", "end_bid"]).filter(
            lambda x: len(x) >= self.connectome_preprocessing.min_synapse_count_cutoff
        )
        return synapses

    def get_synapse_neuropil(
        self,
        synapse_ids: list[int],
    ) -> pd.DataFrame:
        """
        Get the neuropil of the synapses.
        In MANC v1.2, the synapse id is unique only for a given presynaptic neuron,
        so we need to filter on the presynaptic neuron as well.
        """
        synapses = pd.read_feather(self._synapses_file)

        # filter on synapse_ids
        synapses = synapses.loc[synapses[self._syn_id].isin(synapse_ids)]

        synapses = synapses[
            [
                self._syn_id,
                self._syn_neuropil,
            ]
        ]
        synapses.columns = ["synapse_id", "neuropil"]
        return synapses


@typing.overload
def MANC(
    version: typing.Literal["v1.0"],
    connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
) -> MANC_v_1_0: ...


@typing.overload
def MANC(
    version: typing.Literal["v1.2"],
    connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
) -> MANC_v_1_2: ...


def MANC(
    version: typing.Literal["v1.0", "v1.2"],
    connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
) -> MANCReader:
    """Get a connectome reader for one of the versions of the Male Adult Neuronal Connectome (MANC).

    Args:
        version (typing.Literal["v1.0", "v1.2"]): The two valid versions of MANC

    Raises:
        ValueError: If an incorrect connectome version is provided

    Returns:
        MANCReader: Either MANC_v_1_0 or MANC_v_1_2
    """
    match version:
        case "v1.0":
            return MANC_v_1_0(connectome_preprocessing)
        case "v1.2":
            return MANC_v_1_2(connectome_preprocessing)
    raise ValueError(
        f"Version {version} is not a supported version for the MANC connectome. Supported versions are v1.0 and v1.2"
    )


# === FAFB: Female Adult Fly Brain
class FAFBReader(ConnectomeReader):
    def __init__(
        self,
        connectome_version: str,
        connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
    ):
        super().__init__("fafb", connectome_version, connectome_preprocessing)

        self.nt_weights = {
            "ACH": +1,
            "GABA": -1,
            "GLUT": -1 if self.connectome_preprocessing.glutamate_inhibitory else +1,
            "DA": 0,
            "OCT": 0,
            "SER": 0,
            None: 0,
            np.nan: 0,
        }

    # ----- overwritten methods -----
    def _load_specific_namefields(self):
        """
        Need to define the fields that are common to all connectomes.
        BodyId, start_bid, end_bid, syn_count, nt_type, class_1, class_2, neuron_attributes
        """

        # common to all
        self._body_id = "root_id"
        self._syn_id = "synapse_id"
        self._start_bid = "pre_root_id"
        self._end_bid = "post_root_id"
        self._syn_count = "syn_count"
        self._nt_type = "nt_type"
        self._nt_proba = "nt_type_score"
        self._class_1 = "super_class"
        self._class_2 = "class"
        self._name = "cell_type"  # this is not ideal but mostly matches
        self._side = "side"
        self._neuropil = "group"
        self._hemilineage = "hemilineage"
        self._size = "size_nm"
        self._position = "position"
        self._target = "sub_class"
        # specific to FAFB -> need to add to ::sna()
        self._nerve = "nerve"
        self._area = "area_nm"
        self._length = "length_nm"
        self._flow = "flow"

    def _load_specific_neuron_classes(self):
        """
        Name the neuron classes. Map internal variable to data set names.
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
        self._visual_centrifugal = "visual_centrifugal"
        self._visual_projection = "visual_projection"
        self._other = "other"

    def _load_data_directories(self):
        """
        Need to define the directories that are common to all connectomes.
        """
        self._connectome_dir = os.path.join(
            params.RAW_DATA_DIR,
            self.connectome_name,
            self.connectome_version,
        )
        self._connections_file = os.path.join(self._connectome_dir, "connections.csv")

        # specific to FAFB
        # all information for a neuron
        self._node_stats_file = os.path.join(  # length, area, size
            self._connectome_dir, "cell_stats.csv"
        )
        self._node_class_file = os.path.join(  # class, hemilineage, side etc.
            self._connectome_dir, "classification.csv"
        )
        self._node_position_file = os.path.join(  # position = [x y z]
            self._connectome_dir, "coordinates.csv"
        )
        self._node_nt_type_file = os.path.join(  # nt_type, nt_type_score
            self._connectome_dir, "neurons.csv"
        )
        # synapses
        self._synapses_file = os.path.join(  # pre_bid, post_bid, x, y, z
            self._connectome_dir, "synapse_coordinates.csv"
        )

    def _get_traced_bids(self) -> list[BodyId]:
        """
        Get the body ids of the traced neurons.
        """
        raise NotImplementedError("Method should not be called on FAFB.")

    def _load_connections(self) -> pd.DataFrame:
        """
        Load the connections of the connectome.
        Needs to gather the columns ['start_bid', 'end_bid', 'syn_count', 'nt_type'].
        """
        columns: list[NeuronAttribute] = [
            "start_bid",
            "end_bid",
            "syn_count",
            "nt_type",
        ]
        columns_to_read = [self.sna(a) for a in columns]
        # here the root_id length is not an issue because there are mixed types
        # in the dataframe. In practice that means that the root_ids are parsed
        # first, and converted to int only if they are integers, which is fine.
        connections = pd.read_csv(self._connections_file, usecols=columns_to_read)
        read_columns = connections.columns
        connections.columns = [self.decode_neuron_attribute(c) for c in read_columns]
        return connections

    # --- specific private methods
    def _filter_neurons(
        self,
        attribute: NeuronAttribute,
        value,
    ) -> set[BodyId]:
        """
        Return the set of neuron body_ids for which the attribute has the value.
        """
        if attribute == "body_id":
            return {value}

        att = self.sna(attribute)  # specific name attribute

        def _simply_filter_df(filename: str, att: str, value) -> set[BodyId]:
            data = pd.read_csv(
                filename,
                usecols=[self._body_id, att],
                dtype={self._body_id: str},  # in case there are only ints in the file
            )
            data = data[data[att] == value]
            return set(data[self._body_id].astype(int).values)

        if attribute in ["nt_type", "nt_proba", "neuropil"]:
            filename = self._node_nt_type_file
            valid_nodes = _simply_filter_df(filename, att, value)
        elif attribute in ["length", "area", "size"]:
            filename = self._node_stats_file
            valid_nodes = _simply_filter_df(filename, att, value)
        elif attribute == "position":
            filename = self._node_position_file
            valid_nodes = _simply_filter_df(filename, att, value)
        else:
            filename = self._node_class_file
            if attribute == "class_1":
                # need to check both the generic and the specific names
                global_match = _simply_filter_df(filename, att, value)
                specific_match = _simply_filter_df(
                    filename, att, self.specific_neuron_class(value)
                )
                valid_nodes = global_match.union(specific_match)
            else:
                valid_nodes = _simply_filter_df(filename, att, value)

        return valid_nodes

    # public methods
    def get_synapse_df(self, body_id: BodyId | int) -> pd.DataFrame:
        """
        Load the synapse ids for the neuron.
        should define the columns
        ['synapse_id','start_bid','end_bid', 'X', 'Y', 'Z']
        """
        type_dict = {  # the rootids are so long that they are corrupted upon reading
            # need to first read them as a string before converting to int
            "pre_root_id": str,
            "post_root_id": str,
            "x": int,
            "y": int,
            "z": int,
        }
        all_synapses = pd.read_csv(self._synapses_file, dtype=type_dict)
        all_synapses.columns = ["start_bid", "end_bid", "X", "Y", "Z"]  # file order
        all_synapses.ffill(inplace=True)  # fill the NaNs with the previous value
        all_synapses["synapse_id"] = (
            all_synapses.index
        )  # do before filtering for potential comparison across bodyids
        all_synapses = all_synapses.astype({"start_bid": int, "end_bid": int})

        # filter for the synapses of the neuron
        synapses = all_synapses[all_synapses["start_bid"] == body_id]
        return synapses

    def get_synapse_neuropil(
        self,
        synapse_ids: list[int],
    ) -> pd.DataFrame:
        raise NotImplementedError(
            'No trivial match in FAFB between coordinates and neuropil...\
            You can try to circumvent with the number of synapses in a neuropil for a given neuron.\
            For that, load "neuropil_synapse_table.csv"'
        )

    def sna(self, generic_n_a: NeuronAttribute) -> str:
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

    def decode_neuron_attribute(self, specific_attribute: str) -> NeuronAttribute:
        """
        Decode the specific attribute to the generic one.
        """
        try:
            converted_attr = super().decode_neuron_attribute(specific_attribute)
        except KeyError:
            # look for specific attributes only defined in this connectome
            mapping: dict[str, NeuronAttribute] = {
                self._nerve: "nerve",
                self._area: "area",
                self._length: "length",
                self._flow: "flow",
            }
            try:
                converted_attr = mapping.get(specific_attribute)
                if converted_attr is None:
                    raise KeyError
            except KeyError:
                raise ValueError(
                    f"ConnectomeReader::decode_neuron_attribute().\
                    The attribute {specific_attribute} is not defined in {self.connectome_name}."
                )
        return converted_attr

    def specific_neuron_class(self, generic_n_c: NeuronClass):
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
                "visual_centrifugal": self._visual_centrifugal,
                "visual_projection": self._visual_projection,
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

    def decode_neuron_class(self, specific_class: str | None) -> NeuronClass:
        """
        Decode the specific class to the generic one.
        """
        try:
            converted_class = super().decode_neuron_class(specific_class)
        except KeyError:
            # look for specific classes only defined in this connectome
            mapping: dict[str | None, NeuronClass] = {
                self._central: "central",
                self._endocrine: "endocrine",
                self._optic: "optic",
                self._visual_centrifugal: "visual_centrifugal",
                self._visual_projection: "visual_projection",
                self._other: "other",
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

    def list_possible_attributes(self):
        """
        List the possible attributes of the dataset.
        """
        all_attr = [
            "nt_type",
            "nt_proba",
            "class_1",
            "class_2",
            "side",
            "neuropil",
            "hemilineage",
            "size",
            "nerve",
            "area",
            "length",
            "flow",
        ]
        return all_attr

    def list_all_nodes(self) -> list[BodyId]:
        """
        List all the neurons existing in the connectome.
        """
        data = pd.read_csv(
            self._node_stats_file,
            usecols=[self._body_id],
            dtype={self._body_id: str},  # in case there are only ints in the file
        )
        return list(data[self._body_id].astype(int).values)

    def get_neuron_bodyids(
        self,
        selection_dict: SelectionDict = {},
        nodes: Optional[list[BodyId] | list[int]] = None,
    ) -> list[BodyId]:
        """
        Get the Ids of the neurons in the dataset.
        Select (keep) according to the selection_dict.
        Different criteria are treated as 'and' conditions.

        For the specific case of "class_1" that refers to the NeuronClass,
        we need to verify both the generic and the specific names.
        """

        # Treat each attribute in the selection dict independently:
        # get the nodes that satisfy each condition, and return the intersection of all
        if nodes is not None:
            valid_nodes = set(nodes)
        else:
            valid_nodes = set(self.list_all_nodes())
        for key, value in selection_dict.items():
            specific_valid_nodes = self._filter_neurons(
                attribute=key,
                value=value,
            )
            valid_nodes = valid_nodes.intersection(specific_valid_nodes)

        return list(valid_nodes)

    def load_data_neuron(
        self,
        id_: BodyId | int,
        attributes: list[NeuronAttribute] = [],
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
        return self.load_data_neuron_set([id_], attributes)

    def load_data_neuron_set(
        self,
        ids: list[BodyId] | list[int],
        attributes: list[NeuronAttribute] = [],
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

        def _load_df(
            filename: str,
            columns: list[NeuronAttribute],
            bids: list[BodyId] | list[int],
        ) -> pd.DataFrame:
            if columns is None or len(columns) == 0:
                raise ValueError("No columns to load.")
            columns.append("body_id")
            columns_to_read = [self.sna(a) for a in columns]
            data = pd.read_csv(
                filename,
                usecols=columns_to_read,
                dtype={self._body_id: str},  # in case there are only ints in the file
            )
            loaded_columns = data.columns  # does not respect our ordering
            data.columns = [self.decode_neuron_attribute(c) for c in loaded_columns]
            data["body_id"] = data["body_id"].astype(int)
            data = data[data["body_id"].isin(bids)]
            return data

        # Load data
        neurons = pd.DataFrame({"body_id": ids})
        # split the data loading as a function of the files required
        # 1. nt_type, nt_proba, neuropil
        nt_fields = list(
            set(attributes).intersection({"nt_type", "nt_proba", "neuropil"})
        )
        if len(nt_fields) > 0:
            data = _load_df(self._node_nt_type_file, nt_fields, ids)
            neurons = neurons.merge(data, on="body_id", how="inner")
        # 2. length, area, size
        stat_fields = list(set(attributes).intersection({"length", "area", "size"}))
        if len(stat_fields) > 0:
            data = _load_df(self._node_stats_file, stat_fields, ids)
            neurons = neurons.merge(data, on="body_id", how="inner")
        # 3. position
        if "position" in attributes:
            data = _load_df(self._node_position_file, ["position"], ids)
            neurons = neurons.merge(data, on="body_id", how="inner")
        # 4. class, hemilineage, side etc.
        class_fields = list(
            set(attributes).difference(
                set(nt_fields).union(set(stat_fields)).union({"position"})
            )
        )
        if len(class_fields) > 0:
            data = _load_df(self._node_class_file, class_fields, ids)
            neurons = neurons.merge(data, on="body_id", how="inner")

        if attributes is not None:
            if "body_id" not in attributes:
                attributes.append("body_id")
            return neurons[neurons["body_id"].isin(ids)][attributes]
        return neurons[neurons["body_id"].isin(ids)]


# Specific versions of FAFB
class FAFB_v630(FAFBReader):
    def __init__(
        self, connectome_preprocessing: ConnectomePreprocessingOptions | None = None
    ):
        super().__init__("v630", connectome_preprocessing)


class FAFB_v783(FAFBReader):
    def __init__(
        self, connectome_preprocessing: ConnectomePreprocessingOptions | None = None
    ):
        super().__init__("v783", connectome_preprocessing)


@typing.overload
def FAFB(
    version: typing.Literal["v630"],
    connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
) -> FAFB_v630: ...


@typing.overload
def FAFB(
    version: typing.Literal["v783"],
    connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
) -> FAFB_v783: ...


def FAFB(
    version: typing.Literal["v630", "v783"],
    connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
) -> FAFBReader:
    """Get a connectome reader for one of the versions of the Full Adult Fly Brain connectome (FAFB).

    Args:
        version (typing.Literal["v630", "v783"]): The two valid versions of FAFB

    Raises:
        ValueError: If an incorrect connectome version is provided

    Returns:
        FAFBReader: Either FAFB_v630 or FAFB_v783
    """
    match version:
        case "v630":
            return FAFB_v630(connectome_preprocessing)
        case "v783":
            return FAFB_v783(connectome_preprocessing)
    raise ValueError(
        f"Version {version} is not a supported version for the FAFB connectome. Supported versions are v630 and v783"
    )

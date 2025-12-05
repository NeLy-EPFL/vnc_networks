#!/usr/bin/env python3
"""
Definition of the ConnectomeReader class.
Class defining the methods and labels to read different connectomes.

Each instance also redefines the generic NeuronAttribute and NeuronClass
data types to the specific ones of the connectome.
"""

from __future__ import annotations

import ast
import os
import typing
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from typing import Any, Optional

import numpy as np
import polars as pl
import polars.selectors as cs
from bidict import bidict

from . import params
from .params import (
    BodyId,
    ConnectomePreprocessingOptions,
    NeuronAttribute,
    NeuronClass,
    SelectionDict,
)


def default_connectome_reader() -> ConnectomeReader:
    return MANC("v1.2.3")


# --- Parent class, abstract --- #
class ConnectomeReader(ABC):
    # Names of common neuron attributes
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
    _target = "target"

    # Names of common neuron classes
    # these get overwritten by custom connectomes which use different names
    _sensory = "sensory"
    _motor = "motor"
    _ascending = "ascending"
    _descending = "descending"

    # connectomes need to implement weight assignment for their neurotransmitters
    nt_weights: Mapping[Any, int]

    # how to map from attributes and classes specific to each connectome to generic ones
    # bidict lets us go from generic->specific as well as specific->generic (using .inverse property)
    generic_to_specific_attribute: bidict[NeuronAttribute, str]
    """Maps between a generic (common) neuron attribute and a neuron attribute specific to a particular connectome"""
    generic_to_specific_class: bidict[NeuronClass, str]
    """Maps between a generic (common) neuron class and a neuron class specific to a particular connectome"""

    att_map: bidict[NeuronAttribute, str]

    # common properties to all connectomes
    _connectome_dir: str

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

        # fail early if the connectome data folder doesn't exist
        self._load_data_directories()
        if not os.path.exists(self._connectome_dir):
            raise FileNotFoundError(
                f"Could't find {self.connectome_name.upper()} {self.connectome_version} data in {self._connectome_dir}."
            )

        # specific namefields
        self._load_specific_neuron_attributes()
        self._load_specific_neuron_classes()
        self._build_attribute_mapping()
        self._build_class_mapping()
        self._create_output_directories()

    def _build_attribute_mapping(self):
        self.generic_to_specific_attribute = bidict(
            {
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
        )

    def _build_class_mapping(self):
        self.generic_to_specific_class = bidict(
            {
                "sensory": self._sensory,
                "ascending": self._ascending,
                "motor": self._motor,
                "descending": self._descending,
            }
        )

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
    def _load_specific_neuron_attributes(self): ...

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
    def _load_connections(self) -> pl.DataFrame:
        """
        Load the connections of the connectome.
        """
        ...

    # ----- public methods -----
    # --- abstract methods
    @abstractmethod
    def get_synapse_df(self, body_id: BodyId | int) -> pl.DataFrame:
        """
        Get the locations of the synapses.
        """
        ...

    @abstractmethod
    def get_synapse_neuropil(
        self,
        synapse_ids: list[int],
    ) -> pl.DataFrame:
        """
        Get the neuropil of the synapses.
        """
        ...

    @abstractmethod
    def get_synapse_counts_by_neuropil(
        self,
        synapse_count_type: typing.Literal[
            "downstream", "upstream", "pre", "post", "total_synapses"
        ],
        body_id_subset: list[BodyId] | list[int] | None = None,
    ) -> pl.DataFrame:
        """
        Get neuron or synapse counts for each neuron in each neuropil

        Args:
            synapse_count_type (typing.Literal[ "downstream", "upstream", "pre", "post", "total_synapses"]): which count to get
                * `"downstream"` number of downstream neurons
                * `"upstream"` number of upstream neurons
                * `"pre"` number of presynaptic synapses (ie. synapses to upstream neurons)
                * `"post"` number of postsynaptic synapses (ie. synapses to downstream neurons)
                * `"total_synapses"` total number of synapses, sum of pre and post

            body_id_subset (list[BodyId] | list[int] | None, optional): Only return counts for a certain set of neurons.
                If None, return counts for all. Defaults to None.

        Returns:
            polars.DataFrame: a table [body_id, rois...] with the counts for each neuropil for each body_id.
                **Note:** ROI columns won't be returned if no neurons have a count in that column (ie. if specifying
                a small number of neurons for `body_id_subset`).
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
        Get the BodyIds of the neurons in the dataset that fulfil the conditions in the selection_dict.

        For the specific case of "class_1" that refers to the NeuronClass, we need to verify both the generic and the specific names.

        Args:
            selection_dict (SelectionDict, optional): Criteria that the returned neurons need to fulfil. Different criteria are treated as 'and' conditions. Defaults to {}.
            nodes (Optional[list[BodyId]  |  list[int]], optional): If not None, only return BodyIds which are contained in this list. Defaults to None.

        Returns:
            list[BodyId]: list of the BodyIds of neurons that fulfilled all supplied conditions.
        """
        ...

    @abstractmethod
    def load_data_neuron(
        self,
        id_: BodyId | int,
        attributes: list[NeuronAttribute] = [],
    ) -> pl.DataFrame:
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
        polars.DataFrame
            The data of the neuron.
        """
        ...

    @abstractmethod
    def load_data_neuron_set(
        self,
        ids: list[BodyId] | list[int],
        attributes: list[NeuronAttribute] = [],
    ) -> pl.DataFrame:
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
        polars.DataFrame
            The data of the neurons.
        """
        ...

    # --- common methods
    def sna(self, generic_attribute: NeuronAttribute) -> str:
        """
        Returns the specific Neuron Attribute defined for the connectome.
        Example: the generic 'class_1' input will return 'class:string' for MANC
        and 'super_class' for FAFB.

        Extra attributes can be added by child classes modifying the mapping.
        """
        try:
            return self.generic_to_specific_attribute[generic_attribute]
        except KeyError:
            raise ValueError(
                f"ConnectomeReader::sna().\
                The attribute {generic_attribute} is not defined in {self.connectome_name} {self.connectome_version}."
            )

    def decode_neuron_attribute(self, specific_attribute: str) -> NeuronAttribute:
        """
        Decode the specific attribute to the generic one.
        """
        try:
            return self.generic_to_specific_attribute.inverse[specific_attribute]
        except KeyError:
            raise ValueError(
                f"ConnectomeReader::decode_neuron_attribute().\
                The attribute {specific_attribute} is not defined in {self.connectome_name} {self.connectome_version}."
            )

    def specific_neuron_class(self, generic_class: NeuronClass):
        """
        Returns the specific Neuron Class defined for the connectome.
        """
        try:
            return self.generic_to_specific_class[generic_class]
        except KeyError:
            raise ValueError(
                f"ConnectomeReader::specific_neuron_class().\
                The attribute {generic_class} is not defined in {self.connectome_name} {self.connectome_version}."
            )

    def decode_neuron_class(self, specific_class: str | None) -> NeuronClass:
        """
        Decode the specific class to the generic one.
        If the specific class is None, it will be mapped to "unknown"
        """
        if specific_class is None:
            return "unknown"
        try:
            return self.generic_to_specific_class.inverse[specific_class]
        except KeyError:
            raise ValueError(
                f"ConnectomeReader::decode_neuron_class().\
                The attribute {specific_class} is not defined in {self.connectome_name} {self.connectome_version}."
            )

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
            df = df.filter(
                pl.col("start_bid").is_in(traced_bids)
                & pl.col("end_bid").is_in(traced_bids)
            )

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
        return self.get_neuron_bodyids({"class_1": specific_class})

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

    def list_possible_attributes(self) -> list[NeuronAttribute]:
        """
        List the possible attributes for a neuron.
        """
        return list(self.generic_to_specific_attribute.keys())

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
    _root_side: str

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
        Name the neuron attributes and classes.
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

    def _build_attribute_mapping(self):
        super()._build_attribute_mapping()
        self.generic_to_specific_attribute.update(
            {
                "type": self._type,
                "tracing_status": self._tracing_status,
                "entry_nerve": self._entry_nerve,
                "exit_nerve": self._exit_nerve,
                "nb_pre_synapses": self._nb_pre_synapses,
                "nb_post_synapses": self._nb_post_synapses,
                "nb_pre_neurons": self._nb_pre_neurons,
                "nb_post_neurons": self._nb_post_neurons,
                "root_side": self._root_side,
            }
        )

    def _build_class_mapping(self):
        super()._build_class_mapping()

        self.generic_to_specific_class.update(
            {
                "vnc_intrinsic": self._intrinsic,
                "glia": self._glia,
                "sensory_ascending": self._sensory_ascending,
                "efferent": self._efferent,
                "efferent_ascending": self._efferent_ascending,
                "unknown": self._unknown,
                "sensory_unknown": self._sensory_unknown,
                "interneuron_unknown": self._interneuron_unknown,
            }
        )

    def _get_traced_bids(self) -> list[BodyId]:
        """
        Get the body ids of the traced neurons.
        """
        return (
            pl.scan_ipc(self._nodes_file)
            .filter(pl.col(self._tracing_status) == self.traced_entry)
            .select(self._body_id)
            .collect()
            .get_column(self._body_id)
            .to_list()
        )

    def _load_connections(self) -> pl.DataFrame:
        """
        Load the connections of the connectome.
        Needs to gather the columns ['start_bid', 'end_bid', 'syn_count', 'nt_type'].
        """
        return (
            (
                # Loading data in the connections file
                pl.scan_ipc(
                    self._connections_file,
                )
                .select(self._start_bid, self._end_bid, self._syn_count)
                .rename(self.decode_neuron_attribute)
            )
            .join(
                # add nt_type information
                pl.scan_ipc(
                    self._nodes_file,
                )
                .select(self._body_id, self._nt_type)
                .rename(self.decode_neuron_attribute),
                left_on="start_bid",
                right_on="body_id",
                how="left",
            )
            .collect()
        )

    # public methods
    def list_all_nodes(self) -> list[BodyId]:
        """
        List all the neurons existing in the connectome.
        """
        return (
            pl.read_ipc(self._nodes_file, columns=[self._body_id], memory_map=False)
            .get_column(self._body_id)
            .to_list()
        )

    def get_neuron_bodyids(
        self,
        selection_dict: SelectionDict = {},
        nodes: Optional[list[BodyId] | list[int]] = None,
    ) -> list[BodyId]:
        """
        Get the BodyIds of the neurons in the dataset that fulfil the conditions in the selection_dict.

        For the specific case of "class_1" that refers to the NeuronClass, we need to verify both the generic and the specific names.

        Args:
            selection_dict (SelectionDict, optional): Criteria that the returned neurons need to fulfil. Different criteria are treated as 'and' conditions. Defaults to {}.
            nodes (Optional[list[BodyId]  |  list[int]], optional): If not None, only return BodyIds which are contained in this list. Defaults to None.

        Returns:
            list[BodyId]: list of the BodyIds of neurons that fulfilled all supplied conditions.
        """
        s_dict = self.specific_selection_dict(selection_dict)

        neurons = pl.scan_ipc(self._nodes_file)

        for key in s_dict.keys():
            if key == self.sna("class_1"):
                requested_value = s_dict[key]  # can be 'sensory' or 'sensory neuron'
                try:  # will work if a generic NeuronClass is given
                    specific_value = self.specific_neuron_class(requested_value)  # type: ignore requested_value might be a generic NeuronClass, or if not a specific class already
                except ValueError:  # will work if a specific NeuronClass is given
                    specific_value = requested_value
                neurons = neurons.filter(pl.col(self._class_1) == specific_value)
            else:
                neurons = neurons.filter(pl.col(key) == s_dict[key])
        if nodes is not None:
            neurons = neurons.filter(pl.col(self._body_id).is_in(nodes))

        return neurons.collect().get_column(self._body_id).to_list()

    def load_data_neuron(
        self,
        id_: BodyId | int,
        attributes: list[NeuronAttribute] = [],
    ) -> pl.DataFrame:
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
        polars.DataFrame
            The data of the neuron.
        """
        if "body_id" not in attributes:
            attributes.append("body_id")

        # Identify columns to load
        columns_to_read = [self.sna(a) for a in attributes]

        # Load data
        return (
            pl.scan_ipc(self._nodes_file)
            .select(columns_to_read)
            .rename(self.decode_neuron_attribute)
            .filter(body_id=id_)
            .collect()
        )

    def load_data_neuron_set(
        self,
        ids: list[BodyId] | list[int],
        attributes: list[NeuronAttribute] = [],
    ) -> pl.DataFrame:
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
        polars.DataFrame
            The data of the neurons.
        """
        if "body_id" not in attributes:
            attributes.append("body_id")

        # Identify columns to load
        columns_to_read = [self.sna(a) for a in attributes]

        # Load data
        return (
            pl.scan_ipc(self._nodes_file)
            .select(columns_to_read)
            .rename(self.decode_neuron_attribute)
            .filter(pl.col("body_id").is_in(ids))
            .collect()
        )


# Specific versions of MANC
class MANC_v_1_0(MANCReader):
    def __init__(
        self, connectome_preprocessing: ConnectomePreprocessingOptions | None = None
    ):
        super().__init__("v1.0", connectome_preprocessing)

    # ----- overwritten methods -----
    def _load_specific_neuron_attributes(self):
        """
        Need to define the attribute naming conventions.
        """
        # rename common attributes
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
        self._target = "target:string"
        # attributes specific to MANC
        self._type = "type:string"
        self._tracing_status = "status:string"
        self._entry_nerve = "entryNerve:string"
        self._exit_nerve = "exitNerve:string"
        self._nb_pre_synapses = "pre:int"
        self._nb_post_synapses = "post:int"
        self._nb_pre_neurons = "upstream:int"
        self._nb_post_neurons = "downstream:int"
        self._root_side = "rootSide:string"
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
    def _load_synapse_locations(self, synapse_ids: Collection[int]) -> pl.DataFrame:
        """
        Get the locations of the synapses.

        Returns a dataframe with columns: synapse_id, X, Y, Z
        """
        return (
            pl.scan_ipc(self._synapse_file)
            .filter(pl.col(self._syn_id).is_in(synapse_ids))
            .with_columns(
                pl.col(self._syn_location)
                # location column looks like {x:25381, y:12390, z:24749}
                # need to wrap x, y, and z in "" so it's valid json
                .str.replace_all("(.):", '"${1}":')
                .str.json_decode(
                    pl.Struct(
                        [
                            pl.Field("x", pl.Int32),
                            pl.Field("y", pl.Int32),
                            pl.Field("z", pl.Int32),
                        ]
                    )
                )
                .struct.unnest()
            )
            .rename({self._syn_id: "synapse_id", "x": "X", "y": "Y", "z": "Z"})
            .select("synapse_id", "X", "Y", "Z")
        ).collect()

    # public methods
    def get_synapse_df(self, body_id: BodyId | int) -> pl.DataFrame:
        """
        Load the synapse ids for the neuron.
        should define the columns
        ['synapse_id','start_bid','end_bid', 'X', 'Y', 'Z']
        """
        # neuron to synapse set
        synset_list = (
            (
                pl.scan_ipc(self._neuron_synapseset_file, memory_map=False)
                .filter(pl.col(self._start_bid) == body_id)
                .select(pl.col(self._end_synset_id))
            )
            .collect()
            .get_column(self._end_synset_id)
        )

        # build a dataframe with the start and end neuron ids for each synapse.
        # We get this by parsing the synset_id for this synapse which has the format
        # `<start_bid>_<end_bid>_<pre/post>`.
        # We just care the presynaptic neurons that belong to this neuron
        synapse_df = (
            pl.scan_ipc(self._synapseset_file, memory_map=False)
            .filter(pl.col(self._start_synset_id).is_in(synset_list))
            .rename(
                {self._end_syn_id: "synapse_id", self._start_synset_id: "synset_id"}
            )
            .with_columns(
                start_bid=pl.col("synset_id").str.split("_").list.get(0).cast(pl.Int64),
                end_bid=pl.col("synset_id").str.split("_").list.get(1).cast(pl.Int64),
            )
            .filter(pl.col("synset_id").str.split("_").list.get(2) == "pre")
            .select("synapse_id", "start_bid", "end_bid")
        )

        # add a column with 'tracing_status' of the postsynaptic neuron if it exists
        if self.exists_tracing_status():
            nodes_data = (
                pl.scan_ipc(
                    self._nodes_file,
                    memory_map=False,
                )
                .select(self._body_id, self._tracing_status)
                .rename(self.decode_neuron_attribute)
            )
            synapse_df = (
                synapse_df.join(
                    nodes_data, left_on="end_bid", right_on="body_id", how="left"
                )
                # remove the rows where the postsynaptic neuron is not traced
                .filter(tracing_status=self.traced_entry)
                .drop(["tracing_status"])
            )

        # remove the rows where there are fewer than threshold synapses from
        # a presynaptic neuron to a postsynaptic neuron
        synapse_df = synapse_df.filter(
            pl.len().over("start_bid", "end_bid")
            >= self.connectome_preprocessing.min_synapse_count_cutoff
        )

        # Load the synapse locations
        synapse_df = synapse_df.collect()
        data = self._load_synapse_locations(synapse_df["synapse_id"])
        synapse_df = synapse_df.join(data, on="synapse_id", how="inner")

        return synapse_df

    def get_synapse_neuropil(
        self,
        synapse_ids: Collection[int],
    ) -> pl.DataFrame:
        """
        Get the neuropil of the synapses.
        In MANC v1.0, this means finding the name of the neuropil column for which
        the entry is True.
        In v1.0, each synapse has a unique identifier, so we can directly
        use the synapse id to find the neuropil.
        """
        # the synapse file looks like this
        # |  synapse_id | ROI_1 | ROI_2 | ROI_3 | ... |
        # | 99000000001 | False |  True | False | ... |
        # | 99000000002 |  True | False | False | ... |
        # | 99000000003 | False | False |  True | ... |
        #
        # we want to convert it to the following format
        # |  synapse_id | neuropil |
        # | 99000000001 |    ROI_2 |
        # | 99000000002 |    ROI_1 |
        # | 99000000003 |    ROI_3 |

        # all the ROI column headers are boolean type
        roi_columns = [
            name
            for name in pl.scan_ipc(self._synapse_file).collect_schema().names()
            if name.endswith(":boolean")
        ]

        # make a map from synapse_id to neuropil
        return (
            pl.concat(
                # load each ROI column individually to reduce memory usage
                pl.collect_all(
                    [
                        pl.scan_ipc(self._synapse_file)
                        .rename({self._syn_id: "synapse_id"})
                        # just keep the rows where this ROI is true
                        .filter(
                            pl.col("synapse_id").is_in(synapse_ids),
                            pl.col(roi_column) == True,
                        )
                        # now create the neuropil column for this ROI
                        .with_columns(neuropil=pl.lit(roi_column.rstrip(":boolean")))
                        .select("synapse_id", "neuropil")
                        for roi_column in roi_columns
                    ]
                )
            )
            # add the synapse IDs that don't have neuropils
            .join(
                pl.DataFrame({"synapse_id": synapse_ids}), on="synapse_id", how="right"
            )
            .fill_null("None")
            .select("synapse_id", "neuropil")
        )

    def get_synapse_counts_by_neuropil(
        self,
        synapse_count_type: typing.Literal[
            "downstream", "upstream", "pre", "post", "total_synapses"
        ],
        body_id_subset: list[BodyId] | list[int] | None = None,
    ) -> pl.DataFrame:
        """
        Get neuron or synapse counts for each neuron in each neuropil

        Args:
            synapse_count_type (typing.Literal[ "downstream", "upstream", "pre", "post", "total_synapses"]): which count to get
                * `"downstream"` number of downstream neurons
                * `"upstream"` number of upstream neurons
                * `"pre"` number of presynaptic synapses (ie. synapses to upstream neurons)
                * `"post"` number of postsynaptic synapses (ie. synapses to downstream neurons)
                * `"total_synapses"` total number of synapses, sum of pre and post

            body_id_subset (list[BodyId] | list[int] | None, optional): Only return counts for a certain set of neurons.
                If None, return counts for all. Defaults to None.

        Returns:
            pl.DataFrame: a table [body_id, rois...] with the counts for each neuropil for each body_id.
                **Note:** ROI columns won't be returned if no neurons have a count in that column (ie. if specifying
                a small number of neurons for `body_id_subset`).
        """
        raise NotImplementedError(
            "Not sure how to get synapse counts by neuropil for MANCv1.0 yet..."
        )


class MANC_v_1_2(MANCReader):
    def __init__(
        self,
        connectome_version: typing.Literal["v1.2.1", "v1.2.3"],
        connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
    ):
        super().__init__(connectome_version, connectome_preprocessing)

    # ----- overwritten methods -----
    def _load_specific_neuron_attributes(self):
        # rename common attributes
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
        self._target = "target"

        # attributes specific to MANC
        self._type = "type"
        self._tracing_status = "status"
        self._entry_nerve = "entryNerve"
        self._exit_nerve = "exitNerve"
        self._nb_pre_synapses = "pre"
        self._nb_post_synapses = "post"
        self._nb_pre_neurons = "upstream"
        self._nb_post_neurons = "downstream"
        self._root_side = "rootSide"
        self._roi_info = "roiInfo"

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

    # public methods
    def get_synapse_df(self, body_id: BodyId | int) -> pl.DataFrame:
        """
        Load the synapse ids for the neuron.
        should define the columns
        ['synapse_id','start_bid','end_bid', 'X', 'Y', 'Z']
        """
        column_name_mapping = {
            self._syn_id: "synapse_id",
            self._start_bid: "start_bid",
            self._end_bid: "end_bid",
            self._synapse_x: "X",
            self._synapse_y: "Y",
            self._synapse_z: "Z",
        }
        synapses = (
            pl.scan_ipc(self._synapses_file)
            .select(list(column_name_mapping.keys()))
            # filter on the presynaptic neuron
            .filter(pl.col(self._start_bid) == body_id)
            .rename(column_name_mapping)
        )

        # filter out non traced postsynaptic neurons
        # add a column with 'tracing_status' of the postsynaptic neuron if it exists
        if self.exists_tracing_status():
            nodes_data = (
                pl.scan_ipc(self._nodes_file)
                .select(self._body_id, self._tracing_status)
                .rename(self.decode_neuron_attribute)
            )
            # remove the rows where the postsynaptic neuron is not traced
            synapses = (
                synapses.join(
                    nodes_data, left_on="end_bid", right_on="body_id", how="left"
                )
                .filter(pl.col("tracing_status") == self.traced_entry)
                .drop(["tracing_status"])
            )

        # remove the rows where there are fewer than threshold synapses from
        # a presynaptic neuron to a postsynaptic neuron
        return synapses.filter(
            pl.len().over("start_bid", "end_bid")
            >= self.connectome_preprocessing.min_synapse_count_cutoff
        ).collect()

    def get_synapse_neuropil(
        self,
        synapse_ids: list[int],
    ) -> pl.DataFrame:
        """
        Get the neuropil of the synapses.
        In MANC v1.2, the synapse id is unique only for a given presynaptic neuron,
        so we need to filter on the presynaptic neuron as well.
        """
        column_name_mapping = {
            self._syn_id: "synapse_id",
            self._syn_neuropil: "neuropil",
        }
        return (
            pl.scan_ipc(self._synapses_file)
            .select(list(column_name_mapping.keys()))
            # filter on synapse_ids
            .filter(pl.col(self._syn_id).is_in(synapse_ids))
            .rename(column_name_mapping)
            .collect()
        )

    def get_synapse_counts_by_neuropil(
        self,
        synapse_count_type: typing.Literal[
            "downstream", "upstream", "pre", "post", "total_synapses"
        ],
        body_id_subset: list[BodyId] | list[int] | None = None,
    ) -> pl.DataFrame:
        """
        Get neuron or synapse counts for each neuron in each neuropil

        Args:
            synapse_count_type (typing.Literal[ "downstream", "upstream", "pre", "post", "total_synapses"]): which count to get
                * `"downstream"` number of downstream neurons
                * `"upstream"` number of upstream neurons
                * `"pre"` number of presynaptic synapses (ie. synapses to upstream neurons)
                * `"post"` number of postsynaptic synapses (ie. synapses to downstream neurons)
                * `"total_synapses"` total number of synapses, sum of pre and post

            body_id_subset (list[BodyId] | list[int] | None, optional): Only return counts for a certain set of neurons.
                If None, return counts for all. Defaults to None.

        Returns:
            polars.DataFrame: a table [body_id, rois...] with the counts for each neuropil for each body_id.
                **Note:** ROI columns won't be returned if no neurons have a count in that column (ie. if specifying
                a small number of neurons for `body_id_subset`).
        """
        if synapse_count_type not in {
            "downstream",
            "upstream",
            "pre",
            "post",
            "total_synapses",
        }:
            raise ValueError(
                f"Synapse count type {synapse_count_type} not recognised. Valid values are: ['downstream', 'upstream', 'pre', 'post', 'total_synapses']"
            )
        # the total synapses count is called "synweight" in MANC. Renamed it in the arguments so it's more intuitive
        synapse_count_type_name = (
            "synweight"
            if synapse_count_type == "total_synapses"
            else synapse_count_type
        )

        roi_info_table = pl.scan_ipc(self._nodes_file).select(
            self._body_id, self._roi_info
        )
        if body_id_subset is not None:
            # Note: this removes some ROIs from the columns...
            roi_info_table = roi_info_table.filter(
                pl.col(self._body_id).is_in(body_id_subset)
            )
        roi_info_table = roi_info_table.collect()

        # the roi_Info column contains the number of synapses in a json string like:
        # {ROI: {"downstream" : 0, "upstream" : 1, ...}, ...}
        # we parse each row (which is a string) into a dict using ast.literal_eval
        # then use pl.json_normalize to parse this list of nested dicts into a dataframe
        # the column names end up like ROI.downstream, ROI.upstream, ...
        # so we choose just the columns with the type of count we want and drop the rest
        # then rename the columns
        return (
            pl.json_normalize(
                [ast.literal_eval(x) for x in roi_info_table.get_column("roiInfo")]
            )
            .select(cs.ends_with(f".{synapse_count_type_name}"))
            .rename(lambda column_name: column_name.split(".")[0])
            .fill_null(0)
            .select(roi_info_table.get_column(self._body_id).alias("body_id"), pl.all())
        )


class MANC_v_1_2_1(MANC_v_1_2):
    def __init__(
        self, connectome_preprocessing: ConnectomePreprocessingOptions | None = None
    ):
        super().__init__("v1.2.1", connectome_preprocessing)

    def _load_data_directories(self):
        """
        Need to define the directories that are common to all connectomes.
        """
        # have a fallback if the files are still stored in the v1.2 folder instead of v1.2.1
        super()._load_data_directories()
        if not os.path.exists(self._connectome_dir):
            original_connectome_dir = self._connectome_dir
            self._connectome_dir = os.path.join(
                params.RAW_DATA_DIR,
                self.connectome_name,
                "v1.2",
            )
            # but throw an explicit error here showing that we tried both
            if not os.path.exists(self._connectome_dir):
                raise FileNotFoundError(
                    f"Could't find {self.connectome_name.upper()} {self.connectome_version} files in {original_connectome_dir} or in fallback location {self._connectome_dir}"
                )

        self._nodes_file = os.path.join(self._connectome_dir, "neurons.ftr")
        self._synapses_file = os.path.join(self._connectome_dir, "synapses.ftr")
        self._connections_file = os.path.join(self._connectome_dir, "connections.ftr")


class MANC_v_1_2_3(MANC_v_1_2):
    def __init__(
        self, connectome_preprocessing: ConnectomePreprocessingOptions | None = None
    ):
        super().__init__("v1.2.3", connectome_preprocessing)

    def _load_specific_neuron_classes(self):
        """
        Name the neuron classes.
        """
        MANC_v_1_2._load_specific_neuron_classes(self)
        # MANC 1.2.3 is the same as MANC 1.2 except sensory descending was added
        self._sensory_descending = "sensory descending"

    def _build_class_mapping(self):
        super()._build_class_mapping()

        # MANC 1.2.3 is the same as MANC 1.2 except sensory descending was added
        self.generic_to_specific_class.update(
            {
                "sensory_descending": self._sensory_descending,
            }
        )

    def get_synapse_counts_by_neuropil(
        self,
        synapse_count_type: typing.Literal[
            "downstream", "upstream", "pre", "post", "total_synapses"
        ],
        body_id_subset: list[BodyId] | list[int] | None = None,
    ) -> pl.DataFrame:
        """
        Get neuron or synapse counts for each neuron in each neuropil

        Args:
            synapse_count_type (typing.Literal[ "downstream", "upstream", "pre", "post", "total_synapses"]): which count to get
                * `"downstream"` number of downstream neurons
                * `"upstream"` number of upstream neurons
                * `"pre"` number of presynaptic synapses (ie. synapses to upstream neurons)
                * `"post"` number of postsynaptic synapses (ie. synapses to downstream neurons)
                * `"total_synapses"` total number of synapses, sum of pre and post

            body_id_subset (list[BodyId] | list[int] | None, optional): Only return counts for a certain set of neurons.
                If None, return counts for all. Defaults to None.

        Returns:
            polars.DataFrame: a table [body_id, rois...] with the counts for each neuropil for each body_id.
                **Note:** ROI columns won't be returned if no neurons have a count in that column (ie. if specifying
                a small number of neurons for `body_id_subset`).
        """
        if synapse_count_type not in {
            "downstream",
            "upstream",
            "pre",
            "post",
            "total_synapses",
        }:
            raise ValueError(
                f"Synapse count type {synapse_count_type} not recognised. Valid values are: ['downstream', 'upstream', 'pre', 'post', 'total_synapses']"
            )
        # the total synapses count is called "synweight" in MANC. Renamed it in the arguments so it's more intuitive
        synapse_count_type_name = (
            "synweight"
            if synapse_count_type == "total_synapses"
            else synapse_count_type
        )

        roi_info_table = pl.scan_ipc(self._nodes_file).select(
            self._body_id, self._roi_info
        )
        if body_id_subset is not None:
            # Note: this removes some ROIs from the columns...
            roi_info_table = roi_info_table.filter(
                pl.col(self._body_id).is_in(body_id_subset)
            )
        roi_info_table = roi_info_table.unnest("roiInfo").collect()

        # the roi_Info column contains the number of synapses in a struct like:
        # {ROI: {downstream : 0, upstream : 1, ...}, ...}
        # we check extract the synapse count type that we care about from each struct column
        # and drop the columns that don't contain any synapse counts (all null for all body ids)
        return (
            roi_info_table.select(
                [
                    (
                        pl.col(series.name)
                        .struct[synapse_count_type_name]
                        .alias(series.name)
                        if series.name != "bodyId"
                        else pl.col(series.name)
                    )
                    for series in roi_info_table
                    if series.null_count() < series.len()
                ]
            )
            .fill_null(0)
            .rename({self._body_id: "body_id"})
        )


@typing.overload
def MANC(
    version: typing.Literal["v1.0"],
    connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
) -> MANC_v_1_0: ...


@typing.overload
def MANC(
    version: typing.Literal["v1.2"],
    connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
) -> MANC_v_1_2_1: ...


@typing.overload
def MANC(
    version: typing.Literal["v1.2.1"],
    connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
) -> MANC_v_1_2_1: ...


@typing.overload
def MANC(
    version: typing.Literal["v1.2.3"],
    connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
) -> MANC_v_1_2_3: ...


def MANC(
    version: typing.Literal["v1.0", "v1.2", "v1.2.1", "v1.2.3"],
    connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
) -> MANCReader:
    """Get a connectome reader for one of the versions of the Male Adult Neuronal Connectome (MANC).

    Args:
        version (typing.Literal["v1.0", "v1.2", "v1.2.1", "v1.2.3"]): The valid versions of MANC (1.2 is an alias for 1.2.1)

    Raises:
        ValueError: If an incorrect connectome version is provided

    Returns:
        MANCReader: Either MANC_v_1_0, MANC_v_1_2_1, or MANC_v_1_2_3
    """
    match version:
        case "v1.0":
            return MANC_v_1_0(connectome_preprocessing)
        case "v1.2" | "v1.2.1":
            return MANC_v_1_2_1(connectome_preprocessing)
        case "v1.2.3":
            return MANC_v_1_2_3(connectome_preprocessing)
    raise ValueError(
        f"Version {version} is not a supported version for the MANC connectome. Supported versions are v1.0, v1.2, v1.2.1, and v1.2.3"
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
    def _load_specific_neuron_attributes(self):
        """
        Need to define the fields that are common to all connectomes.
        BodyId, start_bid, end_bid, syn_count, nt_type, class_1, class_2, neuron_attributes
        """

        # rename common attributes
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
        # attributes specific to FAFB
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

    def _build_attribute_mapping(self):
        super()._build_attribute_mapping()

        self.generic_to_specific_attribute.update(
            {
                "nerve": self._nerve,
                "area": self._area,
                "length": self._length,
                "flow": self._flow,
            }
        )

    def _build_class_mapping(self):
        super()._build_class_mapping()

        self.generic_to_specific_class.update(
            {
                "central": self._central,
                "endocrine": self._endocrine,
                "optic": self._optic,
                "visual_centrifugal": self._visual_centrifugal,
                "visual_projection": self._visual_projection,
                "other": self._other,
            }
        )

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

    def _load_connections(self) -> pl.DataFrame:
        """
        Load the connections of the connectome.
        Needs to gather the columns ['start_bid', 'end_bid', 'syn_count', 'nt_type'].
        """
        return pl.read_csv(
            self._connections_file,
            columns=[self._start_bid, self._end_bid, self._syn_count, self._nt_type],
        ).rename(self.decode_neuron_attribute)

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
            return set(
                pl.scan_csv(filename, schema_overrides={self._body_id: pl.UInt64})
                .filter(pl.col(att) == value)
                .select(self._body_id)
                .collect()
                .get_column(self._body_id)
            )

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
    def get_synapse_df(self, body_id: BodyId | int) -> pl.DataFrame:
        """
        Load the synapse ids for the neuron.
        should define the columns
        ['synapse_id','start_bid','end_bid', 'X', 'Y', 'Z']
        """
        type_dict = {
            "pre_root_id": pl.UInt64,
            "post_root_id": pl.UInt64,
            "x": pl.Int32,
            "y": pl.Int32,
            "z": pl.Int32,
        }
        return (
            pl.scan_csv(
                self._synapses_file,
                schema=type_dict,
            )
            .rename(
                {
                    "pre_root_id": "start_bid",
                    "post_root_id": "end_bid",
                    "x": "X",
                    "y": "Y",
                    "z": "Z",
                }
            )
            .fill_null(strategy="forward")
            # put the synapse_id column first
            .select(
                pl.int_range(pl.len(), dtype=pl.UInt64).alias("synapse_id"),
                pl.all(),
            )
            .filter(pl.col("start_bid") == body_id)
            .collect()
        )

    def get_synapse_neuropil(
        self,
        synapse_ids: list[int],
    ) -> pl.DataFrame:
        raise NotImplementedError(
            'No trivial match in FAFB between coordinates and neuropil...\
            You can try to circumvent with the number of synapses in a neuropil for a given neuron.\
            For that, load "neuropil_synapse_table.csv"'
        )

    def get_synapse_counts_by_neuropil(
        self,
        synapse_count_type: typing.Literal[
            "downstream", "upstream", "pre", "post", "total_synapses"
        ],
        body_id_subset: list[BodyId] | list[int] | None = None,
    ):
        """
        Get neuron or synapse counts for each neuron in each neuropil

        Args:
            synapse_count_type (typing.Literal[ "downstream", "upstream", "pre", "post", "total_synapses"]): which count to get
                * `"downstream"` number of downstream neurons
                * `"upstream"` number of upstream neurons
                * `"pre"` number of presynaptic synapses (ie. synapses to upstream neurons)
                * `"post"` number of postsynaptic synapses (ie. synapses to downstream neurons)
                * `"total_synapses"` total number of synapses, sum of pre and post

            body_id_subset (list[BodyId] | list[int] | None, optional): Only return counts for a certain set of neurons.
                If None, return counts for all. Defaults to None.

        Returns:
            pl.DataFrame: a table [body_id, rois...] with the counts for each neuropil for each body_id.
                **Note:** ROI columns won't be returned if no neurons have a count in that column (ie. if specifying
                a small number of neurons for `body_id_subset`).
        """
        connections_table = pl.scan_csv(
            self._connections_file,
        )

        match synapse_count_type:
            case "total_synapses":
                if body_id_subset is not None:
                    # Note: this removes some ROIs from the columns...
                    connections_table = connections_table.filter(
                        pl.col(self._start_bid).is_in(body_id_subset)
                        | pl.col(self._end_bid).is_in(body_id_subset)
                    )
                # separately get the pre and post synapse counts then sum them
                synapse_counts = (
                    pl.concat(
                        [
                            connections_table.select(
                                neuron_body_id_we_care_about, "neuropil", "syn_count"
                            )
                            .group_by([neuron_body_id_we_care_about, "neuropil"])
                            .sum()
                            .collect()
                            .pivot(
                                index=neuron_body_id_we_care_about,
                                on="neuropil",
                                values="syn_count",
                            )
                            .fill_null(0)
                            .rename({neuron_body_id_we_care_about: "body_id"})
                            for neuron_body_id_we_care_about in [
                                self._start_bid,
                                self._end_bid,
                            ]
                        ],
                        how="diagonal",  # because the column order mightn't be the same
                    )
                    .group_by("body_id")
                    .sum()
                )
                if body_id_subset is not None:
                    synapse_counts = synapse_counts.filter(
                        pl.col("body_id").is_in(body_id_subset)
                    )
                return synapse_counts
            case "downstream" | "post":
                neuron_body_id_we_care_about = self._start_bid
            case "upstream" | "pre":
                neuron_body_id_we_care_about = self._end_bid
            case _:
                raise ValueError(
                    f"Synapse count type {synapse_count_type} not recognised. Valid values are: ['downstream', 'upstream', 'pre', 'post', 'total_synapses']"
                )

        if body_id_subset is not None:
            # Note: this removes some ROIs from the columns...
            connections_table = connections_table.filter(
                pl.col(neuron_body_id_we_care_about).is_in(body_id_subset)
            )

        if synapse_count_type in ["downstream", "upstream"]:
            # resulting column should be called syn_count
            aggregate_group_function = lambda group: group.len("syn_count")
        else:
            # sum preserves column names
            aggregate_group_function = lambda group: group.sum()

        return (
            aggregate_group_function(
                connections_table.select(
                    neuron_body_id_we_care_about, "neuropil", "syn_count"
                ).group_by([neuron_body_id_we_care_about, "neuropil"])
            )
            .collect()
            .pivot(
                index=neuron_body_id_we_care_about,
                on="neuropil",
                values="syn_count",
            )
            .fill_null(0)
            .rename({neuron_body_id_we_care_about: "body_id"})
        )

    def list_all_nodes(self) -> list[BodyId]:
        """
        List all the neurons existing in the connectome.
        """
        return (
            pl.read_csv(
                self._node_stats_file,
                columns=[self._body_id],
                schema_overrides={self._body_id: pl.UInt64},
            )
            .get_column(self._body_id)
            .to_list()
        )

    def get_neuron_bodyids(
        self,
        selection_dict: SelectionDict = {},
        nodes: Optional[list[BodyId] | list[int]] = None,
    ) -> list[BodyId]:
        """
        Get the BodyIds of the neurons in the dataset that fulfil the conditions in the selection_dict.

        For the specific case of "class_1" that refers to the NeuronClass, we need to verify both the generic and the specific names.

        Args:
            selection_dict (SelectionDict, optional): Criteria that the returned neurons need to fulfil. Different criteria are treated as 'and' conditions. Defaults to {}.
            nodes (Optional[list[BodyId]  |  list[int]], optional): If not None, only return BodyIds which are contained in this list. Defaults to None.

        Returns:
            list[BodyId]: list of the BodyIds of neurons that fulfilled all supplied conditions.
        """
        # get all neurons in the dataset that are also in the nodes list
        valid_nodes = set(self.list_all_nodes())
        if nodes is not None:
            valid_nodes = valid_nodes.intersection(nodes)

        # Treat each attribute in the selection dict independently:
        # get the nodes that satisfy each condition, and return the intersection of all
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
    ) -> pl.DataFrame:
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
        polars.DataFrame
            The data of the neuron.
        """
        return self.load_data_neuron_set([id_], attributes)

    def load_data_neuron_set(
        self,
        ids: list[BodyId] | list[int],
        attributes: list[NeuronAttribute] = [],
    ) -> pl.DataFrame:
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
        polars.DataFrame
            The data of the neurons.
        """

        def _load_df(
            filename: str,
            columns: set[NeuronAttribute],
            bids: list[BodyId] | list[int],
        ) -> pl.DataFrame:
            columns.add("body_id")
            columns_to_read = [self.sna(a) for a in columns]
            return (
                pl.scan_csv(filename, schema_overrides={self._body_id: pl.UInt64})
                .select(columns_to_read)
                .rename(self.decode_neuron_attribute)
                .filter(pl.col("body_id").is_in(bids))
                .collect()
            )

        # Load data
        neurons = pl.DataFrame({"body_id": ids})
        # split the data loading as a function of the files required
        # 1. nt_type, nt_proba, neuropil
        nt_fields = set(attributes).intersection(
            {
                "nt_type",
                "nt_proba",
                "neuropil",
            }
        )
        if len(nt_fields) > 0:
            data = _load_df(self._node_nt_type_file, nt_fields, ids)
            neurons = neurons.join(data, on="body_id", how="inner")

        # 2. length, area, size
        stat_fields = set(attributes).intersection({"length", "area", "size"})
        if len(stat_fields) > 0:
            data = _load_df(self._node_stats_file, stat_fields, ids)
            neurons = neurons.join(data, on="body_id", how="inner")

        # 3. position
        if "position" in attributes:
            data = _load_df(self._node_position_file, {"position"}, ids)
            # FIX: data contains multiple positions for the same body id. Why?
            data = data.unique("body_id")
            neurons = neurons.join(data, on="body_id", how="inner")

        # 4. class, hemilineage, side etc.
        class_fields = (
            set(attributes)
            .difference(nt_fields)
            .difference(stat_fields)
            .difference({"position", "body_id"})
        )
        if len(class_fields) > 0:
            data = _load_df(self._node_class_file, class_fields, ids)
            neurons = neurons.join(data, on="body_id", how="inner")

        if "body_id" not in attributes:
            attributes.append("body_id")
        return neurons.filter(pl.col("body_id").is_in(ids))[attributes]


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


# === BANC: Brain and nerve cord
class BANCReader(ConnectomeReader):
    def __init__(
        self,
        connectome_version: str,
        connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
    ):
        super().__init__("banc", connectome_version, connectome_preprocessing)

        self.nt_weights = {
            "ACH": +1,
            "GABA": -1,
            "GLUT": -1 if self.connectome_preprocessing.glutamate_inhibitory else +1,
            "DA": 0,
            "OCT": 0,
            "SER": 0,
            "TYR": 0,
            "HIST": 0,
            None: 0,
        }

    # ----- overwritten methods -----
    def _load_specific_neuron_attributes(self):
        """
        Need to define the fields that are common to all connectomes.
        BodyId, start_bid, end_bid, syn_count, nt_type, class_1, class_2, neuron_attributes
        """

        # rename common attributes
        self._body_id = "Root ID"
        self._syn_id = "None-syn_id"
        self._start_bid = "pre_root_id"
        self._end_bid = "post_root_id"
        self._syn_count = "syn_count"
        self._nt_type = "Predicted NT type"
        self._nt_proba = "None-nt_proba"
        self._class_1 = (
            "Super Class"  # There's also flow but I think it's a bit too coarse grained
        )
        self._class_2 = "Class"
        self._name = "Primary Cell Type"
        self._side = "Soma side"
        self._neuropil = "None-neuropil"
        self._hemilineage = "Hemilineage"
        self._size = "Volume (nm^3)"
        # self._position = "position"
        # self._target = "Sub Class"
        # attributes specific to BANC
        self._nerve = "Nerve"
        self._area = "Surface area (nm^2)"
        self._length = "Cable length (nm)"
        self._flow = "Flow"
        self._class_3 = "Sub Class"
        self._name_alternatives = "Alternative Cell Type(s)"

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
        self._ascending_visceral = "ascending_visceral_circulatory"
        self._central = "central_brain_intrinsic"
        self._glia = "glia"
        self._optic = "optic_lobe_intrinsic"
        self._sensory_ascending = "sensory_ascending"
        self._sensory_descending = "sensory_descending"
        self._vnc_intrinsic = "ventral_nerve_cord_intrinsic"
        self._visceral_circulatory = "visceral_circulatory"
        self._visual_centrifugal = "visual_centrifugal"
        self._visual_projection = "visual_projection"

    def _build_attribute_mapping(self):
        super()._build_attribute_mapping()

        self.generic_to_specific_attribute.update(
            {
                "nerve": self._nerve,
                "area": self._area,
                "length": self._length,
                "flow": self._flow,
                "class_3": self._class_3,
                "name_alternatives": self._name_alternatives,
            }
        )

    def _build_class_mapping(self):
        super()._build_class_mapping()

        self.generic_to_specific_class.update(
            {
                "ascending_visceral": self._ascending_visceral,
                "central": self._central,
                "glia": self._glia,
                "optic": self._optic,
                "sensory_ascending": self._sensory_ascending,
                "sensory_descending": self._sensory_descending,
                "vnc_intrinsic": self._vnc_intrinsic,
                "visceral_circulatory": self._visceral_circulatory,
                "visual_centrifugal": self._visual_centrifugal,
                "visual_projection": self._visual_projection,
            }
        )

    def _load_data_directories(self):
        """
        Need to define the directories that are common to all connectomes.
        """
        self._connectome_dir = os.path.join(
            params.RAW_DATA_DIR,
            self.connectome_name,
            self.connectome_version,
        )

        # all information for a neuron
        self._node_stats_file = os.path.join(  # length, area, size
            self._connectome_dir, "neurons.csv"
        )

        # connections between neurons
        self._connections_file = os.path.join(
            self._connectome_dir, "connections_princeton.csv"
        )

    def _get_traced_bids(self) -> list[BodyId]:
        """
        Get the body ids of the traced neurons.
        """
        raise NotImplementedError("Method should not be called on BANC.")

    def _load_connections(self) -> pl.DataFrame:
        """
        Load the connections of the connectome.
        Needs to gather the columns ['start_bid', 'end_bid', 'syn_count', 'nt_type'].
        """
        # the NT column in the connections file is all null, so we need to get the information from the presynaptic neuron
        return (
            pl.scan_csv(
                self._connections_file,
            )
            .select(self._start_bid, self._end_bid, self._syn_count)
            .join(
                pl.scan_csv(self._node_stats_file).select(self._body_id, self._nt_type),
                left_on=self._start_bid,
                right_on=self._body_id,
            )
            .rename(self.decode_neuron_attribute)
            .collect()
        )

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

        attribute_specific = self.sna(attribute)

        neurons_df = pl.scan_csv(
            self._node_stats_file, schema_overrides={self._body_id: pl.UInt64}
        )

        if attribute == "class_1":
            # need to check both the generic and the specific names
            neurons_df = neurons_df.filter(
                (pl.col(attribute_specific) == value)
                | (pl.col(attribute_specific) == self.specific_neuron_class(value))
            )
        else:
            neurons_df = neurons_df.filter(pl.col(attribute_specific) == value)

        return set(neurons_df.select(self._body_id).collect().get_column(self._body_id))

    # public methods
    def get_synapse_df(self, body_id: BodyId | int) -> pl.DataFrame:
        """
        Load the synapse ids for the neuron.
        should define the columns
        ['synapse_id','start_bid','end_bid', 'X', 'Y', 'Z']
        """
        raise NotImplementedError("Can't get synapse coordinates for BANC")

    def get_synapse_neuropil(
        self,
        synapse_ids: list[int],
    ) -> pl.DataFrame:
        raise NotImplementedError(
            'No trivial match in FAFB between coordinates and neuropil...\
            You can try to circumvent with the number of synapses in a neuropil for a given neuron.\
            For that, load "neuropil_synapse_table.csv"'
        )

    def get_synapse_counts_by_neuropil(
        self,
        synapse_count_type: typing.Literal[
            "downstream", "upstream", "pre", "post", "total_synapses"
        ],
        body_id_subset: list[BodyId] | list[int] | None = None,
    ):
        """
        Get neuron or synapse counts for each neuron in each neuropil

        Args:
            synapse_count_type (typing.Literal[ "downstream", "upstream", "pre", "post", "total_synapses"]): which count to get
                * `"downstream"` number of downstream neurons
                * `"upstream"` number of upstream neurons
                * `"pre"` number of presynaptic synapses (ie. synapses to upstream neurons)
                * `"post"` number of postsynaptic synapses (ie. synapses to downstream neurons)
                * `"total_synapses"` total number of synapses, sum of pre and post

            body_id_subset (list[BodyId] | list[int] | None, optional): Only return counts for a certain set of neurons.
                If None, return counts for all. Defaults to None.

        Returns:
            pl.DataFrame: a table [body_id, rois...] with the counts for each neuropil for each body_id.
                **Note:** ROI columns won't be returned if no neurons have a count in that column (ie. if specifying
                a small number of neurons for `body_id_subset`).
        """
        raise NotImplementedError("Can't get synapse counts by neuropil for the BANC")

    def list_all_nodes(self) -> list[BodyId]:
        """
        List all the neurons existing in the connectome.
        """
        return (
            pl.read_csv(
                self._node_stats_file,
                columns=[self._body_id],
                schema_overrides={self._body_id: pl.UInt64},
            )
            .get_column(self._body_id)
            .to_list()
        )

    def get_neuron_bodyids(
        self,
        selection_dict: SelectionDict = {},
        nodes: Optional[list[BodyId] | list[int]] = None,
    ) -> list[BodyId]:
        """
        Get the BodyIds of the neurons in the dataset that fulfil the conditions in the selection_dict.

        For the specific case of "class_1" that refers to the NeuronClass, we need to verify both the generic and the specific names.

        Args:
            selection_dict (SelectionDict, optional): Criteria that the returned neurons need to fulfil. Different criteria are treated as 'and' conditions. Defaults to {}.
            nodes (Optional[list[BodyId]  |  list[int]], optional): If not None, only return BodyIds which are contained in this list. Defaults to None.

        Returns:
            list[BodyId]: list of the BodyIds of neurons that fulfilled all supplied conditions.
        """
        # get all neurons in the dataset that are also in the nodes list
        valid_nodes = set(self.list_all_nodes())
        if nodes is not None:
            valid_nodes = valid_nodes.intersection(nodes)

        # Treat each attribute in the selection dict independently:
        # get the nodes that satisfy each condition, and return the intersection of all
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
    ) -> pl.DataFrame:
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
        polars.DataFrame
            The data of the neuron.
        """
        return self.load_data_neuron_set([id_], attributes)

    def load_data_neuron_set(
        self,
        ids: list[BodyId] | list[int],
        attributes: list[NeuronAttribute] = [],
    ) -> pl.DataFrame:
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
        polars.DataFrame
            The data of the neurons.
        """

        if "body_id" not in attributes:
            attributes.insert(0, "body_id")
        columns_to_read = [self.sna(a) for a in attributes]
        return (
            # creating a dataframe with the body ids then joining the attributes keeps the rows in the same order
            pl.DataFrame({"body_id": ids})
            .lazy()
            .join(
                pl.scan_csv(
                    self._node_stats_file, schema_overrides={self._body_id: pl.UInt64}
                )
                .select(columns_to_read)
                .rename(self.decode_neuron_attribute)
                .filter(pl.col("body_id").is_in(ids)),
                on="body_id",
            )
            # return attributes in the specified order
            .select(attributes)
        ).collect()


# Specific versions of BANC
class BANC_v626(BANCReader):
    def __init__(
        self, connectome_preprocessing: ConnectomePreprocessingOptions | None = None
    ):
        super().__init__("v626", connectome_preprocessing)


def BANC(
    version: typing.Literal["v626"],
    connectome_preprocessing: ConnectomePreprocessingOptions | None = None,
) -> BANCReader:
    """Get a connectome reader for one of the versions of the Brain and Nerve Cord connectome (BANC).

    Args:
        version (typing.Literal["v626"]): The valid versions of the BANC

    Raises:
        ValueError: If an incorrect connectome version is provided

    Returns:
        FAFBReader: BANC_v626
    """
    match version:
        case "v626":
            return BANC_v626(connectome_preprocessing)
    raise ValueError(
        f"Version {version} is not a supported version for the BANC connectome. Supported version is v626"
    )

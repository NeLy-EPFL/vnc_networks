"""
Can only run when the data is available. Not run in the CI.
"""

from vnc_networks.params import SelectionDict


class TestDataLoading:
    """
    Test the data loading functions.
    """

    def test_connections_instantiation_default(self):
        """
        Test the instantiation of the Connections class.
        By default it should use the MANC v1.2.3 connectome
        """
        from vnc_networks.connections import Connections
        from vnc_networks.connectome_reader import MANC

        # Instantiate a Connections object
        valid_connections = Connections()
        assert valid_connections is not None, "Connections object not instantiated"

        cr_name = valid_connections.CR.connectome_name
        assert cr_name == "manc", "Incorrect connectome name"
        cr_version = valid_connections.CR.connectome_version
        assert cr_version == "v1.2.3", "Incorrect connectome version"

    '''
    def test_connections_instantiation_MANCv1_0(self):
        """
        Test the instantiation of the Connections class.
        """
        from vnc_networks.connections import Connections
        from vnc_networks.connectome_reader import MANC

        # Instantiate a Connections object
        valid_connections = Connections(CR=MANC("v1.0"))
        assert valid_connections is not None, "Connections object not instantiated"

        cr_name = valid_connections.CR.connectome_name
        assert cr_name == "manc", "Incorrect connectome name"
        cr_version = valid_connections.CR.connectome_version
        assert cr_version == "v1.0", "Incorrect connectome version"

        # check the data loading
        df = valid_connections.get_dataframe()
        assert df is not None, "Couldn't get connections dataframe"

        df_1 = df[(df["start_bid"] == 10000) & (df["end_bid"] == 14882)]
        assert df_1["eff_weight"].values[0] == 136, "Incorrect data values"

        df_2 = df[(df["start_bid"] == 10001) & (df["end_bid"] == 29119)]
        assert df_2["eff_weight"].values[0] == -5, "Incorrect nt_type handling"
    '''

    def test_connectome_reader_version_mapping_MANCv1_2(self):
        from vnc_networks.connectome_reader import MANC

        connectome_reader = MANC("v1.2")
        assert connectome_reader.connectome_name == "manc", "Incorrect connectome name"
        assert (
            connectome_reader.connectome_version == "v1.2.1"
        ), "Incorrect connectome version. MANC v1.2 should map to v1.2.1"

    def test_connectome_reader_version_mapping_MANCv1_2_1(self):
        from vnc_networks.connectome_reader import MANC

        connectome_reader = MANC("v1.2.1")
        assert connectome_reader.connectome_name == "manc", "Incorrect connectome name"
        assert (
            connectome_reader.connectome_version == "v1.2.1"
        ), "Incorrect connectome version. MANC v1.2.1 should map to v1.2.1"

    def test_connectome_reader_version_mapping_MANCv1_2_3(self):
        from vnc_networks.connectome_reader import MANC

        connectome_reader = MANC("v1.2.3")
        assert connectome_reader.connectome_name == "manc", "Incorrect connectome name"
        assert (
            connectome_reader.connectome_version == "v1.2.3"
        ), "Incorrect connectome version. MANC v1.2.3 should map to v1.2.3"

    def test_connections_instantiation_MANCv1_2_1(self):
        """
        Test the instantiation of the Connections class.
        """
        import vnc_networks
        from vnc_networks.connections import Connections

        # Instantiate a Connections object
        valid_connections = Connections(vnc_networks.connectome_reader.MANC("v1.2.1"))
        assert valid_connections is not None, "Connections object not instantiated"

        # check the data loading
        df = valid_connections.get_dataframe()
        assert df is not None, "Couldn't get connections dataframe"

        df_1 = df[(df["start_bid"] == 11218) & (df["end_bid"] == 10094)]
        assert df_1["eff_weight"].values[0] == 915, "Incorrect data values"

        df_2 = df[(df["start_bid"] == 10725) & (df["end_bid"] == 10439)]
        assert df_2["eff_weight"].values[0] == -1080, "Incorrect nt_type handling"

    def test_connections_instantiation_MANCv1_2_3(self):
        """
        Test the instantiation of the Connections class.
        """
        import vnc_networks
        from vnc_networks.connections import Connections

        # Instantiate a Connections object
        valid_connections = Connections(vnc_networks.connectome_reader.MANC("v1.2.3"))
        assert valid_connections is not None, "Connections object not instantiated"

        # check the data loading
        df = valid_connections.get_dataframe()
        assert df is not None, "Couldn't get connections dataframe"

        df_1 = df[(df["start_bid"] == 11218) & (df["end_bid"] == 10094)]
        assert df_1["eff_weight"].values[0] == 915, "Incorrect data values"

        df_2 = df[(df["start_bid"] == 10725) & (df["end_bid"] == 10439)]
        assert df_2["eff_weight"].values[0] == -1080, "Incorrect nt_type handling"

    def test_connections_getting_neuron_ids_MANCv1_2_1(self):
        """
        Test that we get the same results if we get uids or bodyids and convert between the two.
        """
        import vnc_networks
        from vnc_networks.connections import Connections

        # Instantiate a Connections object
        connections = Connections(vnc_networks.connectome_reader.MANC("v1.2.1"))

        # test a few different selection_dicts
        selection_dicts: list[SelectionDict | None] = [
            None,
            {},
            {"class_1": "descending"},
            {"class_1": "ascending", "nt_type": "GABA"},
        ]
        for selection_dict in selection_dicts:
            body_ids = connections.get_neuron_bodyids(selection_dict)
            uids = connections.get_neuron_ids(selection_dict)
            assert (
                set(connections.get_uids_from_bodyids(body_ids)) == set(uids)
            ), f"Getting bodyids and converting to uids doesn't match with selection_dict {selection_dict}."
            assert (
                set(connections.get_bodyids_from_uids(uids)) == set(body_ids)
            ), f"Getting uids and converting to bodyids doesn't match with selection_dict {selection_dict}."

    def test_connections_getting_neuron_ids_MANCv1_2_3(self):
        """
        Test that we get the same results if we get uids or bodyids and convert between the two.
        """
        import vnc_networks
        from vnc_networks.connections import Connections

        # Instantiate a Connections object
        connections = Connections(vnc_networks.connectome_reader.MANC("v1.2.3"))

        # test a few different selection_dicts
        selection_dicts: list[SelectionDict | None] = [
            None,
            {},
            {"class_1": "descending"},
            {"class_1": "ascending", "nt_type": "GABA"},
        ]
        for selection_dict in selection_dicts:
            body_ids = connections.get_neuron_bodyids(selection_dict)
            uids = connections.get_neuron_ids(selection_dict)
            assert (
                set(connections.get_uids_from_bodyids(body_ids)) == set(uids)
            ), f"Getting bodyids and converting to uids doesn't match with selection_dict {selection_dict}."
            assert (
                set(connections.get_bodyids_from_uids(uids)) == set(body_ids)
            ), f"Getting uids and converting to bodyids doesn't match with selection_dict {selection_dict}."

    def test_getting_counts_by_neuropil_MANCv1_2_1(self):
        """
        Test that we can get neuron and synapse counts by neuropil
        """
        import pandas as pd

        import vnc_networks

        # Instantiate a Connections object
        connectome_reader = vnc_networks.connectome_reader.MANC("v1.2.1")

        # check that this matches what we expect
        pd.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "downstream", [10000, 23458]
            ),
            pd.DataFrame(
                {
                    "body_id": [10000, 23458],
                    "CV": [703, 0],
                    "IntTct": [313, 0],
                    "LTct": [3181, 0],
                    "LegNp(T3)(R)": [0, 1688],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "upstream", [10000, 23458]
            ),
            pd.DataFrame(
                {
                    "body_id": [10000, 23458],
                    "CV": [224, 0],
                    "IntTct": [185, 0],
                    "LTct": [1462, 0],
                    "LegNp(T3)(R)": [0, 685],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil("pre", [10000, 23458]),
            pd.DataFrame(
                {
                    "body_id": [10000, 23458],
                    "CV": [138, 0],
                    "IntTct": [73, 0],
                    "LTct": [752, 0],
                    "LegNp(T3)(R)": [0, 207],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil("post", [10000, 23458]),
            pd.DataFrame(
                {
                    "body_id": [10000, 23458],
                    "CV": [224, 0],
                    "IntTct": [185, 0],
                    "LTct": [1462, 0],
                    "LegNp(T3)(R)": [0, 685],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "total_synapses", [10000, 23458]
            ),
            pd.DataFrame(
                {
                    "body_id": [10000, 23458],
                    "CV": [927, 0],
                    "IntTct": [498, 0],
                    "LTct": [4643, 0],
                    "LegNp(T3)(R)": [0, 2373],
                }
            ),
        )

    def test_getting_counts_by_neuropil_FAFBv783(self):
        """
        Test that we can get neuron and synapse counts by neuropil
        """
        import pandas as pd

        import vnc_networks

        # Instantiate a Connections object
        connectome_reader = vnc_networks.connectome_reader.FAFB_v783()

        # check that this matches what we expect
        pd.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "downstream", [720575940627036426, 720575940633587552]
            ),
            pd.DataFrame(
                {
                    "body_id": [720575940627036426, 720575940633587552],
                    "LOP_L": [9, 0],
                    "LO_L": [2, 0],
                    "SLP_R": [0, 2],
                    "SMP_R": [0, 31],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "upstream", [720575940627036426, 720575940633587552]
            ),
            pd.DataFrame(
                {
                    "body_id": [720575940627036426, 720575940633587552],
                    "LOP_L": [3, 0],
                    "LO_L": [9, 0],
                    "SLP_R": [0, 11],
                    "SMP_R": [0, 13],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "pre", [720575940627036426, 720575940633587552]
            ),
            pd.DataFrame(
                {
                    "body_id": [720575940627036426, 720575940633587552],
                    "LOP_L": [14, 0],
                    "LO_L": [83, 0],
                    "SLP_R": [0, 33],
                    "SMP_R": [0, 75],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "post", [720575940627036426, 720575940633587552]
            ),
            pd.DataFrame(
                {
                    "body_id": [720575940627036426, 720575940633587552],
                    "LOP_L": [100, 0],
                    "LO_L": [8, 0],
                    "SLP_R": [0, 4],
                    "SMP_R": [0, 284],
                }
            ),
        )
        pd.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "total_synapses", [720575940627036426, 720575940633587552]
            ),
            pd.DataFrame(
                {
                    "body_id": [720575940627036426, 720575940633587552],
                    "LOP_L": [114, 0],
                    "LO_L": [91, 0],
                    "SLP_R": [0, 37],
                    "SMP_R": [0, 359],
                }
            ),
        )

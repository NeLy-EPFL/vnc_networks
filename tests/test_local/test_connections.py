"""
Can only run when the data is available. Not run in the CI.
"""


class TestConnections:
    """
    Test the data loading functions.
    """

    def test_connections_instantiation_default(self):
        """
        Test the instantiation of the Connections class.
        By default it should use the MANC v1.2.3 connectome
        """
        from vnc_networks.connections import Connections

        # Instantiate a Connections object
        valid_connections = Connections()
        assert valid_connections is not None, "Connections object not instantiated"

        cr_name = valid_connections.CR.connectome_name
        assert cr_name == "manc", "Incorrect connectome name"
        cr_version = valid_connections.CR.connectome_version
        assert cr_version == "v1.2.3", "Incorrect connectome version"

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

        df_1 = df.filter(start_bid=10000, end_bid=14882)
        assert df_1[0, "eff_weight"] == 136, "Incorrect data values"

        df_2 = df.filter(start_bid=10001, end_bid=29119)
        assert df_2[0, "eff_weight"] == -5, "Incorrect nt_type handling"

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

        df_1 = df.filter(start_bid=11218, end_bid=10094)
        assert df_1[0, "eff_weight"] == 915, "Incorrect data values"

        df_2 = df.filter(start_bid=10725, end_bid=10439)
        assert df_2[0, "eff_weight"] == -1080, "Incorrect nt_type handling"

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

        df_1 = df.filter(start_bid=11218, end_bid=10094)
        assert df_1[0, "eff_weight"] == 915, "Incorrect data values"

        df_2 = df.filter(start_bid=10725, end_bid=10439)
        assert df_2[0, "eff_weight"] == -1080, "Incorrect nt_type handling"

    def test_connections_getting_neuron_ids_MANCv1_2_1(self):
        """
        Test that we get the same results if we get uids or bodyids and convert between the two.
        """
        import vnc_networks
        from vnc_networks.connections import Connections
        from vnc_networks.params import SelectionDict

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
            assert set(connections.get_uids_from_bodyids(body_ids)) == set(uids), (
                f"Getting bodyids and converting to uids doesn't match with selection_dict {selection_dict}."
            )
            assert set(connections.get_bodyids_from_uids(uids)) == set(body_ids), (
                f"Getting uids and converting to bodyids doesn't match with selection_dict {selection_dict}."
            )

    def test_connections_getting_neuron_ids_MANCv1_2_3(self):
        """
        Test that we get the same results if we get uids or bodyids and convert between the two.
        """
        import vnc_networks
        from vnc_networks.connections import Connections
        from vnc_networks.params import SelectionDict

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
            assert set(connections.get_uids_from_bodyids(body_ids)) == set(uids), (
                f"Getting bodyids and converting to uids doesn't match with selection_dict {selection_dict}."
            )
            assert set(connections.get_bodyids_from_uids(uids)) == set(body_ids), (
                f"Getting uids and converting to bodyids doesn't match with selection_dict {selection_dict}."
            )

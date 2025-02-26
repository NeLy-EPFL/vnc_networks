"""
Can only run when the data is available. Not run in the CI.
"""


class TestDataLoading:
    """
    Test the data loading functions.
    """

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

    def test_connections_instantiation_MANCv1_2(self):
        """
        Test the instantiation of the Connections class.
        """
        import vnc_networks
        from vnc_networks.connections import Connections

        # Instantiate a Connections object
        valid_connections = Connections(vnc_networks.connectome_reader.MANC("v1.2"))
        assert valid_connections is not None, "Connections object not instantiated"
        cr_name = valid_connections.CR.connectome_name
        assert cr_name == "manc", "Incorrect connectome name"
        cr_version = valid_connections.CR.connectome_version
        assert cr_version == "v1.2", "Incorrect connectome version"

        # check the data loading
        df = valid_connections.get_dataframe()
        assert df is not None, "Couldn't get connections dataframe"

        df_1 = df[(df["start_bid"] == 11218) & (df["end_bid"] == 10094)]
        assert df_1["eff_weight"].values[0] == 915, "Incorrect data values"

        df_2 = df[(df["start_bid"] == 10725) & (df["end_bid"] == 10439)]
        assert df_2["eff_weight"].values[0] == -1080, "Incorrect nt_type handling"

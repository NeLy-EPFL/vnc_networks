'''
Can only run when the data is available. Not run in the CI.
'''

class TestDataLoading:
    """
    Test the data loading functions.
    """
    
    def test_connections_instantiation(self):
        """
        Test the instanciation of the Connections class.
        """
        from vnc_networks.connections import Connections

        # Instanciate a Connections object
        valid_connections = Connections()
        assert valid_connections is not None, "Connections object not instanciated"

        # default associated ConnectomeReader should be MANC v1.0
        cr_name = valid_connections.CR.connectome_name
        assert cr_name == 'MANC', "Incorrect connectome name"
        cr_version = valid_connections.CR.connectome_version
        assert cr_version == 'v1.0', "Incorrect connectome version"

        # check the data loading
        df = valid_connections.get_dataframe()

        df_1 = df[(df['start_bid'] == 10000) & (df['end_bid'] == 14882)]
        assert df_1['eff_weight'].values[0] == 136, "Incorrect data values"

        df_2 = df[(df['start_bid'] == 10001) & (df['end_bid'] == 29119)]
        assert df_2['eff_weight'].values[0] == -5, "Incorrect nt_type handeling"

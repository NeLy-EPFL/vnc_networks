"""
Initialisation tests for the code.
"""

import pytest

class TestImport:
    def test_import(self):
        """
        Test the import of the module.
        """
        try:
            import vnc_networks
            assert True
        except ImportError:
            assert False

    def test_import_submodules(self):
        """
        Test the import of the submodules.
        """
        try:
            from vnc_networks import (
                cmatrix,
                connections,
                connectome_reader,
                neuron,
                params,
            )
            assert True
        except ImportError:
            assert False

    def test_import_classes(self):
        """
        Test the import of the classes.
        """
        try:
            from vnc_networks.cmatrix import CMatrix
            from vnc_networks.connections import Connections
            from vnc_networks.connectome_reader import FAFB, MANC
            from vnc_networks.neuron import Neuron
            assert True
        except ImportError:
            assert False




class TestInstanciation:
    def test_connectome_reader_instanciation(self):
        """
        Test the instanciation of the ConnectomeReader class.
        """
        from vnc_networks.connectome_reader import MANC

        # Should work
        valid_reader = MANC('v1.0')
        assert valid_reader is not None
        body_id_name = valid_reader.sna('body_id')
        assert body_id_name == ":ID(Body-ID)", "Incorrect name conversion"

        # Should raise an error
        try:
            invalid_reader = MANC('v1.1')
            assert False
        except ValueError:
            assert True

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




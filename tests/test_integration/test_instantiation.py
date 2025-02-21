"""
Initialisation tests for the code.
"""

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


class TestInstantiation:
    def test_connectome_reader_instantiation(self):
        """
        Test the instantiation of the ConnectomeReader class.
        """
        from vnc_networks.connectome_reader import FAFB, MANC

        # Should work
        valid_reader_brain = FAFB('v630')
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




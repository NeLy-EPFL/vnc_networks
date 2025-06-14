"""
Initialisation tests for the code.
"""

import pytest


class TestImport:
    def test_import(self):
        """
        Test the import of the module.
        """
        import vnc_networks

    def test_import_submodules(self):
        """
        Test the import of the submodules.
        """
        from vnc_networks import (
            cmatrix,
            connections,
            connectome_reader,
            neuron,
            params,
        )

    def test_import_classes(self):
        """
        Test the import of the classes.
        """
        from vnc_networks.cmatrix import CMatrix
        from vnc_networks.connections import Connections
        from vnc_networks.connectome_reader import FAFB, MANC
        from vnc_networks.neuron import Neuron

        assert True


class TestInstantiation:
    def test_connectome_reader_instantiation_fafb(self):
        """
        Test the instantiation of the ConnectomeReader class for FAFB.
        """
        import os

        from vnc_networks.connectome_reader import FAFB
        from vnc_networks.params import RAW_DATA_DIR

        # Check FAFB initialisation
        # Need to create the data folder for the connectome
        fafb_data_dir = os.path.join(
            RAW_DATA_DIR,
            "fafb",
            "v630",
        )
        fafb_data_exists = os.path.exists(fafb_data_dir)
        if not fafb_data_exists:
            os.makedirs(fafb_data_dir)

        valid_reader_brain = FAFB("v630")

        # cleanup
        if not fafb_data_exists:
            os.rmdir(fafb_data_dir)

    def test_connectome_reader_instantiation_manc(self):
        """
        Test the instantiation of the ConnectomeReader class for MANC.
        """
        import os

        from vnc_networks.connectome_reader import MANC
        from vnc_networks.params import RAW_DATA_DIR

        # Check FAFB initialisation
        # Need to create the data folder for the connectome
        manc_data_dir = os.path.join(
            RAW_DATA_DIR,
            "manc",
            "v1.2.1",
        )
        manc_data_exists = os.path.exists(manc_data_dir)
        if not manc_data_exists:
            os.makedirs(manc_data_dir)

        valid_reader = MANC("v1.2.1")

        assert valid_reader is not None
        body_id_name = valid_reader.sna("body_id")
        assert body_id_name == "bodyId", "Incorrect name conversion"

        # cleanup
        if not manc_data_exists:
            os.rmdir(manc_data_dir)

    def test_connectome_reader_instantiation_incorrect_version(self):
        from vnc_networks.connectome_reader import MANC

        # Should raise an error
        with pytest.raises(ValueError):
            invalid_reader = MANC("v1.1")  # type: ignore because we're explicitly testing that this throws an error

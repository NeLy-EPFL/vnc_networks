"""
Can only run when the data is available. Not run in the CI.
"""


class TestConnectomeReaderMANC:
    def test_version_mapping_MANCv1_2(self):
        from vnc_networks.connectome_reader import MANC

        connectome_reader = MANC("v1.2")
        assert connectome_reader.connectome_name == "manc", "Incorrect connectome name"
        assert connectome_reader.connectome_version == "v1.2.1", (
            "Incorrect connectome version. MANC v1.2 should map to v1.2.1"
        )

    def test_version_mapping_MANCv1_2_1(self):
        from vnc_networks.connectome_reader import MANC

        connectome_reader = MANC("v1.2.1")
        assert connectome_reader.connectome_name == "manc", "Incorrect connectome name"
        assert connectome_reader.connectome_version == "v1.2.1", (
            "Incorrect connectome version. MANC v1.2.1 should map to v1.2.1"
        )

    def test_version_mapping_MANCv1_2_3(self):
        from vnc_networks.connectome_reader import MANC

        connectome_reader = MANC("v1.2.3")
        assert connectome_reader.connectome_name == "manc", "Incorrect connectome name"
        assert connectome_reader.connectome_version == "v1.2.3", (
            "Incorrect connectome version. MANC v1.2.3 should map to v1.2.3"
        )

    def test_getting_counts_by_neuropil_MANCv1_2_1(self):
        """
        Test that we can get neuron and synapse counts by neuropil
        """
        import polars as pl
        import polars.testing

        import vnc_networks

        # Instantiate ConnectomeReader
        connectome_reader = vnc_networks.connectome_reader.MANC("v1.2.1")

        # check that this matches what we expect
        polars.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "downstream", [10000, 23458]
            ),
            pl.DataFrame(
                {
                    "body_id": [10000, 23458],
                    "CV": [703, 0],
                    "IntTct": [313, 0],
                    "LTct": [3181, 0],
                    "LegNp(T3)(R)": [0, 1688],
                }
            ),
        )
        polars.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "upstream", [10000, 23458]
            ),
            pl.DataFrame(
                {
                    "body_id": [10000, 23458],
                    "CV": [224, 0],
                    "IntTct": [185, 0],
                    "LTct": [1462, 0],
                    "LegNp(T3)(R)": [0, 685],
                }
            ),
        )
        polars.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil("pre", [10000, 23458]),
            pl.DataFrame(
                {
                    "body_id": [10000, 23458],
                    "CV": [138, 0],
                    "IntTct": [73, 0],
                    "LTct": [752, 0],
                    "LegNp(T3)(R)": [0, 207],
                }
            ),
        )
        polars.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil("post", [10000, 23458]),
            pl.DataFrame(
                {
                    "body_id": [10000, 23458],
                    "CV": [224, 0],
                    "IntTct": [185, 0],
                    "LTct": [1462, 0],
                    "LegNp(T3)(R)": [0, 685],
                }
            ),
        )
        polars.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "total_synapses", [10000, 23458]
            ),
            pl.DataFrame(
                {
                    "body_id": [10000, 23458],
                    "CV": [927, 0],
                    "IntTct": [498, 0],
                    "LTct": [4643, 0],
                    "LegNp(T3)(R)": [0, 2373],
                }
            ),
        )


class TestConnectomeReaderFAFB:
    def test_getting_counts_by_neuropil_FAFBv783(self):
        """
        Test that we can get neuron and synapse counts by neuropil
        """
        import polars as pl
        import polars.testing

        import vnc_networks

        # Instantiate ConnectomeReader
        connectome_reader = vnc_networks.connectome_reader.FAFB_v783()

        # check that this matches what we expect
        polars.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "downstream", [720575940627036426, 720575940633587552]
            ),
            pl.DataFrame(
                {
                    "body_id": [720575940627036426, 720575940633587552],
                    "LOP_L": [9, 0],
                    "LO_L": [2, 0],
                    "SLP_R": [0, 2],
                    "SMP_R": [0, 31],
                }
            ),
            check_column_order=False,
            check_row_order=False,
            check_dtypes=False,
        )
        polars.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "upstream", [720575940627036426, 720575940633587552]
            ),
            pl.DataFrame(
                {
                    "body_id": [720575940627036426, 720575940633587552],
                    "LOP_L": [3, 0],
                    "LO_L": [9, 0],
                    "SLP_R": [0, 11],
                    "SMP_R": [0, 13],
                }
            ),
            check_column_order=False,
            check_row_order=False,
            check_dtypes=False,
        )
        polars.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "pre", [720575940627036426, 720575940633587552]
            ),
            pl.DataFrame(
                {
                    "body_id": [720575940627036426, 720575940633587552],
                    "LOP_L": [14, 0],
                    "LO_L": [83, 0],
                    "SLP_R": [0, 33],
                    "SMP_R": [0, 75],
                }
            ),
            check_column_order=False,
            check_row_order=False,
            check_dtypes=False,
        )
        polars.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "post", [720575940627036426, 720575940633587552]
            ),
            pl.DataFrame(
                {
                    "body_id": [720575940627036426, 720575940633587552],
                    "LOP_L": [100, 0],
                    "LO_L": [8, 0],
                    "SLP_R": [0, 4],
                    "SMP_R": [0, 284],
                }
            ),
            check_column_order=False,
            check_row_order=False,
            check_dtypes=False,
        )
        polars.testing.assert_frame_equal(
            connectome_reader.get_synapse_counts_by_neuropil(
                "total_synapses", [720575940627036426, 720575940633587552]
            ),
            pl.DataFrame(
                {
                    "body_id": [720575940627036426, 720575940633587552],
                    "LOP_L": [114, 0],
                    "LO_L": [91, 0],
                    "SLP_R": [0, 37],
                    "SMP_R": [0, 359],
                }
            ),
            check_column_order=False,
            check_row_order=False,
            check_dtypes=False,
        )

    def test_get_synapse_df_FAFBv783(self):
        import polars as pl
        import polars.testing

        import vnc_networks

        # Instantiate ConnectomeReader
        connectome_reader = vnc_networks.connectome_reader.FAFB_v783()

        # check that this matches what we expect
        polars.testing.assert_frame_equal(
            connectome_reader.get_synapse_df(720575940625986035),
            pl.DataFrame(
                {
                    "synapse_id": list(range(18194614, 18194618 + 1)),
                    "start_bid": [720575940625986035] * 5,
                    "end_bid": [720575940613052200] * 5,
                    "X": [482078, 482094, 482426, 486034, 487160],
                    "Y": [140110, 139918, 140470, 143752, 143970],
                    "Z": [103400, 103480, 103080, 101040, 100760],
                },
                schema={
                    "synapse_id": pl.UInt64,
                    "start_bid": pl.UInt64,
                    "end_bid": pl.UInt64,
                    "X": pl.Int32,
                    "Y": pl.Int32,
                    "Z": pl.Int32,
                },
            ),
        )

    def test_get_synapse_neuropil_FAFBv783(self):
        import pytest

        import vnc_networks

        connectome_reader = vnc_networks.connectome_reader.FAFB_v783()
        with pytest.raises(NotImplementedError):
            connectome_reader.get_synapse_neuropil([1])

    def test_list_all_nodes_FAFBv783(self):
        import vnc_networks

        connectome_reader = vnc_networks.connectome_reader.FAFB_v783()

        assert len(connectome_reader.list_all_nodes()) == 139246

    def test_get_neuron_bodyids_FAFBv783(self):
        import vnc_networks

        connectome_reader = vnc_networks.connectome_reader.FAFB_v783()

        assert len(connectome_reader.get_neuron_bodyids()) == 139246
        assert (
            len(connectome_reader.get_neuron_bodyids({"class_1": "descending"})) == 1303
        )
        assert (
            len(
                connectome_reader.get_neuron_bodyids(
                    {"class_1": "descending", "side": "left"}
                )
            )
            == 646
        )
        assert (
            len(
                connectome_reader.get_neuron_bodyids(
                    {"class_1": "descending", "side": "mid"}
                )
            )
            == 0
        )

    def test_load_data_neuron_FAFBv783(self):
        import polars as pl
        import polars.testing

        import vnc_networks

        connectome_reader = vnc_networks.connectome_reader.FAFB_v783()

        polars.testing.assert_frame_equal(
            connectome_reader.load_data_neuron(
                720575940628842314, ["class_1", "class_2", "name", "neuropil"]
            ),
            pl.DataFrame(
                {
                    "class_1": ["central"],
                    "class_2": ["mAL"],
                    "name": ["mAL"],
                    "neuropil": ["SLP"],
                    "body_id": [720575940628842314],
                }
            ),
        )

    def test_load_data_neuron_set_FAFBv783(self):
        import polars as pl
        import polars.testing

        import vnc_networks

        connectome_reader = vnc_networks.connectome_reader.FAFB_v783()

        polars.testing.assert_frame_equal(
            connectome_reader.load_data_neuron_set(
                [720575940628842314, 720575940624547622],
                ["class_1", "class_2", "name", "neuropil"],
            ),
            pl.DataFrame(
                {
                    "class_1": ["central", "central"],
                    "class_2": ["mAL", "MBIN"],
                    "name": ["mAL", None],
                    "neuropil": ["SLP", "MB_ML"],
                    "body_id": [720575940628842314, 720575940624547622],
                }
            ),
        )

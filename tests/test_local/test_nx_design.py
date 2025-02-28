"""
Can only run when the data is available. Not run in the CI.
"""


class TestDisplayInteractiveGraph:
    """
    Test the data loading functions.
    """

    def _get_connections_subnetwork(self):
        from vnc_networks.connections import Connections
        from vnc_networks.connectome_reader import MANC

        # Instantiate a Connections object
        connections = Connections(CR=MANC("v1.2"))
        assert connections is not None, "Connections object not instantiated"

        dng11 = connections.get_neuron_ids({"type": "DNg11"})
        mns = connections.get_neuron_ids(
                {"class_1": "motor", "neuropil": "T1", "side": "LHS"}
            )

        paths = connections.paths_length_n(2, dng11, mns)
        subnetwork = connections.subgraph(paths.nodes, list(paths.edges))

        return dng11, mns, subnetwork

    def test_interactive_graph(self):
        """
        Test displaying an interactive graph from a connections object.
        """
        import tempfile

        from vnc_networks.utils.nx_design import display_interactive_graph

        _, _, subnetwork = self._get_connections_subnetwork()

        # save output to a temporary file so we don't have to clean up afterwards
        with tempfile.NamedTemporaryFile("w", suffix=".html") as f:
            display_interactive_graph(subnetwork, output_file=f.name)
        

    def test_interactive_graph_merged_nodes(self):
        """
        Test displaying an interactive graph from a connections object with merged nodes.
        """
        import tempfile

        from vnc_networks.utils.nx_design import display_interactive_graph

        neurons_from, _, subnetwork = self._get_connections_subnetwork()
        subnetwork.merge_nodes(neurons_from)

        # save output to a temporary file so we don't have to clean up afterwards
        with tempfile.NamedTemporaryFile("w", suffix=".html") as f:
            display_interactive_graph(subnetwork, output_file=f.name)

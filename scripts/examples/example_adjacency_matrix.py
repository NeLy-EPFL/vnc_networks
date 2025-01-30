import matplotlib.pyplot as plt
import specific_neurons.mdn_helper as mdn_helper

if __name__ == "__main__":
    neuropil = "hl"
    side = "RHS"

    # Load the VNC network
    VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
    )

    # Identify specific neurons
    source_neurons = mdn_helper.get_subdivided_mdns(VNC, neuropil, side)

    # right hind leg motor neurons
    target_neurons = VNC.get_neuron_ids(
        {
            "class_1": "motor",
            "side": side,
            "class_2": neuropil,
        }
    )

    # Get the graph made of neurons contacting MDNs -> motor neurons
    # within n hops
    n_hops = 2
    graph = VNC.paths_length_n(n_hops, source_neurons, target_neurons)

    # Get the graph with all existing connections between the interesting neurons
    # (not only the shortest paths)
    subconnections = VNC.subgraph(
        graph.nodes,
    )  # new Connections object

    # Get the adjacency matrix only
    # adjacency_matrix = subconnections.get_adjacency_matrix()
    # plt.imshow(adjacency_matrix)
    # plt.savefig('adjacency_matrix.png')

    # Get the adjacency matrix with tracking of the neurons names
    cmatrix = subconnections.get_cmatrix(type_="syn_count")
    print(cmatrix.get_lookup().head())
    cmatrix.imshow()
    plt.savefig("adjacency_matrix.png")

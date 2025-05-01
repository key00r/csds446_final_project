import numpy as np
import pickle
import networkx as nx
from tqdm import tqdm


class MethodShortestPath():
    """
    Computes shortest path distances between all nodes in the graph.
    These distances will be used for RoPE (Rotary Position Encoding).
    """
    dataset_name = None

    def __init__(self):
        self.data = None

    def run(self):
        """
        Main function to compute shortest path distances
        """
        print('Computing shortest path distances for RoPE...')

        # Extract graph structure from the data
        edges = self.data['edges']
        idx = self.data['idx']

        # Create a NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(idx)
        G.add_edges_from(edges)

        # Compute shortest path lengths between all pairs of nodes
        # Using all_pairs_shortest_path_length for efficiency
        shortest_paths = {}

        print('Computing shortest paths (this may take a while for large graphs)...')
        # For each node, compute shortest paths to all other nodes
        for node in tqdm(idx):
            # Get shortest path lengths from current node to all others
            path_lengths = nx.single_source_shortest_path_length(G, node)
            shortest_paths[node] = path_lengths

        # Return the shortest path distances
        return shortest_paths
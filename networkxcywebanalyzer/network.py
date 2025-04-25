import networkx as nx
from ndex2 import constants as ndex2constants
import numpy as np


def add_cytoscape_centralization_net_attrib(net_cx2=None, networkx_graph=None): # must review
    """
    Calculates network centralization EXACTLY matching Cytoscape's behavior:
    1. Uses degree centrality (normalized degree) instead of raw degrees
    2. Normalizes by (N-1) where N = number of nodes
    3. Handles directed graphs by using total degree (in + out)
    """
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")
    
    # Calculate degree centrality (normalized degree)
    if networkx_graph.is_directed():
        degrees = [d for n, d in networkx_graph.degree()]  # Total degree (in + out)
    else:
        degrees = [d for n, d in networkx_graph.degree()]
    
    N = len(networkx_graph.nodes())
    if N <= 1:
        centralization = 0.0
    else:
        # Normalize degrees (Cytoscape's key step)
        normalized_degrees = [d / (N - 1) for d in degrees]
        max_deg = max(normalized_degrees)
        avg_deg = sum(normalized_degrees) / N
        centralization = (max_deg - avg_deg)  # No additional normalization needed

    # Add to CX2 network
    net_cx2.add_network_attribute(
        key="Network Centralization",
        value=str(round(centralization, 3)),
        datatype=ndex2constants.STRING_DATATYPE  # Cytoscape stores as string
    )


def add_avg_neighbors_net_attrib(net_cx2=None):
    """ Computes the 'Average number of unique neighbors' matching the method
    used in the Cytoscape's Network Analyzer (disregards multiple edges).
    """
    # Get all edges and nodes
    edges = net_cx2.get_edges()
    nodes = net_cx2.get_nodes()
    
    # Track unique neighbors per node (ignoring edge multiplicities)
    neighbors = {n: set() for n in nodes} # Using a set to avoid duplicates
     
    # Populate neighbor sets
    for edge in edges.values():
        neighbors[edge['s']].add(edge['t'])
        neighbors[edge['t']].add(edge['s'])
    
    # Calculate average neighbors (unique connections)
    avg_neighbors = sum(len(items) for items in neighbors.values()) / len(nodes)

    # Add network attribute
    net_cx2.add_network_attribute(
        key="Avg. unique neighbors",
        value=str(round(avg_neighbors, 3)),
        datatype=ndex2constants.STRING_DATATYPE
    )

def add_heterogeneity_net_attrib(net_cx2=None, networkx_graph=None): # must review
    """
    Calculates the 'network heterogeneity' metric
    """
    degrees = [d for _, d in networkx_graph.degree()]
    heterogeneity = np.std(degrees) / np.mean(degrees)
    net_cx2.add_network_attribute(
        key="Network Heterogeneity",
        value=str(round(heterogeneity, 3)),
        datatype=ndex2constants.STRING_DATATYPE
    )
        
    
def add_characteristic_path_length_net_attrib(net_cx2=None, networkx_graph=None):
    """
    Computes the 'Characteristic path lenght' metric and handles cases where the network is not fully connected.
     by including calculation of the largest connected component
     """
    
    # Get all connected components
    connected_components = list(nx.connected_components(networkx_graph))
    
    if len(connected_components) == 1:
        # Network is connected
        cpl = nx.average_shortest_path_length(networkx_graph)
        net_cx2.add_network_attribute(
            key="Characteristic path length",
            value=str(round(cpl, 3)),
            datatype=ndex2constants.STRING_DATATYPE
        )
    else:
        # Network is disconnected - use largest component
        largest_component = max(connected_components, key=len)
        subgraph = networkx_graph.subgraph(largest_component)
        cpl = nx.average_shortest_path_length(subgraph)
        
        net_cx2.add_network_attribute(
            key="Characteristic path length",
            value=f"{round(cpl, 3)} (largest component)",
            datatype=ndex2constants.STRING_DATATYPE
        )
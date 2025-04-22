#!/usr/bin/env python

import os
import sys
import argparse
import json
import networkx as nx
from ndex2.cx2 import RawCX2NetworkFactory, CX2NetworkXFactory, CX2Network
from ndex2 import constants as ndex2constants
from itertools import combinations
from collections import defaultdict
import numpy as np



def _parse_arguments(desc, args):
    """
    Parses command line arguments
    :param desc:
    :param args:
    :return:
    """
    help_fm = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=help_fm)
    parser.add_argument('input',
                        help='Input: network in CX2 format')
    parser.add_argument('--mode',
                        choices=['analyze'],
                        default='analyze',
                        help='Mode. Default: analzye.')
    parser.add_argument('--outputonlycx2', action='store_true',
                        help='If set just output CX2 to standard out')
    return parser.parse_args(args)


def analyze_network(net_cx2):
    factory = CX2NetworkXFactory()
    networkx_graph = factory.get_graph(net_cx2, networkx_graph=nx.MultiGraph())
    networkx_degree = networkx_graph.degree()

    ### Network-level metrics ###
    net_cx2.add_network_attribute(key='Number of nodes', value=str(len(net_cx2.get_nodes())))                                                              
    net_cx2.add_network_attribute(key='Number of edges', value=str(len(net_cx2.get_edges())))                                                              
    add_avg_neighbors_net_attrib(net_cx2=net_cx2)
    net_cx2.add_network_attribute(key='Average degree',
                                  value=str(round(sum(dict(networkx_degree).values()) / networkx_graph.number_of_nodes(), 3)))

    add_eccentricity_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph) # includes diameter and radius metrics
    add_characteristic_path_length_net_attrib(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_multigraph_unsupported_metrics(net_cx2=net_cx2, networkx_graph=networkx_graph)
    net_cx2.add_network_attribute(key='Density', value=str(round(nx.density(networkx_graph), 3)))
    add_heterogeneity_net_attrib(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_cytoscape_centralization_net_attrib(net_cx2=net_cx2, networkx_graph=networkx_graph)
    net_cx2.add_network_attribute(key='Connected components', value=str(len(list(nx.connected_components(networkx_graph)))))

    ### Node-level metrics ###
    add_cytoscape_average_shortest_path_lenght(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_closeness_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_multiedge_partner_node_attribute(net_cx2=net_cx2)
    add_self_loops_node_attribute(net_cx2=net_cx2)
    add_cytoscape_stress_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_cytoscape_stress_node_attribute_2(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_degree_node_attribute(net_cx2=net_cx2, networkx_degrees=networkx_graph.degree())  # Total degree
    # Or use in_degree()/out_degree() for directional graphs
    
    add_degree_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_betweenness_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_neighborhood_connectivity_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_cytoscape_radiality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_topological_coefficient_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_cytoscape_topological_coefficient_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)

    if len(net_cx2.get_edges()) > 0:
        src_target_map = get_source_target_tuple_map(net_cx2=net_cx2)
        add_edge_betweeness_centrality(net_cx2=net_cx2, networkx_graph=networkx_graph,
                                       src_target_map=src_target_map)

    return net_cx2.to_cx2()
    

def get_source_target_tuple_map(net_cx2=None):
    """
    Builds map for both simple graphs and multigraphs.
    
    For simple graphs: (SRC, TARGET) => EDGE_ID
                      (TARGET, SRC) => EDGE_ID
    
    For multigraphs:   (SRC, TARGET, KEY) => EDGE_ID
                      (TARGET, SRC, KEY) => EDGE_ID
    """
    src_target_map = {}
    for edge_id, edge in net_cx2.get_edges().items():
        src = edge[ndex2constants.EDGE_SOURCE]
        target = edge[ndex2constants.EDGE_TARGET]
        
        # Get edge key if it exists (for multigraphs), default to 0 for simple graphs
        edge_key = edge.get('key', 0)
        
        # Create both (src, target) and (target, src) mappings
        src_target_map[(src, target, edge_key)] = edge_id
        src_target_map[(target, src, edge_key)] = edge_id  # Reverse direction
        
        # Maintain backward compatibility for simple graphs
        if edge_key == 0:  # Only add simple (u,v) mappings if this is definitely not a multigraph
            src_target_map[(src, target)] = edge_id
            src_target_map[(target, src)] = edge_id

    return src_target_map
    
    
# EDGE-LEVEL FUNCTIONS

def add_edge_betweeness_centrality(net_cx2=None, networkx_graph=None, src_target_map=None):
    """
    Adds edge betweenness centrality, working with both Graph and MultiGraph.
    """
    edge_betweenness = nx.edge_betweenness_centrality(networkx_graph)
    
    for nx_edge_id, val in edge_betweenness.items():
        # Handle both (u,v) and (u,v,k) edge identifiers
        if len(nx_edge_id) == 2:  # Simple graph edge
            u, v = nx_edge_id
            lookup_keys = [(u, v), (v, u)]  # Try both directions
        else:  # MultiGraph edge (u, v, k)
            u, v, k = nx_edge_id
            lookup_keys = [(u, v, k), (v, u, k)]  # Try both directions with key
            
        # Find the first matching key in our map
        edge_id = None
        for key in lookup_keys:
            if key in src_target_map:
                edge_id = src_target_map[key]
                break
                
        if edge_id is not None:
            net_cx2.add_edge_attribute(
                edge_id=edge_id,
                key='Betweenness Centrality',
                value=val,
                datatype=ndex2constants.DOUBLE_DATATYPE
            )
        else:
            sys.stderr.write(f"Warning: Could not find mapping for edge {nx_edge_id}\n")


# NODE-LEVEL FUNCTIONS

def add_cytoscape_topological_coefficient_node_attribute(net_cx2=None, networkx_graph=None):
    """Matches Cytoscape's topological coefficient exactly."""
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")
    
    tc = {}
    for node in networkx_graph.nodes():
        neighbors = set(networkx_graph.neighbors(node)) - {node}  # Key change: exclude self
        if len(neighbors) < 2:
            tc[node] = 0.0  # Cytoscape returns 0 for degree < 2
            continue
            
        total = 0
        for u, v in combinations(neighbors, 2):
            shared = (set(networkx_graph.neighbors(u)) & 
                     set(networkx_graph.neighbors(v))) - {node}  # Exclude self
            total += len(shared)
        
        denominator = len(neighbors) * (len(neighbors) - 1) / 2
        tc[node] = total / denominator if denominator > 0 else 0.0
    
    # Add to CX2
    for node_id in net_cx2.get_nodes():
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key='Cytoscape Topological Coefficient',
            value=float(tc.get(node_id, 0.0)),
            datatype=ndex2constants.DOUBLE_DATATYPE
        )

def add_cytoscape_radiality_node_attribute(net_cx2=None, networkx_graph=None):
    """Matches Cytoscape's radiality calculation exactly."""
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")
    
    # Use largest connected component (like Cytoscape)
    largest_cc = max(nx.connected_components(networkx_graph), key=len)
    subgraph = networkx_graph.subgraph(largest_cc)
    
    diameter = nx.diameter(subgraph) if len(largest_cc) > 1 else 0
    radiality = {}
    
    for node in subgraph.nodes():
        try:
            spl = nx.shortest_path_length(subgraph, source=node, weight='weight')  # Key change
            avg_dist = sum(spl.values()) / (len(subgraph) - 1)
            radiality[node] = (diameter + 1 - avg_dist) / diameter if diameter > 0 else 0
        except (nx.NetworkXError, ZeroDivisionError):
            radiality[node] = 0.0
    
    # Add to CX2 (assign 0 to nodes outside largest component)
    for node_id in net_cx2.get_nodes():
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key='Cytoscape Radiality',
            value=float(radiality.get(node_id, 0.0)),
            datatype=ndex2constants.DOUBLE_DATATYPE
        )

def add_cytoscape_stress_node_attribute(net_cx2=None, networkx_graph=None):
    """Calculates Stress Centrality matching Cytoscape's implementation exactly."""
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")
    
    stress = defaultdict(int)
    
    # Cytoscape counts ALL shortest paths (including through endpoints)
    for source in networkx_graph.nodes():
        # Use single_source_shortest_path instead of all_pairs
        paths = nx.single_source_shortest_path(networkx_graph, source)
        for target, path in paths.items():
            if source == target:
                continue  # Skip self-paths
            for node in path:  # Count all nodes in path (including endpoints)
                stress[node] += 1
    
    # Add to CX2 network
    for node_id in net_cx2.get_nodes():
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key='Cytoscape Stress',
            value=int(stress.get(node_id, 0)),
            datatype=ndex2constants.INTEGER_DATATYPE
        )

def add_cytoscape_stress_node_attribute_2(net_cx2=None, networkx_graph=None):
    """Calculates Stress Centrality matching Cytoscape's implementation exactly."""
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")
    
    stress = defaultdict(int)
    
    # Cytoscape counts ALL shortest paths (including through endpoints)
    for source in networkx_graph.nodes():
        # Use single_source_shortest_path instead of all_pairs
        paths = nx.single_source_shortest_path(networkx_graph, source)
        for target, path in paths.items():
            if source == target:
                continue  # Skip self-paths
            for node in path[1:-1]:  # excluding endpoints
                stress[node] += 1
    
    # Add to CX2 network
    for node_id in net_cx2.get_nodes():
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key='Cytoscape Stress (excl endpoints)',
            value=int(stress.get(node_id, 0)),
            datatype=ndex2constants.INTEGER_DATATYPE
        )

def add_cytoscape_stress_node_attribute_correct(net_cx2=None, networkx_graph=None):
    """Calculates Stress Centrality matching Cytoscape's implementation."""
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")
    
    stress = defaultdict(int)
    
    # Iterate over all node pairs
    nodes = list(networkx_graph.nodes())
    for i, source in enumerate(nodes):
        for target in nodes[i+1:]:  # Avoid duplicate pairs (undirected)
            if source == target:
                continue
            
            # Get ALL shortest paths between source and target
            all_paths = list(nx.all_shortest_paths(networkx_graph, source, target))
            
            # For each path, count intermediate nodes
            for path in all_paths:
                for node in path[1:-1]:  # Exclude endpoints
                    stress[node] += 1
    
    # Add to CX2 network
    for node_id in net_cx2.get_nodes():
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key='Cytoscape Stress (Correct)',
            value=int(stress.get(node_id, 0)),
            datatype=ndex2constants.INTEGER_DATATYPE
        )

def add_cytoscape_average_shortest_path_lenght(net_cx2=None, networkx_graph=None):
    """
    Replicates Cytoscape's node-level 'AverageShortestPathLength' analysis.
    """
    for node in networkx_graph.nodes():
        spl = nx.shortest_path_length(networkx_graph, source=node)
        avg_spl = sum(spl.values()) / (len(networkx_graph) - 1)
        
        net_cx2.add_node_attribute(
            node_id=int(node),
            key='Average Shortest Path Length',
            value=float(avg_spl),
            datatype=ndex2constants.DOUBLE_DATATYPE
        )

def add_degree_node_attribute(net_cx2=None, networkx_degrees=None):
    """
    Adds node degree as a node attribute to a CX2 network.
    
    Args:
        net_cx2 (ndex2.cx2.CX2Network): The CX2 network to modify
        networkx_degrees (dict or DegreeView): Node degrees from NetworkX's degree() 
            (either dict or DegreeView object)
    """
    # Input validation
    if net_cx2 is None or networkx_degrees is None:
        raise ValueError("Both net_cx2 and networkx_degrees must be provided")
    
    # Convert DegreeView to dict if needed
    degrees_dict = dict(networkx_degrees)
    
    for node_id, degree in degrees_dict.items():
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key='Degree',
            value=int(degree),
            datatype=ndex2constants.INTEGER_DATATYPE
        )
        
def add_degree_centrality_node_attribute(net_cx2=None, networkx_graph=None):
    """
    Adds 'node degree_centraility' node attribute
    """
    node_centrality = nx.degree_centrality(networkx_graph)
    for node_id, val in node_centrality.items():
        # sys.stderr.write('nodeid: ' + str(node_id) + ' => ' + str(val) + '\n')
        net_cx2.add_node_attribute(node_id=int(node_id), key='Degree Centrality',
                                   value=val,
                                   datatype=ndex2constants.DOUBLE_DATATYPE)

def add_betweenness_centrality_node_attribute(net_cx2=None, networkx_graph=None):
    """
    Adds 'node betweenness centrality' node attribute
    """
    betweenness_centrality = nx.betweenness_centrality(networkx_graph)
    for node_id, val in betweenness_centrality.items():
        # sys.stderr.write('nodeid: ' + str(node_id) + ' => ' + str(val) + '\n')
        net_cx2.add_node_attribute(node_id=int(node_id), key='Betweenness Centrality',
                                   value=val,
                                   datatype=ndex2constants.DOUBLE_DATATYPE)

def add_closeness_centrality_node_attribute(net_cx2=None, networkx_graph=None):
    """
    Adds 'node closeness centrality' node attribute
    """
    closeness_centrality = nx.closeness_centrality(networkx_graph)
    for node_id, val in closeness_centrality.items():
        # sys.stderr.write('nodeid: ' + str(node_id) + ' => ' + str(val) + '\n')
        net_cx2.add_node_attribute(node_id=int(node_id), key='Closeness Centrality',
                                   value=val,
                                   datatype=ndex2constants.DOUBLE_DATATYPE)

def add_multigraph_unsupported_metrics(net_cx2=None, networkx_graph=None):
    """
    Adds node and networks attributes that are not implemented for the MultiGraph() NetworkX class.
    The metrics handled in this function are:
    
    - Clustering coefficient (node) > simple graph, disregard multiple edges
    - Eigenvector centrality (node) > weighted graph, accounts for multiple edges
    - Average clustering coefficient (network, pure Python implementation)
    - Transitivity (network)
    
    """
    # 1. Create a weighted Graph() object from the unweighted MultiGraph()
    G_simple = nx.Graph(networkx_graph) # use this to compute clustering and match Cytsocape's logic
    G_w = nx.Graph() # use this to compute Eigenvector using wheights
    for u, v in networkx_graph.edges():
        if G_w.has_edge(u, v):
            G_w[u][v]['weight'] += 1
        else:
            G_w.add_edge(u, v, weight=1)
    
    # 2. Compute per‑node "clustering coeff" and "Eigenvector centrality", using the Graph() class and 'weight' attribute
    clustering_coeff = nx.clustering(G_simple)
    eigenvector = nx.eigenvector_centrality(G_w, weight='weight')
    
    # 3. Compute "average clustering coefficient" and "transitivity"
    avg_clustering_coeff = str(round(sum(clustering_coeff.values()) / len(clustering_coeff), 3))
    transitivity = str(round(nx.transitivity(G_w), 3))
    
    # 4. Set network-level attributes
    net_cx2.add_network_attribute(key=' Avg. clustering Coefficient', value=str(avg_clustering_coeff))
    net_cx2.add_network_attribute(key='Transitivity', value=str(transitivity))
    
    # 5. Set node-level attributes
    for node_id, val in clustering_coeff.items():
        # sys.stderr.write('nodeid: ' + str(node_id) + ' => ' + str(val) + '\n')
        net_cx2.add_node_attribute(node_id=int(node_id), key='Clustering Coefficient',
                                   value=val,
                                   datatype=ndex2constants.DOUBLE_DATATYPE)

    for node_id, val in eigenvector.items():
        # sys.stderr.write('nodeid: ' + str(node_id) + ' => ' + str(val) + '\n')
        net_cx2.add_node_attribute(node_id=int(node_id), key='Eigenvector Centrality',
                                   value=val,
                                   datatype=ndex2constants.DOUBLE_DATATYPE)
        

def add_eccentricity_attribute(net_cx2=None, networkx_graph=None):
    """
    Adds 'node eccentricity' node attribute as well as Radius and Diameter network attributes (Max and Min Eccentricities).
    Preserves multi edges and ensures Metrics are computed on largest connected component, like Cytoscape.
    """
    largest_cc = max(nx.connected_components(networkx_graph), key=len)
    subnetwork = networkx_graph.subgraph(largest_cc)
    
    eccentricities = nx.eccentricity(subnetwork)
    
    for node_id, val in eccentricities.items():
        # sys.stderr.write('nodeid: ' + str(node_id) + ' => ' + str(val) + '\n')
        net_cx2.add_node_attribute(node_id=int(node_id), key='Eccentricity',
                                   value=val,
                                   datatype=ndex2constants.INTEGER_DATATYPE)

    net_cx2.add_network_attribute(key='Network diameter', value=str(max(eccentricities.values())))
    net_cx2.add_network_attribute(key='Network radius', value=str(min(eccentricities.values())))

def add_self_loops_node_attribute(net_cx2=None):
    """
    Counts self-loops for each node and adds them as node attributes in a CX2 network.
    
    Args:
        net_cx2: A CX2 network object from ndex2
        
    Raises:
        ValueError: If no network object is provided
    """
    if net_cx2 is None:
        raise ValueError("A CX2 network object must be passed as argument to this function")

    self_loop_counts = {}
    
    # Count self-loops
    for _, edge in net_cx2.get_edges().items(): # _ replaces the edge_id variable that is unused in the code
        if edge['s'] == edge['t']:  # Self-loop detected
            self_loop_counts[edge['s']] = self_loop_counts.get(edge['s'], 0) + 1

    # Add attributes
    for node_id in net_cx2.get_nodes():
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key="Self Loops",
            value=self_loop_counts.get(node_id, 0),
            datatype=ndex2constants.INTEGER_DATATYPE
        )

def add_topological_coefficient_node_attribute(net_cx2=None, networkx_graph=None):
    """
    Adds 'Topological Coefficient' node attribute to CX2 network, which is a measure
    of how interconnected a node's neighbors are with each other.
    
    Args:
        net_cx2: CX2 network object
        networkx_graph: NetworkX graph object (for efficient path calculations)
        
    Raises:
        ValueError: If inputs are missing or invalid
    """
    # Input validation
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both CX2 network and NetworkX graph objects must be provided")

    # Calculate topological coefficient for each node
    tc = {}
    for node in networkx_graph.nodes():
        neighbors = list(networkx_graph.neighbors(node))
        neighbor_pairs = list(combinations(neighbors, 2))
        
        if len(neighbor_pairs) == 0:
            tc[node] = 0.0
            continue
            
        shared_neighbors = 0
        for u, v in neighbor_pairs:
            # Count common neighbors between each neighbor pair
            shared_neighbors += len(list(nx.common_neighbors(networkx_graph, u, v)))
            
        # Normalize by number of possible neighbor pairs
        tc[node] = shared_neighbors / len(neighbor_pairs)

    # Add to CX2 network
    for node_id, value in tc.items():
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key='Topological Coefficient',
            value=float(value),
            datatype=ndex2constants.DOUBLE_DATATYPE
        )

def add_neighborhood_connectivity_node_attribute(net_cx2=None, networkx_graph=None):
    """Calculates and adds neighborhood connectivity as a node attribute to a CX2 network.
    
    Neighborhood connectivity measures the average degree of a node's neighbors. 
    Higher values indicate nodes connected to well-connected neighbors (network hubs).
    
    Args:
        net_cx2 (ndex2.cx2.CX2Network): The CX2 network object to modify.
        networkx_graph (networkx.Graph): NetworkX graph for calculations.
        
    Raises:
        ValueError: If networks are None, empty, or node sets mismatch.
    """
    # Input validation
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both network objects must be provided")
    if len(networkx_graph) == 0 or len(net_cx2.get_nodes()) == 0:
        raise ValueError("Networks cannot be empty")
    if set(net_cx2.get_nodes()) != set(networkx_graph.nodes()):
        raise ValueError("Node sets between CX2 and NetworkX graphs must match")

    # Calculate neighborhood connectivity
    nc = {}
    for node in networkx_graph.nodes():
        neighbors = list(networkx_graph.neighbors(node))
        
        if not neighbors:  # Handle isolated nodes
            nc[node] = 0.0
            continue
            
        # Calculate average neighbor degree
        total = sum(networkx_graph.degree(n) for n in neighbors)
        nc[node] = total / len(neighbors)

    # Add to CX2 network
    for node_id, value in nc.items():
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key='Neighborhood Connectivity',
            value=float(value),
            datatype=ndex2constants.DOUBLE_DATATYPE
        )

def add_multiedge_partner_node_attribute(net_cx2=None):
    """Calculates and adds 'Partner of Multi-edged Node Pairs' as a node attribute."""
    # Input validation
    if net_cx2 is None:
        raise ValueError("CX2 network object must be provided")
    if len(net_cx2.get_nodes()) == 0:
        raise ValueError("Network cannot be empty")

    # Step 1: Get edges and identify multi-edge pairs
    edge_counts = defaultdict(int)
    edges_dict = net_cx2.get_edges()  # Returns dictionary of edges
    
    # Check edge dictionary structure
    if not isinstance(edges_dict, dict):
        raise TypeError("get_edges() must return a dictionary")
    
    # Iterate through edge dictionary (key=edge_id, value=edge_data)
    for edge_id, edge_data in edges_dict.items():
        try:
            u, v = edge_data['s'], edge_data['t']  # Access source/target from edge data
            u, v = sorted((int(u), int(v)))  # Ensure consistent ordering
            edge_counts[(u, v)] += 1
        except (KeyError, TypeError) as e:
            raise ValueError(f"Edge {edge_id} has invalid format: {str(e)}")

    # Step 2: Count multi-edge partnerships (same as before)
    multi_edge_pairs = {pair for pair, count in edge_counts.items() 
                       if count > 1 and pair[0] != pair[1]}
    
    node_scores = defaultdict(int)
    for u, v in multi_edge_pairs:
        node_scores[u] += 1
        node_scores[v] += 1

    # Step 3: Add attributes
    for node_id in net_cx2.get_nodes():
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key='Partner of Multi-edged Node Pairs',
            value=int(node_scores.get(node_id, 0)),
            datatype=ndex2constants.INTEGER_DATATYPE
        )
                                       
# NETWORK-LEVEL FUNCTIONS

def add_cytoscape_centralization_net_attrib(net_cx2=None, networkx_graph=None):
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
        key="Cytoscape Network Centralization",
        value=str(round(centralization, 3)),
        datatype=ndex2constants.STRING_DATATYPE  # Cytoscape stores as string
    )


def add_avg_neighbors_net_attrib(net_cx2=None):
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
        key="Avg. number of neighbors",
        value=str(round(avg_neighbors, 3)),
        datatype=ndex2constants.STRING_DATATYPE
    )

def add_heterogeneity_net_attrib(net_cx2=None, networkx_graph=None):
    degrees = [d for _, d in networkx_graph.degree()]
    heterogeneity = np.std(degrees) / np.mean(degrees)
    net_cx2.add_network_attribute(
        key="Network Heterogeneity",
        value=str(round(heterogeneity, 3)),
        datatype=ndex2constants.STRING_DATATYPE
    )
        
def add_centralization_net_attrib(net_cx2=None, networkx_graph=None):
    degrees = [d for _, d in networkx_graph.degree()]
    centralization = (max(degrees) - np.mean(degrees)) / (len(networkx_graph) - 1)
    net_cx2.add_network_attribute(
        key="Network Centralization",
        value=str(round(centralization, 3)),
        datatype=ndex2constants.STRING_DATATYPE
    )
    
def add_characteristic_path_length_net_attrib(net_cx2=None, networkx_graph=None):
    try:
        cpl = nx.average_shortest_path_length(networkx_graph)
        net_cx2.add_network_attribute(
            key="Characteristic path length",
            value=str(round(cpl, 3)),
            datatype=ndex2constants.STRING_DATATYPE
        )
    except nx.NetworkXError:  # Disconnected graph
        net_cx2.add_network_attribute(
            key="Characteristic Path Length",
            value="undefined (disconnected)",
            datatype=ndex2constants.STRING_DATATYPE
        )

####### END OF METRICS #######


def get_cx2_net_from_input(input_path):
    net_cx2_path = os.path.abspath(input_path)
    factory = RawCX2NetworkFactory()
    return factory.get_cx2network(net_cx2_path)


def main(args):
    """
    Main entry point for program

    :param args: command line arguments usually :py:const:`sys.argv`
    :return: 0 for success otherwise failure
    :rtype: int
    """
    desc = """
    TODO
    """

    theargs = _parse_arguments(desc, args[1:])
    try:
        theres = None

        if theargs.mode == 'analyze':
            net_cx2 = get_cx2_net_from_input(theargs.input)
            theres = analyze_network(net_cx2)

        if theres is None:
            sys.stderr.write('No results\n')
        else:
            if theargs.outputonlycx2 is True:
                newres = theres
            else:
                newres = [{'action': 'updateNetwork',
                           'data': theres}]
            json.dump(newres, sys.stdout, indent=2)
        sys.stdout.flush()
        sys.stderr.flush()

        return 0
    except Exception as e:
        sys.stderr.write('Caught exception: ' + str(e))
        sys.stderr.flush()
        return 2


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv))

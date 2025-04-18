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
    networkx_graph = factory.get_graph(net_cx2, networkx_graph=nx.Graph())
    networkx_degree = networkx_graph.degree()

    # Network-level metrics
    net_cx2.add_network_attribute(key='Number of Nodes', value=str(len(net_cx2.get_nodes())))
                                                                   
    net_cx2.add_network_attribute(key='Number of Edges', value=str(len(net_cx2.get_edges())))
                                                                   
    add_avg_neighbors_net_attrib(net_cx2=net_cx2)
    
    net_cx2.add_network_attribute(key='Average Degree',
                                  value=str(round(sum(dict(networkx_degree).values()) / networkx_graph.number_of_nodes(), 3)))

    net_cx2.add_network_attribute(key='Diameter', value=str(nx.diameter(networkx_graph)))

    net_cx2.add_network_attribute(key='Diameter (Max. Eccentricity)', value=str(max(nx.eccentricity(networkx_graph).values())))

    add_characteristic_path_length_net_attrib(net_cx2=net_cx2, networkx_graph=networkx_graph)
    
    net_cx2.add_network_attribute(key=' Average Clustering Coefficient', value=str(round(nx.average_clustering(networkx_graph), 3)))
    
    net_cx2.add_network_attribute(key='Density', value=str(round(nx.density(networkx_graph), 3)))

    add_heterogeneity_net_attrib(net_cx2=net_cx2, networkx_graph=networkx_graph)

    add_centralization_net_attrib(net_cx2=net_cx2, networkx_graph=networkx_graph)

    net_cx2.add_network_attribute(key='Transitivity', value=str(round(nx.transitivity(networkx_graph), 3)))

    # Node-level metrics
    add_cytoscape_avg_spl(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_clustering_coeficient_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_closeness_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_multiedge_partner_node_attribute(net_cx2=net_cx2)
    add_self_loops_node_attribute(net_cx2=net_cx2)
    add_eccentricity_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_stress_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_degree_node_attribute(net_cx2=net_cx2, networkx_degrees=networkx_graph.degree())  # Total degree
    # Or use in_degree()/out_degree() for directional graphs
    
    add_degree_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_betweenness_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_eigenvector_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_neighborhood_connectivity_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_radiality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_topological_coefficient_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    
    if len(net_cx2.get_edges()) > 0:
        src_target_map = get_source_target_tuple_map(net_cx2=net_cx2)
        add_edge_betweeness_centrality(net_cx2=net_cx2, networkx_graph=networkx_graph,
                                       src_target_map=src_target_map)

    return net_cx2.to_cx2()


def get_source_target_tuple_map(net_cx2=None):
    """
    Builds map

               (SRC, TARGET)
                              => EDGE_ID
               (TARGET, SRC)
    """
    src_target_map = {}
    for edge_id, edge in net_cx2.get_edges().items():
        src_target_map[(edge[ndex2constants.EDGE_SOURCE], edge[ndex2constants.EDGE_TARGET])] = edge_id
        src_target_map[(edge[ndex2constants.EDGE_TARGET], edge[ndex2constants.EDGE_SOURCE])] = edge_id

    return src_target_map
    
# NETWORK-LEVEL FUNCTIONS

def add_avg_neighbors_net_attrib(net_cx2=None):
    avg_deg = 2 * len(net_cx2.get_edges()) / len(net_cx2.get_nodes())
    net_cx2.add_network_attribute(
        key="Average Neighbors",
        value=str(round(avg_deg, 3)),
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
            key="Characteristic Path Length",
            value=str(round(cpl, 3)),
            datatype=ndex2constants.STRING_DATATYPE
        )
    except nx.NetworkXError:  # Disconnected graph
        net_cx2.add_network_attribute(
            key="Characteristic Path Length",
            value="undefined (disconnected)",
            datatype=ndex2constants.STRING_DATATYPE
        )

# EDGE-LEVEL FUNCTIONS

def add_edge_betweeness_centrality(net_cx2=None, networkx_graph=None,
                                   src_target_map=None):
    """
    Adds "edge betweenness centrality' edge attribute
    """
    edge_betweenness = nx.edge_betweenness_centrality(networkx_graph)
    for nxedge_id, val in edge_betweenness.items():
        sys.stderr.write(str(src_target_map) + '\n\n')
        sys.stderr.write('edge id: ' + str(nxedge_id) + ' => ' + str(val) + '\n')
        net_cx2.add_edge_attribute(edge_id=src_target_map[nxedge_id], key='Betweenness Centrality',
                                   value=val, datatype=ndex2constants.DOUBLE_DATATYPE)

# NODE-LEVEL FUNCTIONS

def add_cytoscape_avg_spl(net_cx2=None, networkx_graph=None):
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

def add_clustering_coeficient_node_attribute(net_cx2=None, networkx_graph=None):
    """
    Adds 'node clustering coefficient' node attribute
    """
    clustering_coeff = nx.clustering(networkx_graph)
    for node_id, val in clustering_coeff.items():
        # sys.stderr.write('nodeid: ' + str(node_id) + ' => ' + str(val) + '\n')
        net_cx2.add_node_attribute(node_id=int(node_id), key='Clustering Coefficient',
                                   value=val,
                                   datatype=ndex2constants.DOUBLE_DATATYPE)

def add_eigenvector_centrality_node_attribute(net_cx2=None, networkx_graph=None):
    """
    Adds 'node eigenvector centrality' node attribute
    """
    eigenvector = nx.eigenvector_centrality(networkx_graph)
    for node_id, val in eigenvector.items():
        # sys.stderr.write('nodeid: ' + str(node_id) + ' => ' + str(val) + '\n')
        net_cx2.add_node_attribute(node_id=int(node_id), key='Eigenvector Centrality',
                                   value=val,
                                   datatype=ndex2constants.DOUBLE_DATATYPE)

def add_eccentricity_node_attribute(net_cx2=None, networkx_graph=None):
    """
    Adds 'node eccentricity' node attribute
    """
    eccentricity = nx.eccentricity(networkx_graph)
    for node_id, val in eccentricity.items():
        # sys.stderr.write('nodeid: ' + str(node_id) + ' => ' + str(val) + '\n')
        net_cx2.add_node_attribute(node_id=int(node_id), key='Eccentricity',
                                   value=val,
                                   datatype=ndex2constants.DOUBLE_DATATYPE)
        

def add_stress_node_attribute(net_cx2=None, networkx_graph=None):
    """
    Adds 'Stress' node attribute to CX2 network by calculating:
    - Number of shortest paths passing through each node (excluding source/target)
    """
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")

    # Calculate stress centrality
    stress = {n: 0 for n in networkx_graph.nodes()}
    all_pairs = nx.all_pairs_shortest_path(networkx_graph)
    
    for source, paths in all_pairs:
        for target, path in paths.items():
            if source != target:
                for node in path[1:-1]:  # Exclude source and target
                    stress[node] += 1

    # Add attributes to CX2 network
    for node_id, val in stress.items():
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key='Stress',
            value=int(val),  # Ensure integer
            datatype=ndex2constants.INTEGER_DATATYPE
        )

def add_radiality_node_attribute(net_cx2=None, networkx_graph=None):
    """
    Adds 'node radiality' node attribute
    """
    # Calculate diameter (handle disconnected graphs)
    try:
        diameter = nx.diameter(networkx_graph)
    except nx.NetworkXError:  # Disconnected graph
        diameter = max([nx.diameter(subgraph) for subgraph in nx.connected_components(networkx_graph)])

    # Compute radiality for each node
    radiality = {}
    num_nodes = len(networkx_graph.nodes())
    
    for node in networkx_graph.nodes():
        try:
            total_distance = sum(nx.shortest_path_length(networkx_graph, source=node).values())
            radiality[node] = (diameter + 1 - (total_distance / (num_nodes - 1))) / diameter
        except nx.NetworkXError:  # Isolated node
            radiality[node] = 0.0  # Default value for isolated nodes

    # Add radiality as node attributes to CX2 network
    for node_id, val in radiality.items():
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key='Radiality',
            value=float(val),
            datatype=ndex2constants.DOUBLE_DATATYPE
        )

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

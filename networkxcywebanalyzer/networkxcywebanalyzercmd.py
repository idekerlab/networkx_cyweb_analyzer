#!/usr/bin/env python

import os
import sys
import argparse
import json
import networkx as nx
from ndex2.cx2 import RawCX2NetworkFactory, CX2NetworkXFactory
from ndex2 import constants as ndex2constants
from networkxcywebanalyzer.node import *
from networkxcywebanalyzer.network import *


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
    add_centralization_net_attrib(net_cx2=net_cx2, networkx_graph=networkx_graph)

    net_cx2.add_network_attribute(key='Connected components', value=str(len(list(nx.connected_components(networkx_graph)))))

    ### Node-level metrics ###
    add_cytoscape_average_shortest_path_lenght(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_closeness_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_multiedge_partner_node_attribute(net_cx2=net_cx2)
    add_self_loops_node_attribute(net_cx2=net_cx2)
    
    add_cytoscape_stress_node_attribute_correct(net_cx2=net_cx2, networkx_graph=networkx_graph) #exclude endpoints
    add_cytoscape_stress_node_attribute_correct_2(net_cx2=net_cx2, networkx_graph=networkx_graph) #include endpoints
    add_cytoscape_stress_node_attribute_3(net_cx2=net_cx2, networkx_graph=networkx_graph) #exclude endpoints
    add_cytoscape_stress_node_attribute_4(net_cx2=net_cx2, networkx_graph=networkx_graph) #include endpoints
    
    add_degree_node_attribute(net_cx2=net_cx2, networkx_degrees=networkx_graph.degree())  # Total degree
    # Or use in_degree()/out_degree() for directional graphs
    
    add_degree_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_betweenness_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_neighborhood_connectivity_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_cytoscape_radiality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_topological_coefficient_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_cytoscape_topological_coefficient_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)

    ### Edge-level metrics ###
    if len(net_cx2.get_edges()) > 0:
        src_target_map = get_source_target_tuple_map(net_cx2=net_cx2)
        add_edge_betweeness_centrality(net_cx2=net_cx2, networkx_graph=networkx_graph,
                                       src_target_map=src_target_map)

    return net_cx2.to_cx2()
    

##### EDGE-LEVEL FUNCTIONS #####

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

##### END OF EDGE-LEVEL FUNCTIONS #####


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
#!/usr/bin/env python

import os
import sys
import argparse
import json
import logging
import networkx as nx
from ndex2.cx2 import RawCX2NetworkFactory, CX2NetworkXFactory
from ndex2 import constants as ndex2constants
from networkxcywebanalyzer.node import *
from networkxcywebanalyzer.network import *


logger = logging.getLogger(__name__)


LOG_FORMAT = "%(asctime)-15s %(levelname)s %(relativeCreated)dms " \
             "%(filename)s::%(funcName)s():%(lineno)d %(message)s"

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
                        help='Mode. Default: analyze.')
    parser.add_argument('--isdirected', action='store_true',
                        help='If set, graph is considered a directed graph')
    parser.add_argument('--outputonlycx2', action='store_true',
                        help='If set just output CX2 to standard out')
    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger. (default None)')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help='Increases verbosity of logger to standard '
                             'error for log messages in this module. Messages are '
                             'output at these python logging levels '
                             '-v = WARNING, -vv = INFO, '
                             '-vvv = DEBUG, -vvvv = NOTSET (default ERROR '
                             'logging)')
    return parser.parse_args(args)


def setup_cmd_logging(args):
    """
    Sets up logging based on parsed command line arguments.
    If **args.logconf** is set use that configuration otherwise look
    at **args.verbose** and set logging for this module

    This function assumes the following:

    * **args.logconf** exists and is ``None`` or set to :py:class:`str`
      containing path to logconf file

    * **args.verbose** exists and is set to :py:class:`int` to one of
      these values:

      * ``0`` = no logging
      * ``1`` = critical
      * ``2`` = error
      * ``3`` = warning
      * ``4`` = info
      * ``5`` = debug

    :param args: parsed command line arguments from argparse
    :type args: :py:class:`argparse.Namespace`
    :raises AttributeError: If args is ``None`` or
                            if **args.logconf** is None or missing or
                            if **args.verbose** is None or missing
    """

    if args.logconf is None:
        level = (50 - (10 * args.verbose))
        logging.basicConfig(format=LOG_FORMAT,
                            level=level)
        logger.setLevel(level)
        return

    # logconf was set use that file
    logging.config.fileConfig(args.logconf,
                              disable_existing_loggers=False)


def analyze_network(net_cx2, isdirected=False):
    factory = CX2NetworkXFactory()
    if isdirected is True:
        logger.debug('Creating multi di graph')
        nxgraph = nx.MultiDiGraph()
    else:
        logger.debug('Creating multi graph')
        nxgraph = nx.MultiGraph()

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
    Builds a lookup table to get CX2 network edge id by passing in a tuple
    of (SRC, TARGET), or (TARGET, SRC) where SRC and TARGET are
    ids of nodes in CX2 network and value is a set of edge ids

    .. code-block::

        # if a CX2Network had the following edge:
        # {"id":5,"s":0,"t":1,"v":{"interaction":"some interaction"}}
        # it would be mapped two these two keys:

        {(0, 1): {5}, (1, 0): {5}}

    :param net_cx2:
    :type net_cx2: :py:class:`ndex2.cx2.CX2Network`
    :return: Map of src target tuples to edge ids
    :rtype: dict
    """
    src_target_map = {}

    for edge_id, edge in net_cx2.get_edges().items():
        src = edge[ndex2constants.EDGE_SOURCE]
        target = edge[ndex2constants.EDGE_TARGET]
        if (src, target) not in src_target_map:
            src_target_map[((src, target))] = set()
        if (target, src) not in src_target_map:
            src_target_map[(target, src)] = set()
        src_target_map[(src, target)].add(edge_id)
        src_target_map[(target, src)].add(edge_id)

    return src_target_map
    

def add_edge_betweeness_centrality(net_cx2=None, networkx_graph=None,
                                   src_target_map=None):
    """
    Adds edge betweenness centrality to **net_cx2** network
    as an edge attribute named ``Betweenness Centrality``

    :param net_cx2: Network to analyze
    :type net_cx2: :py:class:`ndex2.cx2.CX2Network`
    :param networkx_graph: Networkx version of network
    :type networkx_graph: :py:class:`networkx.Graph`
    :param src_target_map: map of (src,target) => edge id
                                  (target,src) => edge id
    :type src_target_map: dict
    """
    edge_betweenness = nx.edge_betweenness_centrality(networkx_graph)
    for nx_edge_id, val in edge_betweenness.items():
        src_target = None
        target_src = None
        # Handle both (u,v) and (u,v,k) edge identifiers
        if len(nx_edge_id) == 2:  # Simple graph edge
            u, v = nx_edge_id
        else:  # MultiGraph edge (u, v, k)
            u, v, _ = nx_edge_id
        src_target = (u, v)
        target_src = (v, u)

        for entry in [src_target, target_src]:
            if src_target not in src_target_map:
                logger.debug(f"Could not find mapping for edge {src_target}. "
                             f"Skipping addition of attribute")
                continue
            for edge_id in src_target_map[entry]:
                net_cx2.add_edge_attribute(
                    edge_id=edge_id,
                    key='Betweenness Centrality',
                    value=val,
                    datatype=ndex2constants.DOUBLE_DATATYPE)

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
    setup_cmd_logging(theargs)
    try:
        theres = None

        if theargs.mode == 'analyze':
            net_cx2 = get_cx2_net_from_input(theargs.input)
            theres = analyze_network(net_cx2,
                                     isdirected=theargs.isdirected)

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
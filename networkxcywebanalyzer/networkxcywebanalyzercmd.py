#!/usr/bin/env python

import os
import sys
import argparse
import json
import logging
import traceback
import networkx as nx

from ndex2.cx2 import RawCX2NetworkFactory, CX2NetworkXFactory
from ndex2 import constants as ndex2constants
import networkxcywebanalyzer
from networkxcywebanalyzer.network import *
from networkxcywebanalyzer.node import *
from networkxcywebanalyzer.edge import *


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
    parser.add_argument('--namespace', default='NXA',
                        help='Cytoscape namespace prefix to use, every attribute will '
                             'have this value with :: prepended')
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


def get_networkx_graph_with_keys(net_cx2, isdirected=False):
    """

    """
    if isdirected is True:
        networkx_graph = nx.MultiDiGraph()
    else:
        networkx_graph = nx.MultiGraph()

    for node_id, node_data in net_cx2.get_nodes().items():
        networkx_graph.add_node(node_id)

    for edge_id, edge_data in net_cx2.get_edges().items():
        source = edge_data[ndex2constants.EDGE_SOURCE]
        target = edge_data[ndex2constants.EDGE_TARGET]
        attrs = edge_data.get(ndex2constants.ASPECT_VALUES, {})
        networkx_graph.add_edge(source, target, key=edge_id)

    for attr, value in net_cx2.get_network_attributes().items():
        networkx_graph.graph[attr] = value
    return networkx_graph


def analyze_network(net_cx2, isdirected=False, namespace=None):
    factory = CX2NetworkXFactory()
    if namespace is not None:
        keyprefix = namespace + '::'
    else:
        keyprefix = ''

    net_cx2.remove_network_attribute('name')
    
    net_cx2.add_network_attribute(
        key='Directed Network',
        value=str(isdirected),
        datatype=ndex2constants.STRING_DATATYPE
    )

    networkx_graph = get_networkx_graph_with_keys(net_cx2=net_cx2, isdirected=isdirected)
    networkx_degree = networkx_graph.degree()

    ### Calls for Network-level metrics ###
    add_number_of_nodes(net_cx2=net_cx2, keyprefix=keyprefix)
    
    add_number_of_edges(net_cx2=net_cx2, keyprefix=keyprefix)
    
    add_avg_neighbors_net_attrib(net_cx2=net_cx2, networkx_graph=networkx_graph, keyprefix=keyprefix) # includes calculation of components coverage (node %, etc)
        
    add_avg_degree_net_attrib(net_cx2=net_cx2, networkx_graph=networkx_graph, keyprefix=keyprefix)

    add_eccentricity_attribute(net_cx2=net_cx2,
                               networkx_graph=networkx_graph,
                               keyprefix=keyprefix) # includes diameter and radius metrics
    
    if networkx_graph.is_directed():
        add_characteristic_path_length_net_attrib_directed(net_cx2=net_cx2,
                               networkx_graph=networkx_graph,
                               keyprefix=keyprefix)
    else:
        add_characteristic_path_length_net_attrib(net_cx2=net_cx2,
                               networkx_graph=networkx_graph,
                               keyprefix=keyprefix)
    
    add_multigraph_unsupported_metrics(net_cx2=net_cx2,
                                       networkx_graph=networkx_graph,
                                       keyprefix=keyprefix)
    
    add_density_net_attrib(net_cx2=net_cx2,
                                       networkx_graph=networkx_graph,
                                       keyprefix=keyprefix)

    add_heterogeneity_net_attrib(net_cx2=net_cx2, networkx_graph=networkx_graph,
                                 keyprefix=keyprefix)
    
    add_centralization_net_attrib(net_cx2=net_cx2, networkx_graph=networkx_graph,
                                            keyprefix=keyprefix)

    add_connected_components_net_attrib(net_cx2=net_cx2, networkx_graph=networkx_graph,
                                            keyprefix=keyprefix)

    ### Calls for Node-level metrics ###
    #add_cytoscape_average_shortest_path_lenght(net_cx2=net_cx2, networkx_graph=networkx_graph, keyprefix=keyprefix)
    #add_closeness_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph, keyprefix=keyprefix)
    #add_multiedge_partner_node_attribute(net_cx2=net_cx2, keyprefix=keyprefix)
    #add_self_loops_node_attribute(net_cx2=net_cx2, keyprefix=keyprefix)
    
    #add_cytoscape_stress_node_attribute_correct(net_cx2=net_cx2, networkx_graph=networkx_graph, keyprefix=keyprefix) #exclude endpoints
    
    add_degree_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph, keyprefix=keyprefix)
    
    #add_degree_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph, keyprefix=keyprefix)
    #add_betweenness_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph, keyprefix=keyprefix)
    #add_neighborhood_connectivity_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph, keyprefix=keyprefix)
    #add_cytoscape_radiality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph, keyprefix=keyprefix)
    #add_topological_coefficient_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph, keyprefix=keyprefix)
    #add_cytoscape_topological_coefficient_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph, keyprefix=keyprefix)

    

    ### Calls for Edge-level metrics ###
    if len(net_cx2.get_edges()) > 0:
        add_edge_betweenness_centrality(net_cx2=net_cx2, networkx_graph=networkx_graph,
                                       keyprefix=keyprefix)

    return net_cx2.to_cx2()


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
                                     isdirected=theargs.isdirected,
                                     namespace=theargs.namespace)

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
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 2


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv))
#!/usr/bin/env python

import os
import sys
import argparse
import json
import networkx as nx
from ndex2.cx2 import RawCX2NetworkFactory, CX2NetworkXFactory, CX2Network
from ndex2 import constants as ndex2constants



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
    net_cx2.add_network_attribute(key='average_degree',
                                  value=str(sum(dict(networkx_degree).values()) / networkx_graph.number_of_nodes()))
    net_cx2.add_network_attribute(key='density', value=str(nx.density(networkx_graph)))

    net_cx2.add_network_attribute(key='clustering_coefficient', value=str(nx.average_clustering(networkx_graph)))

    add_degree_centrality_node_attribute(net_cx2=net_cx2, networkx_graph=networkx_graph)
    add_degree_node_attribute(net_cx2=net_cx2, networkx_degree=networkx_graph.degree())

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
def add_edge_betweeness_centrality(net_cx2=None, networkx_graph=None,
                                   src_target_map=None):
    """

    """
    edge_betweenness = nx.edge_betweenness_centrality(networkx_graph)
    for nxedge_id, val in edge_betweenness.items():
        sys.stderr.write(str(src_target_map) + '\n\n')
        sys.stderr.write('edge id: ' + str(nxedge_id) + ' => ' + str(val) + '\n')
        net_cx2.add_edge_attribute(edge_id=src_target_map[nxedge_id], key='betweenness_centrality',
                                   value=val, datatype=ndex2constants.DOUBLE_DATATYPE)

def add_degree_centrality_node_attribute(net_cx2=None, networkx_graph=None):
    """
    Adds node degree_centraility node attribute
    """
    node_centrality = nx.degree_centrality(networkx_graph)
    for node_id, val in node_centrality.items():
        # sys.stderr.write('nodeid: ' + str(node_id) + ' => ' + str(val) + '\n')
        net_cx2.add_node_attribute(node_id=int(node_id), key='degree_centrality',
                                   value=val,
                                   datatype=ndex2constants.DOUBLE_DATATYPE)

def add_degree_node_attribute(net_cx2=None, networkx_degree=None):
    """
    Adds node degree_centraility node attribute
    """
    for node_degree in networkx_degree:
        # sys.stderr.write('nodeid: ' + str(node_id) + ' => ' + str(val) + '\n')
        net_cx2.add_node_attribute(node_id=int(node_degree[0]), key='degree',
                                   value=node_degree[1],
                                   datatype=ndex2constants.INTEGER_DATATYPE)

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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_networkxcywebanalyzercmd
----------------------------------

Tests for `networkxcywebanalyzercmd` module.
"""

import os
import sys
import unittest
import tempfile
import shutil
import networkx as nx
from ndex2.cx2 import CX2Network, CX2NetworkXFactory

from networkxcywebanalyzer import networkxcywebanalyzercmd



class TestNetworkxCyWebAnalyzer(unittest.TestCase):

    def get_sample1_cx2network_path(self):
        return os.path.join(os.path.dirname(__file__), 'sample1.cx2')

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_parse_args(self):
        myargs = ['inputarg']
        res = networkxcywebanalyzercmd._parse_arguments('desc', myargs)
        self.assertEqual('inputarg', res.input)
        self.assertEqual('analyze', res.mode)
        self.assertFalse(res.outputonlycx2)

    def test_analyze_network_success_with_sample1(self):
        sample1cx2file = self.get_sample1_cx2network_path()
        net_cx2 = networkxcywebanalyzercmd.get_cx2_net_from_input(sample1cx2file)
        res_cx2 = networkxcywebanalyzercmd.analyze_network(net_cx2=net_cx2)


    def test_add_edge_betweenness_centrality_two_node_net_multigraph(self):
        cx2_network = CX2Network()
        node_one_id = cx2_network.add_node(attributes={'name': 'node 1'})
        node_two_id = cx2_network.add_node(attributes={'name': 'node 2'})
        edge_id = cx2_network.add_edge(source=node_one_id,
                                       target=node_two_id,
                                       attributes={'interaction': 'foo'})
        networkx_graph = networkxcywebanalyzercmd.get_networkx_graph_with_keys(net_cx2=cx2_network, isdirected=False)
        networkxcywebanalyzercmd.add_edge_betweenness_centrality(net_cx2=cx2_network,
                                                                networkx_graph=networkx_graph)
        self.assertAlmostEqual(1.0,
                               cx2_network.get_edges()[0]['v'][' Edge Betweenness'])

    def test_add_edge_betweenness_centrality_two_node_net_multidigraph(self):
        cx2_network = CX2Network()
        node_one_id = cx2_network.add_node(attributes={'name': 'node 1'})
        node_two_id = cx2_network.add_node(attributes={'name': 'node 2'})
        edge_id = cx2_network.add_edge(source=node_one_id,
                                       target=node_two_id,
                                       attributes={'interaction': 'foo'})
        networkx_graph = networkxcywebanalyzercmd.get_networkx_graph_with_keys(net_cx2=cx2_network, isdirected=True)
        networkxcywebanalyzercmd.add_edge_betweenness_centrality(net_cx2=cx2_network,
                                                                networkx_graph=networkx_graph)
        self.assertAlmostEqual(0.5,
                               cx2_network.get_edges()[0]['v'][' Edge Betweenness'])

"""
    def test_run_infomap_no_file(self):
        temp_dir = tempfile.mkdtemp()
        try:
            tfile = os.path.join(temp_dir, 'foo')
            myargs = [tfile]
            theargs = cdinfomapcmd._parse_arguments('desc', myargs)
            res = cdinfomapcmd.run_infomap(tfile, theargs)
            self.assertEqual(3, res)
        finally:
            shutil.rmtree(temp_dir)

    def test_run_infomap_empty_file(self):
        temp_dir = tempfile.mkdtemp()
        try:
            tfile = os.path.join(temp_dir, 'foo')
            open(tfile, 'a').close()
            myargs = [tfile]
            theargs = cdinfomapcmd._parse_arguments('desc', myargs)
            res = cdinfomapcmd.run_infomap(tfile, theargs)
            self.assertEqual(4, res)
        finally:
            shutil.rmtree(temp_dir)

    def test_main_invalid_file(self):
        temp_dir = tempfile.mkdtemp()
        try:
            tfile = os.path.join(temp_dir, 'foo')
            myargs = ['prog', tfile]
            res = cdinfomapcmd.main(myargs)
            self.assertEqual(3, res)
        finally:
            shutil.rmtree(temp_dir)

    def test_main_empty_file(self):
        temp_dir = tempfile.mkdtemp()
        try:
            tfile = os.path.join(temp_dir, 'foo')
            open(tfile, 'a').close()
            myargs = ['prog', tfile]
            res = cdinfomapcmd.main(myargs)
            self.assertEqual(4, res)
        finally:
            shutil.rmtree(temp_dir)

    def test_check_if_file_contains_zero(self):
        temp_dir = tempfile.mkdtemp()
        try:
            # first value is 0
            edgelistfile = os.path.join(temp_dir, 'edgelist.txt')
            with open(edgelistfile, 'w') as f:
                f.write('0\t1\n')

            res = cdinfomapcmd.check_if_file_contains_zero(edgelistfile)
            self.assertTrue(res)

            # second value is 0
            with open(edgelistfile, 'w') as f:
                f.write('1\t0\n')

            res = cdinfomapcmd.check_if_file_contains_zero(edgelistfile)
            self.assertTrue(res)

            # no zero
            with open(edgelistfile, 'w') as f:
                f.write('1\t1\n')
                f.write('3\t10\n')

            res = cdinfomapcmd.check_if_file_contains_zero(edgelistfile)
            self.assertFalse(res)
        finally:
            shutil.rmtree(temp_dir)

    def test_get_truncated_file(self):
        self.assertEqual('yo', cdinfomapcmd.get_truncated_file('yo'))

        testpath = '/foo/hi.there'
        self.assertEqual('hi', cdinfomapcmd.get_truncated_file(testpath))

        testpath = '/foo/hi.'
        self.assertEqual('hi', cdinfomapcmd.get_truncated_file(testpath))

        testpath = '/foo/blah/bye'
        self.assertEqual('bye', cdinfomapcmd.get_truncated_file(testpath))
"""

if __name__ == '__main__':
    sys.exit(unittest.main())

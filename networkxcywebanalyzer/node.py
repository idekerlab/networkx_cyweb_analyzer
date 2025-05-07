import networkx as nx
from ndex2 import constants as ndex2constants
from itertools import combinations
from collections import defaultdict


def add_eccentricity_attribute(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Computes and adds node eccentricity attributes, Radius, and Diameter.
    For directed graphs: 
      - WCC metrics are computed on undirected versions (since WCC is inherently undirected)
      - SCC metrics are computed on directed versions (if strongly connected)
    Handles single-node components explicitly.
    """
    def _add_metrics(eccentricities, suffix=''):
        """Helper to add metrics to network, handling empty cases."""
        if not eccentricities:
            return
            
        for node_id, val in eccentricities.items():
            net_cx2.add_node_attribute(
                node_id=int(node_id),
                key=f"{keyprefix} Eccentricity {suffix}",
                value=val,
                datatype=ndex2constants.INTEGER_DATATYPE
            )
        
        net_cx2.add_network_attribute(
            key=f"Network diameter {suffix}",
            value=str(max(eccentricities.values())),
            datatype=ndex2constants.STRING_DATATYPE
        )
        net_cx2.add_network_attribute(
            key=f"Network radius {suffix}",
            value=str(min(eccentricities.values())),
            datatype=ndex2constants.STRING_DATATYPE
        )

    def _compute_single_node_metrics(node_id, suffix):
        """Helper for single-node component case."""
        net_cx2.add_node_attribute(
            node_id=int(node_id),
            key=f"{keyprefix} Eccentricity {suffix}",
            value=0,
            datatype=ndex2constants.INTEGER_DATATYPE
        )
        net_cx2.add_network_attribute(
            key=f"Network diameter {suffix}",
            value='0',
            datatype=ndex2constants.STRING_DATATYPE
        )
        net_cx2.add_network_attribute(
            key=f"Network radius {suffix}",
            value='0',
            datatype=ndex2constants.STRING_DATATYPE
        )

    # Input validation
    if not isinstance(networkx_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("Input must be a NetworkX graph")

    is_directed = isinstance(networkx_graph, (nx.DiGraph, nx.MultiDiGraph))

    if is_directed:
        # --- Weakly Connected Components (WCC) ---
        wccs = list(nx.weakly_connected_components(networkx_graph))
        largest_wcc = max(wccs, key=len) if wccs else set()
        wcc_subgraph = networkx_graph.subgraph(largest_wcc)
        suffix = '(WCC)' if len(wccs) > 1 else ''

        if len(wcc_subgraph) == 1:  # Single-node WCC
            _compute_single_node_metrics(list(wcc_subgraph.nodes())[0], suffix)
        else:
            # WCC is undirected by definition
            wcc_undirected = wcc_subgraph.to_undirected()
            _add_metrics(nx.eccentricity(wcc_undirected), suffix)

        # --- Strongly Connected Components (SCC) ---
        sccs = list(nx.strongly_connected_components(networkx_graph))
        largest_scc = max(sccs, key=len) if sccs else set()
        scc_subgraph = networkx_graph.subgraph(largest_scc)
        suffix = '(SCC)' if len(sccs) > 1 else ''

        if len(scc_subgraph) == 1:  # Single-node SCC
            _compute_single_node_metrics(list(scc_subgraph.nodes())[0], suffix)
        else:
            try:
                _add_metrics(nx.eccentricity(scc_subgraph), suffix)
            except nx.NetworkXError:
                pass  # SCC is not strongly connected (can't compute eccentricity)

    else:
        # --- Undirected Graph Case ---
        components = list(nx.connected_components(networkx_graph))
        largest_component = max(components, key=len) if components else set()
        subgraph = networkx_graph.subgraph(largest_component)
        suffix = '(LCC)' if len(components) > 1 else ''

        if len(subgraph) == 1:  # Single-node component
            _compute_single_node_metrics(list(subgraph.nodes())[0], suffix)
        else:
            _add_metrics(nx.eccentricity(subgraph), suffix)

def add_degree_node_attribute(net_cx2, networkx_graph, keyprefix=''):

    """
    Adds node‐degree attributes to a CX2 network.

    - If `networkx_graph` is undirected, adds a single "<prefix>Degree".
    - If directed, adds both "<prefix>InDegree" and "<prefix>OutDegree".

    Args:
        net_cx2 (ndex2.cx2.CX2Network): The CX2 network to modify.
        networkx_graph (nx.Graph or nx.DiGraph): The NetworkX graph.
        keyprefix (str): Optional prefix for the attribute keys.
    """
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")

    if networkx_graph.is_directed():
        # directed: write in- and out-degrees
        for node, indeg in networkx_graph.in_degree():
            net_cx2.add_node_attribute(
                node_id=int(node),
                key=f"{keyprefix} In-Degree",
                value=int(indeg),
                datatype=ndex2constants.INTEGER_DATATYPE
            )
        for node, outdeg in networkx_graph.out_degree():
            net_cx2.add_node_attribute(
                node_id=int(node),
                key=f"{keyprefix} Out-Degree",
                value=int(outdeg),
                datatype=ndex2constants.INTEGER_DATATYPE
            )
    else:
        # undirected: just write total degree
        for node, deg in networkx_graph.degree():
            net_cx2.add_node_attribute(
                node_id=int(node),
                key=f"{keyprefix} Degree",
                value=int(deg),
                datatype=ndex2constants.INTEGER_DATATYPE
            )


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
            key=f"Network Diameter {suffix}",
            value=str(max(eccentricities.values())),
            datatype=ndex2constants.STRING_DATATYPE
        )
        net_cx2.add_network_attribute(
            key=f"Network Radius {suffix}",
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
            key=f"Network Diameter {suffix}",
            value='0',
            datatype=ndex2constants.STRING_DATATYPE
        )
        net_cx2.add_network_attribute(
            key=f"Network Radius {suffix}",
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

def add_degree_node_attribute(net_cx2=None, networkx_graph=None, keyprefix=''):

    """
    Adds node degree attributes to a CX2 network.

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

def add_comprehensive_topological_metrics(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Compute and add a suite of topological coefficient metrics to a CX2 network,
    covering:
      • Undirected graphs (global + LCC-only if disconnected)
      • Directed graphs 
        - If strongly connected: Successors, Predecessors, Mutual (on full graph)
        - If disconnected: Global (undirected) + Successors, Predecessors, Mutual (on largest SCC)
    """
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")

    def _topological_coefficient(neighbors_dict):
        """Compute Cytoscape style topological coefficient for each node."""
        tc = {}
        for node, neighbors in neighbors_dict.items():
            if len(neighbors) < 2:
                tc[node] = 0.0
                continue
            total = 0
            for u, v in combinations(neighbors, 2):
                shared = neighbors_dict.get(u, set()) & neighbors_dict.get(v, set())
                total += len(shared)
            denom = len(neighbors) * (len(neighbors) - 1) / 2
            tc[node] = total / denom if denom > 0 else 0.0
        return tc

    def _add_attributes(metric_dict, key):
        """Add each metric in metric_dict as a node attribute named key (for all nodes)."""
        for node_id in net_cx2.get_nodes():
            nid = int(node_id)
            net_cx2.add_node_attribute(
                node_id=nid,
                key=f"{keyprefix} {key}",
                value=float(metric_dict.get(nid, 0.0)),
                datatype=ndex2constants.DOUBLE_DATATYPE
            )

    G = networkx_graph

    # ── UNDIRECTED GRAPH ─────────────────────────────────────────────
    if not G.is_directed():
        # 1) Global Topological Coefficient
        nbrs_all = {n: set(G.neighbors(n)) - {n} for n in G.nodes()}
        _add_attributes(_topological_coefficient(nbrs_all),
                        'Topological Coeff.')

        # 2) LCC‐only if disconnected: only for LCC nodes
        if not nx.is_connected(G):
            lcc_nodes = max(nx.connected_components(G), key=len)
            subG = G.subgraph(lcc_nodes)
            nbrs_lcc = {n: set(subG.neighbors(n)) - {n} for n in subG.nodes()}
            tc_lcc = _topological_coefficient(nbrs_lcc)
            for node, value in tc_lcc.items():
                net_cx2.add_node_attribute(
                    node_id=int(node),
                    key='Topological Coeff. (LCC)',
                    value=float(value),
                    datatype=ndex2constants.DOUBLE_DATATYPE
                )

    # ── DIRECTED GRAPH ───────────────────────────────────────────────
    else:
        # Determine SCC and whether to include global TC
        if nx.is_strongly_connected(G):
            scc_nodes = set(G.nodes())
            include_global = False
        else:
            scc_nodes = max(nx.strongly_connected_components(G), key=len)
            include_global = True

        # 1) Global (undirected) TC if disconnected
        if include_global:
            und_G = G.to_undirected()
            nbrs_undir = {n: set(und_G.neighbors(n)) - {n} for n in G.nodes()}
            _add_attributes(_topological_coefficient(nbrs_undir),
                            'Topological Coeff.')

        # 2) Directed-specific metrics on SCC only
        subG_scc = G.subgraph(scc_nodes)
        suffix = ', SCC' if include_global else ''

        # 2a) Successors‐based TC
        nbrs_succ = {n: set(subG_scc.successors(n)) - {n} for n in subG_scc.nodes()}
        tc_succ = _topological_coefficient(nbrs_succ)
        for node, value in tc_succ.items():
            net_cx2.add_node_attribute(
                node_id=int(node),
                key=f'Topological Coeff. (Successors{suffix})',
                value=float(value),
                datatype=ndex2constants.DOUBLE_DATATYPE
            )

        # 2b) Predecessors‐based TC
        nbrs_pred = {n: set(subG_scc.predecessors(n)) - {n} for n in subG_scc.nodes()}
        tc_pred = _topological_coefficient(nbrs_pred)
        for node, value in tc_pred.items():
            net_cx2.add_node_attribute(
                node_id=int(node),
                key=f'Topological Coeff. (Predecessors{suffix})',
                value=float(value),
                datatype=ndex2constants.DOUBLE_DATATYPE
            )

        # 2c) Mutual‐neighbors TC
        nbrs_mutual = {
            n: (set(subG_scc.successors(n)) & set(subG_scc.predecessors(n))) - {n}
            for n in subG_scc.nodes()
        }
        tc_mutual = _topological_coefficient(nbrs_mutual)
        for node, value in tc_mutual.items():
            net_cx2.add_node_attribute(
                node_id=int(node),
                key=f'Topological Coeff. (Mutual{suffix})',
                value=float(value),
                datatype=ndex2constants.DOUBLE_DATATYPE
            )
        

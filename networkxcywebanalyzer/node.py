import networkx as nx
from ndex2 import constants as ndex2constants
from itertools import combinations
from collections import defaultdict


def add_eccentricity_attribute(net_cx2=None, networkx_graph=None, keyprefix=''): #incl. diameter and radius 
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
                key=f"Eccentricity {suffix}",
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
            key=f"Eccentricity {suffix}",
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
                key=f"In-Degree",
                value=int(indeg),
                datatype=ndex2constants.INTEGER_DATATYPE
            )
        for node, outdeg in networkx_graph.out_degree():
            net_cx2.add_node_attribute(
                node_id=int(node),
                key=f"Out-Degree",
                value=int(outdeg),
                datatype=ndex2constants.INTEGER_DATATYPE
            )
    else:
        # undirected: just write total degree
        for node, deg in networkx_graph.degree():
            net_cx2.add_node_attribute(
                node_id=int(node),
                key=f"Degree",
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
                key=f"{key}",
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
        
def add_average_shortest_path_lenght(net_cx2=None, networkx_graph=None, keyprefix=''): # incl. node radiality
    """
    Computes the node-level 'AverageShortestPathLength' and 'Radiality' metrics.

    - Undirected:
      • If connected → on entire graph (no suffix)
      • Else         → on largest connected component (suffix " (LCC)")

    - Directed:
      • If strongly connected → on entire graph (no suffix)
      • Else                  → on
         1) largest strongly connected component (suffix " (SCC)")
         2) largest weakly   connected component (suffix " (WCC)")

    Nodes outside these component(s) are left untouched.
    """
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")

    directed = networkx_graph.is_directed()
    components = []

    # 1) Determine which subgraph(s) to process
    if directed:
        if nx.is_strongly_connected(networkx_graph):
            components.append((networkx_graph, ''))
        else:
            # largest SCC
            scc_nodes = max(nx.strongly_connected_components(networkx_graph), key=len)
            components.append((networkx_graph.subgraph(scc_nodes), ' (SCC)'))
            # largest WCC
            wcc_nodes = max(nx.weakly_connected_components(networkx_graph), key=len)
            components.append((networkx_graph.subgraph(wcc_nodes), ' (WCC)'))
    else:
        if nx.is_connected(networkx_graph):
            components.append((networkx_graph, ''))
        else:
            # largest connected component
            lcc_nodes = max(nx.connected_components(networkx_graph), key=len)
            components.append((networkx_graph.subgraph(lcc_nodes), ' (LCC)'))

    # 2) For each chosen subgraph, compute and write metrics
    for subg, suffix in components:
        n = subg.number_of_nodes()
        if n < 2:
            # too few nodes for any paths → skip
            continue

        # compute diameter for radiality
        try:
            diameter = nx.diameter(subg)
        except nx.NetworkXError:
            diameter = 0

        # prepare attribute keys
        asp_key = f"Average Shortest Path Length{suffix}"
        rad_key = f"Radiality{suffix}"

        # compute and write per-node
        for node in subg.nodes():
            # average shortest path length
            spl = nx.shortest_path_length(subg, source=node)
            avg_spl = sum(spl.values()) / (n - 1)
            net_cx2.add_node_attribute(
                node_id=int(node),
                key=asp_key,
                value=float(avg_spl),
                datatype=ndex2constants.DOUBLE_DATATYPE
            )

            # radiality
            if diameter > 0:
                rad = (diameter + 1 - avg_spl) / diameter
            else:
                rad = 0.0
            net_cx2.add_node_attribute(
                node_id=int(node),
                key=rad_key,
                value=float(rad),
                datatype=ndex2constants.DOUBLE_DATATYPE
            )

def add_stress_node_attribute(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Calculates node Stress Centrality matching Cytoscape's implementation (raw counts).

    - Undirected:
      • If connected → on entire graph (no suffix, treat as undirected)
      • Else         → on largest connected component (suffix " (LCC)", treat as undirected)

    - Directed:
      • If strongly connected → on entire graph (no suffix, treat as directed)
      • Else                  → on
         1) largest strongly connected component (suffix " (SCC)", treat as directed)
         2) largest weakly   connected component (suffix " (WCC)", treat as undirected)

    Nodes outside these component(s) are left untouched.
    """
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")

    components = []

    # 1) Determine which subgraph(s) to process, and whether to treat each as directed
    if networkx_graph.is_directed():
        if nx.is_strongly_connected(networkx_graph):
            components.append((networkx_graph, '', True))
        else:
            # largest strongly connected component (directed)
            scc_nodes = max(nx.strongly_connected_components(networkx_graph), key=len)
            scc_subg = networkx_graph.subgraph(scc_nodes)
            components.append((scc_subg, ' (SCC)', True))
            # largest weakly connected component (treat as undirected)
            wcc_nodes = max(nx.weakly_connected_components(networkx_graph), key=len)
            wcc_subg = networkx_graph.subgraph(wcc_nodes).to_undirected()
            components.append((wcc_subg, ' (WCC)', False))
    else:
        # undirected graph
        if nx.is_connected(networkx_graph):
            components.append((networkx_graph, '', False))
        else:
            lcc_nodes = max(nx.connected_components(networkx_graph), key=len)
            lcc_subg = networkx_graph.subgraph(lcc_nodes)
            components.append((lcc_subg, ' (LCC)', False))

    # 2) Compute and write stress for each selected component
    for subg, suffix, comp_directed in components:
        stress = defaultdict(int)
        nodes = list(subg.nodes())
        if len(nodes) < 2:
            continue

        if comp_directed:
            # directed: consider all ordered source→target pairs
            for source in nodes:
                for target in nodes:
                    if source == target:
                        continue
                    for path in nx.all_shortest_paths(subg, source, target):
                        for n in path[1:-1]:  # exclude endpoints
                            stress[n] += 1
        else:
            # undirected (or WCC treated as undirected): each unordered pair once
            for i, source in enumerate(nodes):
                for target in nodes[i+1:]:
                    for path in nx.all_shortest_paths(subg, source, target):
                        for n in path[1:-1]:
                            stress[n] += 1

        # 3) Write back only for nodes in this subgraph
        attr_name = f"Stress{suffix}"
        for n in subg.nodes():
            net_cx2.add_node_attribute(
                node_id=int(n),
                key=attr_name,
                value=int(stress.get(n, 0)),
                datatype=ndex2constants.INTEGER_DATATYPE
            )

def add_centrality_node_attributes(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Adds Betweenness, Degree, and Closeness centrality node attributes.
    
    - Undirected:
      • If connected → metrics on entire graph (no suffix)
      • Else          → metrics on largest connected component (suffix " (LCC)")
    
    - Directed:
      • If strongly connected → metrics on entire graph (no suffix)
      • Else                  → metrics on
         1) largest strongly connected component (suffix " (SCC)")
         2) largest weakly   connected component (suffix " (WCC)")
    
    Nodes outside these component(s) are left untouched.
    """
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")

    directed = networkx_graph.is_directed()
    components = []

    if directed:
        # fully strongly connected?
        if nx.is_strongly_connected(networkx_graph):
            components.append((networkx_graph, ''))
        else:
            # largest SCC
            scc_nodes = max(nx.strongly_connected_components(networkx_graph), key=len)
            components.append((networkx_graph.subgraph(scc_nodes), ' (SCC)'))
            # largest WCC
            wcc_nodes = max(nx.weakly_connected_components(networkx_graph), key=len)
            components.append((networkx_graph.subgraph(wcc_nodes), ' (WCC)'))
    else:
        # undirected
        if nx.is_connected(networkx_graph):
            components.append((networkx_graph, ''))
        else:
            lcc_nodes = max(nx.connected_components(networkx_graph), key=len)
            components.append((networkx_graph.subgraph(lcc_nodes), ' (LCC)'))

    for subg, suffix in components:
        # compute all three centralities on this subgraph
        bc = nx.betweenness_centrality(subg)
        dc = nx.degree_centrality(subg)
        cc = nx.closeness_centrality(subg)

        # prepare attribute names
        bc_key = f"Betweenness Centr.{suffix}"
        dc_key = f"Degree Centr.{suffix}"
        cc_key = f"Closeness Centr.{suffix}"

        # write only for nodes in this subgraph
        for n in subg.nodes():
            net_cx2.add_node_attribute(
                node_id=int(n),
                key=bc_key,
                value=float(bc.get(n, 0.0)),
                datatype=ndex2constants.DOUBLE_DATATYPE
            )
            net_cx2.add_node_attribute(
                node_id=int(n),
                key=dc_key,
                value=float(dc.get(n, 0.0)),
                datatype=ndex2constants.DOUBLE_DATATYPE
            )
            net_cx2.add_node_attribute(
                node_id=int(n),
                key=cc_key,
                value=float(cc.get(n, 0.0)),
                datatype=ndex2constants.DOUBLE_DATATYPE
            )

def add_neighborhood_connectivity_node_attribute(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Calculates and adds Neighborhood Connectivity as a node attribute to a CX2 network.

    - Undirected:
      • If connected → on entire graph (no suffix)
      • Else         → on largest connected component (suffix " (LCC)")

    - Directed:
      • If strongly connected → on entire graph (no suffix)
      • Else                  → on
         1) largest strongly connected component (suffix " (SCC)")
         2) largest weakly   connected component (suffix " (WCC)")

    Neighborhood connectivity is the average degree of each node’s neighbors.
    Nodes outside these component(s) are left untouched.
    """
    # Input validation (as in the original)
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both network objects must be provided")
    if len(networkx_graph) == 0 or len(net_cx2.get_nodes()) == 0:
        raise ValueError("Networks cannot be empty")
    if set(net_cx2.get_nodes()) != set(networkx_graph.nodes()):
        raise ValueError("Node sets between CX2 and NetworkX graphs must match")

    directed = networkx_graph.is_directed()
    components = []

    # 1) Determine which subgraph(s) to process
    if directed:
        # Strongly connected?
        if nx.is_strongly_connected(networkx_graph):
            components.append((networkx_graph, ''))
        else:
            # Largest SCC (treat as directed)
            scc_nodes = max(nx.strongly_connected_components(networkx_graph), key=len)
            components.append((networkx_graph.subgraph(scc_nodes), ' (SCC)'))
            # Largest WCC (treat as undirected)
            wcc_nodes = max(nx.weakly_connected_components(networkx_graph), key=len)
            wcc_subg = networkx_graph.subgraph(wcc_nodes).to_undirected()
            components.append((wcc_subg, ' (WCC)'))
    else:
        # Undirected graph
        if nx.is_connected(networkx_graph):
            components.append((networkx_graph, ''))
        else:
            # Largest connected component
            lcc_nodes = max(nx.connected_components(networkx_graph), key=len)
            components.append((networkx_graph.subgraph(lcc_nodes), ' (LCC)'))

    # 2) Compute and write Neighborhood Connectivity for each component
    for subg, suffix in components:
        # Compute average neighbor-degree for nodes in subg
        nc = {}
        for node in subg.nodes():
            neighbors = list(subg.neighbors(node))
            if not neighbors:
                nc[node] = 0.0
            else:
                total_deg = sum(subg.degree(nbr) for nbr in neighbors)
                nc[node] = total_deg / len(neighbors)

        # Write back only for nodes in this subgraph
        attr_name = f"Neighborhood Conn.{suffix}"
        for node_id, value in nc.items():
            net_cx2.add_node_attribute(
                node_id=int(node_id),
                key=attr_name,
                value=float(value),
                datatype=ndex2constants.DOUBLE_DATATYPE
            )

def add_self_loops_node_attribute(net_cx2=None, keyprefix=''):
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
            key='Self Loops',
            value=self_loop_counts.get(node_id, 0),
            datatype=ndex2constants.INTEGER_DATATYPE
        )

def add_multiedge_partner_node_attribute(net_cx2=None, keyprefix=''):
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
            
            u, v = int(edge_data['s']), int(edge_data['t']) # Preserve directionality info
            edge_counts[(u, v)] += 1
            
            #u, v = sorted((int(u), int(v)))  # Ensure consistent ordering but looses directionality info
            #edge_counts[(u, v)] += 1
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


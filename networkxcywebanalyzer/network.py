import networkx as nx
from ndex2 import constants as ndex2constants
import numpy as np


def add_number_of_nodes(net_cx2=None, keyprefix=''):

    number_of_nodes = len(net_cx2.get_nodes())

    # Add to CX2 network
    net_cx2.add_network_attribute(
        key=keyprefix + 'Number of Nodes',
        value=str(number_of_nodes),
        datatype=ndex2constants.STRING_DATATYPE
    )

def add_number_of_edges(net_cx2=None, keyprefix=''):

    number_of_edges = len(net_cx2.get_edges())

    # Add to CX2 network
    net_cx2.add_network_attribute(
        key=keyprefix + 'Number of Edges',
        value=str(number_of_edges),
        datatype=ndex2constants.STRING_DATATYPE
    )

def add_avg_neighbors_net_attrib(net_cx2=None, networkx_graph=None, keyprefix=''):
    """Calculates neighbor metrics and component coverage for directed/undirected networks.
    Avoids redundant WCC/SCC outputs for fully connected directed graphs."""
    
    if not isinstance(networkx_graph, (nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph)):
        raise ValueError("Requires NetworkX Graph/MultiGraph or DiGraph/MultiDiGraph")
    
    is_directed = isinstance(networkx_graph, (nx.DiGraph, nx.MultiDiGraph))
    nodes = list(networkx_graph.nodes())
    
    # Handle neighbor calculations
    if is_directed:
        if nx.is_strongly_connected(networkx_graph):
            # Only report SCC metrics (WCC = SCC in fully connected graphs)
            scc_avg = sum(len(set(networkx_graph.successors(n))) for n in nodes) / len(nodes)
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Avg. unique neighbors (SCC)",
                value=str(round(scc_avg, 3)),
                datatype=ndex2constants.STRING_DATATYPE
            )
        else:
            # For disconnected directed graphs, report both WCC and SCC
            wcc_avg = sum(len(set(networkx_graph.neighbors(n))) for n in nodes) / len(nodes)
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Avg. unique neighbors (WCC)",
                value=str(round(wcc_avg, 3)),
                datatype=ndex2constants.STRING_DATATYPE
            )
            scc_avg = sum(len(set(networkx_graph.successors(n))) for n in nodes) / len(nodes)
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Avg. unique neighbors (SCC)",
                value=str(round(scc_avg, 3)),
                datatype=ndex2constants.STRING_DATATYPE
            )
    else:
        # Undirected graphs: simple average
        global_avg = sum(len(set(networkx_graph.neighbors(n))) for n in nodes) / len(nodes)
        net_cx2.add_network_attribute(
            key=f"{keyprefix}Avg. unique neighbors",
            value=str(round(global_avg, 3)),
            datatype=ndex2constants.STRING_DATATYPE
        )
    
    # Handle component metrics (unchanged)
    if is_directed:
        wccs = list(nx.weakly_connected_components(networkx_graph))
        sccs = list(nx.strongly_connected_components(networkx_graph))
        
        if len(wccs) > 1:
            largest_wcc = max(wccs, key=len)
            coverage = len(largest_wcc)/len(nodes)*100
            net_cx2.add_network_attribute(
                key=f"{keyprefix}WCC nodes",
                value=str(len(largest_wcc)),
                datatype=ndex2constants.STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}WCC node coverage (%)",
                value=str(round(coverage, 1)),
                datatype=ndex2constants.STRING_DATATYPE
            )
        
        if len(sccs) > 1:
            largest_scc = max(sccs, key=len)
            coverage = len(largest_scc)/len(nodes)*100
            net_cx2.add_network_attribute(
                key=f"{keyprefix}SCC nodes",
                value=str(len(largest_scc)),
                datatype=ndex2constants.STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}SCC node coverage (%)",
                value=str(round(coverage, 1)),
                datatype=ndex2constants.STRING_DATATYPE
            )
    else:
        components = list(nx.connected_components(networkx_graph))
        if len(components) > 1:
            lcc = max(components, key=len)
            lcc_avg = sum(len(set(networkx_graph.neighbors(n))) for n in lcc) / len(lcc)
            coverage = len(lcc)/len(nodes)*100
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Avg. unique neighbors (LCC)",
                value=str(round(lcc_avg, 3)),
                datatype=ndex2constants.STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}LCC nodes",
                value=str(len(lcc)),
                datatype=ndex2constants.STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}LCC node coverage (%)",
                value=str(round(coverage, 1)),
                datatype=ndex2constants.STRING_DATATYPE
            )

def _calculate_path_length(graph, component_type=None):
    """Shared helper for CPL (Characteristic Path Length) calculations

    Args:
        graph: NetworkX graph object
        component_type: None (undirected), 'WCC', or 'SCC'

    Returns:
        tuple: (path_length, is_connected)
               path_length may be float or "undefined (single node)"
    """
    # 1) pick the right components list & extract the subgraph
    if component_type == 'WCC':
        components = list(nx.weakly_connected_components(graph))
        # largest weakly connected comp
        largest = max(components, key=len)
        # build the subgraph *and* collapse direction
        subgraph = graph.subgraph(largest).to_undirected()
    elif component_type == 'SCC':
        components = list(nx.strongly_connected_components(graph))
        largest = max(components, key=len)
        # keep it directed
        subgraph = graph.subgraph(largest)
    else:
        components = list(nx.connected_components(graph))
        largest = max(components, key=len)
        subgraph = graph.subgraph(largest)

    is_connected = (len(components) == 1)

    # 2) compute the average shortest path (directed for SCC, undirected otherwise)
    try:
        length = nx.average_shortest_path_length(subgraph)
    except nx.NetworkXError:
        # this only happens if the subgraph has one node (no paths)
        length = "undefined (single node)"

    return length, is_connected

def add_characteristic_path_length_net_attrib(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Computes characteristic path length for undirected networks.
    Uses largest connected component if graph is disconnected.
    """
    if not isinstance(networkx_graph, (nx.Graph, nx.MultiGraph)):
        raise ValueError("Requires undirected NetworkX Graph/MultiGraph")
    
    cpl, is_connected = _calculate_path_length(networkx_graph)
    
    key_suffix = 'Characteristic path length' if is_connected else 'Characteristic path length (LCC)'
    net_cx2.add_network_attribute(
        key=f"{keyprefix}{key_suffix}",
        value=str(cpl) if isinstance(cpl, str) else str(round(cpl, 3)),
        datatype=ndex2constants.STRING_DATATYPE
    )

def add_characteristic_path_length_net_attrib_directed(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Computes characteristic path length metrics for directed networks.
    Calculates both WCC (weakly connected) and SCC (strongly connected) metrics.
    """
    if not isinstance(networkx_graph, (nx.DiGraph, nx.MultiDiGraph)):
        raise ValueError("Requires directed NetworkX DiGraph/MultiDiGraph")
    
    # WCC Calculation
    wcc_cpl, is_wcc_connected = _calculate_path_length(networkx_graph, 'WCC')
    wcc_key = 'Characteristic path length' if is_wcc_connected else 'Characteristic path length (WCC)'
    net_cx2.add_network_attribute(
        key=f"{keyprefix}{wcc_key}",
        value=str(wcc_cpl) if isinstance(wcc_cpl, str) else str(round(wcc_cpl, 3)),
        datatype=ndex2constants.STRING_DATATYPE
    )
    
    # SCC Calculation
    scc_cpl, is_scc_connected = _calculate_path_length(networkx_graph, 'SCC')
    scc_key = 'Characteristic path length' if is_scc_connected else 'Characteristic path length (SCC)'
    net_cx2.add_network_attribute(
        key=f"{keyprefix}{scc_key}",
        value=str(scc_cpl) if isinstance(scc_cpl, str) else str(round(scc_cpl, 3)),
        datatype=ndex2constants.STRING_DATATYPE
    )

def add_multigraph_unsupported_metrics(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Computes metrics network-level metrics (Avg. clustering coeff. and Transitivity) as well 
    as node-level metrics (Clustering coefficient and Eigenvector centrality), while correctly handling:
    - Disconnected directed networks (uses largest SCC and WCC).
    - Disconnected undirected networks (uses LCC).
    """
    is_directed = nx.is_directed(networkx_graph)
    G_simple = nx.DiGraph(networkx_graph) if is_directed else nx.Graph(networkx_graph)
    G_w = nx.DiGraph() if is_directed else nx.Graph()

    # Sum multi-edges to compute weights
    for u, v in networkx_graph.edges():
        if G_w.has_edge(u, v):
            G_w[u][v]['weight'] += 1
        else:
            G_w.add_edge(u, v, weight=1)

    # --- Helper function (nested inside) ---
    def compute_and_add_metrics(net_cx2, G_simple, G_w, keyprefix, suffix=''):
        """Helper function to compute "multigraph unsupported metrics" and add them to net_cx2."""
        # Clustering coefficient (node-level)
        clustering_coeff = nx.clustering(G_simple)

        # Eigenvector centrality (now guaranteed to work since we use LCC/SCC/WCC)
        eigenvector = nx.eigenvector_centrality(G_w, weight='weight', max_iter=1000)

        # Global metrics (avg. clustering, transitivity)
        valid_nodes = [n for n in G_simple.nodes() if G_simple.degree(n) >= 2]
        avg_clustering_coeff = (
            sum(clustering_coeff[n] for n in valid_nodes) / len(valid_nodes)
        ) if valid_nodes else 0.0

        transitivity = nx.transitivity(G_simple)

        # Add network-level attributes
        net_cx2.add_network_attribute(
            key=f"{keyprefix} Avg. clustering Coefficient{suffix}",
            value=str(round(avg_clustering_coeff, 3)),
            datatype=ndex2constants.STRING_DATATYPE
        )
        net_cx2.add_network_attribute(
            key=f"{keyprefix} Transitivity{suffix}",
            value=str(round(transitivity, 3)),
            datatype=ndex2constants.STRING_DATATYPE
        )

        # Add node-level attributes
        for node_id, val in clustering_coeff.items():
            net_cx2.add_node_attribute(
                node_id=int(node_id),
                key=f"{keyprefix} Clustering Coefficient{suffix}",
                value=val,
                datatype=ndex2constants.DOUBLE_DATATYPE
            )

        for node_id, val in eigenvector.items():
            net_cx2.add_node_attribute(
                node_id=int(node_id),
                key=f"{keyprefix} Eigenvector Centrality{suffix}",
                value=val,
                datatype=ndex2constants.DOUBLE_DATATYPE
            )

    # --- Handle disconnected networks ---
    if is_directed:
        # Directed: Compute for both SCC and WCC if disconnected
        if not nx.is_strongly_connected(G_simple):
            largest_wcc = max(nx.weakly_connected_components(G_simple), key=len)
            G_wcc = G_simple.subgraph(largest_wcc).copy()
            G_w_wcc = G_w.subgraph(largest_wcc).copy()
            compute_and_add_metrics(net_cx2, G_wcc, G_w_wcc, keyprefix, suffix=' (WCC)')
            
            largest_scc = max(nx.strongly_connected_components(G_simple), key=len)
            G_scc = G_simple.subgraph(largest_scc).copy()
            G_w_scc = G_w.subgraph(largest_scc).copy()
            compute_and_add_metrics(net_cx2, G_scc, G_w_scc, keyprefix, suffix=' (SCC)')

            return
    else:
        # Undirected: Compute for LCC if disconnected
        if not nx.is_connected(G_simple):
            largest_cc = max(nx.connected_components(G_simple), key=len)
            G_lcc = G_simple.subgraph(largest_cc).copy()
            G_w_lcc = G_w.subgraph(largest_cc).copy()
            compute_and_add_metrics(net_cx2, G_lcc, G_w_lcc, keyprefix, suffix=' (LCC)')
            return

    # Default case (connected graph)
    compute_and_add_metrics(net_cx2, G_simple, G_w, keyprefix)

def add_density_net_attrib(net_cx2=None, networkx_graph=None,
                           keyprefix="", density_key="Network Density"):
    """
    Calculates network density, automatically accounting for multi‐edges if present.
    - For disconnected networks, only computes density for the largest component
      (LCC for undirected; WCC & SCC for directed).
    - Rounds to 3 decimal places and stores as STRING_DATATYPE.
    """
    if not isinstance(networkx_graph, (nx.Graph, nx.DiGraph,
                                       nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("Input must be a NetworkX Graph/DiGraph/MultiGraph/MultiDiGraph")
    
    def compute_density(G):
        n = G.number_of_nodes()
        if n < 2:
            return 0.0
        actual_edges = G.number_of_edges()  # includes all parallel edges/arcs
        max_edges = (n * (n - 1)) if G.is_directed() else (n * (n - 1) / 2)
        return actual_edges / max_edges if max_edges > 0 else 0.0

    directed = networkx_graph.is_directed()
    connected = (nx.is_strongly_connected(networkx_graph)
                 if directed else nx.is_connected(networkx_graph))

    if connected:
        d = round(compute_density(networkx_graph), 3)
        net_cx2.add_network_attribute(
            key=f"{keyprefix}{density_key}",
            value=str(d),
            datatype=ndex2constants.STRING_DATATYPE
        )
    else:
        if not directed:
            lcc = max(nx.connected_components(networkx_graph), key=len)
            d = round(compute_density(networkx_graph.subgraph(lcc)), 3)
            net_cx2.add_network_attribute(
                key=f"{keyprefix}{density_key} (LCC)",
                value=str(d),
                datatype=ndex2constants.STRING_DATATYPE
            )
        else:
            wcc = max(nx.weakly_connected_components(networkx_graph), key=len)
            scc = max(nx.strongly_connected_components(networkx_graph), key=len)
            d_w = round(compute_density(networkx_graph.subgraph(wcc)), 3)
            d_s = round(compute_density(networkx_graph.subgraph(scc)), 3)
            net_cx2.add_network_attribute(
                key=f"{keyprefix}{density_key} (WCC)",
                value=str(d_w),
                datatype=ndex2constants.STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}{density_key} (SCC)",
                value=str(d_s),
                datatype=ndex2constants.STRING_DATATYPE
            )

def add_centralization_net_attrib(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Calculates network centralization matching Cytoscape’s behavior.
    - For disconnected networks, only computes on the largest component.
    - Counts multi‐edges/arcs via G.degree().
    - Rounds to 3 decimal places and stores as STRING_DATATYPE.
    """
    if not isinstance(networkx_graph, (nx.Graph, nx.DiGraph,
                                       nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("Input must be a NetworkX Graph/DiGraph/MultiGraph/MultiDiGraph")
    
    def compute_centralization(G):
        N = G.number_of_nodes()
        if N < 2:
            return 0.0
        # total degree for each node (in+out if directed)
        degrees = [d for _, d in G.degree()]
        # normalize by max possible (N-1)
        norm = [d / (N - 1) for d in degrees]
        return max(norm) - (sum(norm) / N)

    directed = networkx_graph.is_directed()
    connected = (nx.is_strongly_connected(networkx_graph)
                 if directed else nx.is_connected(networkx_graph))

    if connected:
        c = round(compute_centralization(networkx_graph), 3)
        net_cx2.add_network_attribute(
            key=f"{keyprefix}Network Centralization",
            value=str(c),
            datatype=ndex2constants.STRING_DATATYPE
        )
    else:
        if not directed:
            lcc = max(nx.connected_components(networkx_graph), key=len)
            c = round(compute_centralization(networkx_graph.subgraph(lcc)), 3)
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Network Centralization (LCC)",
                value=str(c),
                datatype=ndex2constants.STRING_DATATYPE
            )
        else:
            wcc = max(nx.weakly_connected_components(networkx_graph), key=len)
            scc = max(nx.strongly_connected_components(networkx_graph), key=len)
            c_w = round(compute_centralization(networkx_graph.subgraph(wcc)), 3)
            c_s = round(compute_centralization(networkx_graph.subgraph(scc)), 3)
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Network Centralization (WCC)",
                value=str(c_w),
                datatype=ndex2constants.STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Network Centralization (SCC)",
                value=str(c_s),
                datatype=ndex2constants.STRING_DATATYPE
            )

def add_heterogeneity_net_attrib(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Calculates network heterogeneity (coefficient of variation of degrees) for:
    - Directed/undirected graphs
    - Connected/disconnected networks (using LCC/WCC/SCC as appropriate)
    - MultiGraphs/MultiDiGraphs (counts multi-edges)
    - Rounds to 3 decimal places
    - Uses ndex2constants.STRING_DATATYPE
    
    Heterogeneity = std(degrees) / mean(degrees)
    """
    if not isinstance(networkx_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("Input must be a NetworkX Graph, DiGraph, MultiGraph, or MultiDiGraph")
    
    # --- Early return for empty graph ---
    if len(networkx_graph.nodes()) == 0:
        net_cx2.add_network_attribute(
            key=f"{keyprefix}Network Heterogeneity",
            value="0.0",
            datatype=ndex2constants.STRING_DATATYPE
        )
        return
    
    # --- Helper: Compute heterogeneity for a subgraph ---
    def compute_heterogeneity(G):
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            degrees = [G.degree(n) for n in G.nodes()]  # Counts multi-edges
        else:
            degrees = [d for _, d in G.degree()]
        
        if len(degrees) == 0 or np.mean(degrees) == 0:
            return 0.0
        return np.std(degrees) / (np.mean(degrees) + 1e-10)  # Avoid division by zero
    
    # --- Check connectivity ---
    is_undirected = not networkx_graph.is_directed()
    is_connected = (
        nx.is_connected(networkx_graph) if is_undirected 
        else nx.is_strongly_connected(networkx_graph)
    )
    
    # --- Global heterogeneity ---
    global_hetero = compute_heterogeneity(networkx_graph)
    net_cx2.add_network_attribute(
        key=f"{keyprefix}Network Heterogeneity",
        value=str(round(global_hetero, 3)),
        datatype=ndex2constants.STRING_DATATYPE
    )
    
    # --- Disconnected networks ---
    if not is_connected:
        if is_undirected:
            lcc = max(nx.connected_components(networkx_graph), key=len)
            lcc_hetero = compute_heterogeneity(networkx_graph.subgraph(lcc))
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Network Heterogeneity (LCC)",
                value=str(round(lcc_hetero, 3)),
                datatype=ndex2constants.STRING_DATATYPE
            )
        else:
            wcc = max(nx.weakly_connected_components(networkx_graph), key=len)
            scc = max(nx.strongly_connected_components(networkx_graph), key=len)
            
            wcc_hetero = compute_heterogeneity(networkx_graph.subgraph(wcc))
            scc_hetero = compute_heterogeneity(networkx_graph.subgraph(scc))
            
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Network Heterogeneity (WCC)",
                value=str(round(wcc_hetero, 3)),
                datatype=ndex2constants.STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Network Heterogeneity (SCC)",
                value=str(round(scc_hetero, 3)),
                datatype=ndex2constants.STRING_DATATYPE
            )

def add_connected_components_net_attrib(net_cx2=None, networkx_graph=None, keyprefix=''):
    """Calculates component counts for both directed and undirected networks"""
    
    if networkx_graph.is_directed():
        # Handle directed case
        wccs = nx.number_weakly_connected_components(networkx_graph)
        sccs = nx.number_strongly_connected_components(networkx_graph)
        
        if wccs > 1:
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Connected components (WCCs)",
                value=str(wccs),
                datatype=ndex2constants.STRING_DATATYPE
            )
        
        if sccs > 1:
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Connected components (SCCs)",
                value=str(sccs),
                datatype=ndex2constants.STRING_DATATYPE
            )
    else:
        # Handle undirected case
        components = nx.number_connected_components(networkx_graph)
        if components > 1:
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Connected components",
                value=str(components),
                datatype=ndex2constants.STRING_DATATYPE
            )

def add_avg_degree_net_attrib(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Calculates average‐degree metrics for directed/undirected networks.

    Fully connected:
      • Undirected → “Average Degree”  
      • Directed   → “Average In-Degree” & “Average Out-Degree”

    Disconnected:
      • Always → global “Average Degree”  
      • Undirected → + “Average Degree (LCC)”  
      • Directed   → + “Average Degree (WCC)”, “Average In-Degree (WCC)”, “Average Out-Degree (WCC)”
                     + “Average Degree (SCC)”, “Average In-Degree (SCC)”, “Average Out-Degree (SCC)”
    """
    import networkx as nx
    from ndex2.constants import STRING_DATATYPE

    # validation
    if net_cx2 is None or networkx_graph is None:
        raise ValueError("Both net_cx2 and networkx_graph must be provided")
    if not isinstance(networkx_graph, (nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph)):
        raise ValueError("Requires NetworkX Graph/MultiGraph or DiGraph/MultiDiGraph")

    is_directed = networkx_graph.is_directed()
    N = networkx_graph.number_of_nodes()
    if N == 0:
        return

    # fully‐connected test
    if is_directed:
        fully_conn = nx.is_strongly_connected(networkx_graph)
    else:
        fully_conn = nx.is_connected(networkx_graph)

    # --- fully connected ---
    if fully_conn:
        if is_directed:
            avg_in = sum(d for _, d in networkx_graph.in_degree()) / N
            avg_out = sum(d for _, d in networkx_graph.out_degree()) / N
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Average In-Degree",
                value=str(round(avg_in, 3)),
                datatype=STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Average Out-Degree",
                value=str(round(avg_out, 3)),
                datatype=STRING_DATATYPE
            )
        else:
            avg_deg = sum(d for _, d in networkx_graph.degree()) / N
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Average Degree",
                value=str(round(avg_deg, 3)),
                datatype=STRING_DATATYPE
            )

    # --- disconnected ---
    else:
        # global average degree (total degree)
        avg_deg = sum(d for _, d in networkx_graph.degree()) / N
        net_cx2.add_network_attribute(
            key=f"{keyprefix}Average Degree",
            value=str(round(avg_deg, 3)),
            datatype=STRING_DATATYPE
        )

        if is_directed:
            # Largest Weakly Connected Component
            wccs = list(nx.weakly_connected_components(networkx_graph))
            wcc = max(wccs, key=len)
            G_wcc = networkx_graph.subgraph(wcc)
            n_wcc = G_wcc.number_of_nodes()
            avg_wcc_deg = sum(d for _, d in G_wcc.degree()) / n_wcc
            avg_wcc_in  = sum(d for _, d in G_wcc.in_degree()) / n_wcc
            avg_wcc_out = sum(d for _, d in G_wcc.out_degree()) / n_wcc
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Average Degree (WCC)",
                value=str(round(avg_wcc_deg, 3)),
                datatype=STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Average In-Degree (WCC)",
                value=str(round(avg_wcc_in, 3)),
                datatype=STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Average Out-Degree (WCC)",
                value=str(round(avg_wcc_out, 3)),
                datatype=STRING_DATATYPE
            )

            # Largest Strongly Connected Component
            sccs = list(nx.strongly_connected_components(networkx_graph))
            scc = max(sccs, key=len)
            G_scc = networkx_graph.subgraph(scc)
            n_scc = G_scc.number_of_nodes()
            avg_scc_deg = sum(d for _, d in G_scc.degree()) / n_scc
            avg_scc_in  = sum(d for _, d in G_scc.in_degree()) / n_scc
            avg_scc_out = sum(d for _, d in G_scc.out_degree()) / n_scc
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Average Degree (SCC)",
                value=str(round(avg_scc_deg, 3)),
                datatype=STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Average In-Degree (SCC)",
                value=str(round(avg_scc_in, 3)),
                datatype=STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Average Out-Degree (SCC)",
                value=str(round(avg_scc_out, 3)),
                datatype=STRING_DATATYPE
            )

        else:
            # Largest Connected Component (undirected)
            comps = list(nx.connected_components(networkx_graph))
            lcc = max(comps, key=len)
            G_lcc = networkx_graph.subgraph(lcc)
            n_lcc = G_lcc.number_of_nodes()
            avg_lcc_deg = sum(d for _, d in G_lcc.degree()) / n_lcc
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Average Degree (LCC)",
                value=str(round(avg_lcc_deg, 3)),
                datatype=STRING_DATATYPE
            )
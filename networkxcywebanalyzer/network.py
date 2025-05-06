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
    """Shared helper for CPL (Characteristic Path Lenght) calculations
    
    Args:
        graph: NetworkX graph object
        component_type: None (undirected), 'WCC', or 'SCC'
    
    Returns:
        tuple: (path_length, is_connected)
               path_length may be float or "undefined (single node)"
    """
    if component_type == 'WCC':
        components = list(nx.weakly_connected_components(graph))
    elif component_type == 'SCC':
        components = list(nx.strongly_connected_components(graph))
    else:  # undirected
        components = list(nx.connected_components(graph))
    
    is_connected = len(components) == 1
    
    if is_connected:
        try:
            return nx.average_shortest_path_length(graph), is_connected
        except nx.NetworkXError:
            return "undefined (single node)", is_connected
    else:
        largest = max(components, key=len)
        subgraph = graph.subgraph(largest)
        try:
            return nx.average_shortest_path_length(subgraph), is_connected
        except nx.NetworkXError:
            return "undefined (single node)", is_connected

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

def add_density_net_attrib(net_cx2=None, networkx_graph=None, keyprefix="", density_key="Network Density"):
    """
    Calculates network density, automatically accounting for multi-edges if present.
    - For disconnected networks, only computes density for the largest connected component (LCC/WCC/SCC).
    - For MultiGraph/MultiDiGraph: Counts all parallel edges.
    - For Graph/DiGraph: Standard density calculation.
    - Rounds to 3 decimal places.
    - Uses ndex2constants.STRING_DATATYPE.
    """
    if not isinstance(networkx_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("Input must be a NetworkX Graph, DiGraph, MultiGraph, or MultiDiGraph")
    
    def compute_density(G):
        n = G.number_of_nodes()
        if n < 2:
            return 0.0  # Avoid division by zero
        
        is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
        if is_multi:
            actual_edges = sum(len(data) for _, _, data in G.edges(data=True))
        else:
            actual_edges = G.number_of_edges()
        
        max_edges = n * (n - 1) if G.is_directed() else n * (n - 1) / 2
        return actual_edges / max_edges if max_edges > 0 else 0.0
    
    is_undirected = not networkx_graph.is_directed()
    is_connected = (
        nx.is_connected(networkx_graph) if is_undirected 
        else nx.is_strongly_connected(networkx_graph)
    )
    
    # Skip global density if disconnected
    if is_connected:
        global_density = compute_density(networkx_graph)
        net_cx2.add_network_attribute(
            key=f"{keyprefix}{density_key}",
            value=str(round(global_density, 3)),
            datatype=ndex2constants.STRING_DATATYPE
        )
    else:
        if is_undirected:
            # Undirected: Compute LCC density only
            lcc = max(nx.connected_components(networkx_graph), key=len)
            lcc_subgraph = networkx_graph.subgraph(lcc)
            lcc_density = compute_density(lcc_subgraph)
            net_cx2.add_network_attribute(
                key=f"{keyprefix}{density_key} (LCC)",
                value=str(round(lcc_density, 3)),
                datatype=ndex2constants.STRING_DATATYPE
            )
        else:
            # Directed: Compute WCC and SCC density only
            wcc = max(nx.weakly_connected_components(networkx_graph), key=len)
            scc = max(nx.strongly_connected_components(networkx_graph), key=len)
            
            wcc_subgraph = networkx_graph.subgraph(wcc)
            scc_subgraph = networkx_graph.subgraph(scc)
            
            wcc_density = compute_density(wcc_subgraph)
            scc_density = compute_density(scc_subgraph)
            
            net_cx2.add_network_attribute(
                key=f"{keyprefix}{density_key} (WCC)",
                value=str(round(wcc_density, 3)),
                datatype=ndex2constants.STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}{density_key} (SCC)",
                value=str(round(scc_density, 3)),
                datatype=ndex2constants.STRING_DATATYPE
            )

def add_centralization_net_attrib(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Calculates network centralization matching Cytoscape's behavior.
    - For disconnected networks, only computes centralization for the largest connected component (LCC/WCC/SCC).
    - MultiGraph/MultiDiGraph-aware (counts multi-edges).
    - Rounds to 3 decimal places.
    - Uses ndex2constants.STRING_DATATYPE.
    """
    if not isinstance(networkx_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("Input must be a NetworkX Graph, DiGraph, MultiGraph, or MultiDiGraph")
    
    def compute_centralization(G):
        N = G.number_of_nodes()
        if N <= 1:
            return 0.0
        
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            if G.is_directed():
                degrees = [sum(len(data) for _, _, data in G.edges(n, data=True)) 
                         + sum(len(data) for _, _, data in G.in_edges(n, data=True)) 
                         for n in G.nodes()]
            else:
                degrees = [sum(len(data) for _, _, data in G.edges(n, data=True)) 
                         for n in G.nodes()]
        else:
            degrees = [d for _, d in G.degree()]
        
        normalized_degrees = [d / (N - 1) for d in degrees]
        max_deg = max(normalized_degrees)
        avg_deg = sum(normalized_degrees) / N
        return max_deg - avg_deg
    
    is_undirected = not networkx_graph.is_directed()
    is_connected = (
        nx.is_connected(networkx_graph) if is_undirected 
        else nx.is_strongly_connected(networkx_graph)
    )
    
    # Skip global centralization if disconnected
    if is_connected:
        global_centralization = compute_centralization(networkx_graph)
        net_cx2.add_network_attribute(
            key=f"{keyprefix}Network Centralization",
            value=str(round(global_centralization, 3)),
            datatype=ndex2constants.STRING_DATATYPE
        )
    else:
        if is_undirected:
            # Undirected: Compute LCC centralization only
            lcc = max(nx.connected_components(networkx_graph), key=len)
            lcc_subgraph = networkx_graph.subgraph(lcc)
            lcc_centralization = compute_centralization(lcc_subgraph)
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Network Centralization (LCC)",
                value=str(round(lcc_centralization, 3)),
                datatype=ndex2constants.STRING_DATATYPE
            )
        else:
            # Directed: Compute WCC and SCC centralization only
            wcc = max(nx.weakly_connected_components(networkx_graph), key=len)
            scc = max(nx.strongly_connected_components(networkx_graph), key=len)
            
            wcc_subgraph = networkx_graph.subgraph(wcc)
            scc_subgraph = networkx_graph.subgraph(scc)
            
            wcc_centralization = compute_centralization(wcc_subgraph)
            scc_centralization = compute_centralization(scc_subgraph)
            
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Network Centralization (WCC)",
                value=str(round(wcc_centralization, 3)),
                datatype=ndex2constants.STRING_DATATYPE
            )
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Network Centralization (SCC)",
                value=str(round(scc_centralization, 3)),
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
        wccs = list(nx.weakly_connected_components(networkx_graph))
        sccs = list(nx.strongly_connected_components(networkx_graph))
        
        if len(wccs) > 1:
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Connected components (WCCs)",
                value=str(len(wccs)),
                datatype=ndex2constants.STRING_DATATYPE
            )
        
        if len(sccs) > 1:
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Connected components (SCCs)",
                value=str(len(sccs)),
                datatype=ndex2constants.STRING_DATATYPE
            )
    else:
        # Handle undirected case
        components = list(nx.connected_components(networkx_graph))
        if len(components) > 1:
            net_cx2.add_network_attribute(
                key=f"{keyprefix}Connected components",
                value=str(len(components)),
                datatype=ndex2constants.STRING_DATATYPE
            )
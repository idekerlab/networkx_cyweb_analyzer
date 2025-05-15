import networkx as nx
from ndex2 import constants as ndex2constants

def add_edge_betweenness_centrality(net_cx2=None, networkx_graph=None, keyprefix=''):
    """
    Computes edge Edge Betweenness, adding attributes with context-aware naming:
    - For connected graphs: "Edge Betweenness" (no suffix)
    - For disconnected graphs: "Edge Betweenness (LCC/WCC/SCC)" 
    """
    def _compute_and_add(subgraph, attribute_name):
        edge_betweenness = nx.edge_betweenness_centrality(subgraph)
        for edge, val in edge_betweenness.items():
            edge_id = edge[2]  # Assumes CX2 edge ID is stored in third position of the tuple
            net_cx2.add_edge_attribute(
                edge_id=edge_id,
                key=f"{attribute_name}",
                value=val,
                datatype=ndex2constants.DOUBLE_DATATYPE
            )

    if networkx_graph.is_directed():
        # Check if the graph is weakly connected (WCC = full graph)
        wccs = list(nx.weakly_connected_components(networkx_graph))
        if len(wccs) == 1:
            # Fully connected directed graph → use simple name
            _compute_and_add(networkx_graph, "Edge Betweenness")
        else:
            # Disconnected directed graph → compute for LWCC and LSCC
            largest_wcc = max(wccs, key=len)
            _compute_and_add(networkx_graph.subgraph(largest_wcc), 
                            "Edge Betweenness (WCC)")
            
            sccs = list(nx.strongly_connected_components(networkx_graph))
            largest_scc = max(sccs, key=len)
            _compute_and_add(networkx_graph.subgraph(largest_scc), 
                            "Edge Betweenness (SCC)")
    else:
        # Undirected graph
        ccs = list(nx.connected_components(networkx_graph))
        if len(ccs) == 1:
            # Fully connected undirected graph → use simple name
            _compute_and_add(networkx_graph, "Edge Betweenness")
        else:
            # Disconnected undirected graph → compute for LCC
            largest_cc = max(ccs, key=len)
            _compute_and_add(networkx_graph.subgraph(largest_cc), 
                            "Edge Betweenness (LCC)")
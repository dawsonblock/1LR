import networkx as nx
from lgae_v3.curvature import af3_edge,degree_weighted_af3_proxy

def test_af3_known_graphs():
    assert af3_edge(nx.path_graph(4),1,2)==0.0
    assert af3_edge(nx.complete_graph(3),0,1)==3.0

def test_waf3_proxy_is_explicit_proxy():
    g=nx.star_graph(4)
    value=degree_weighted_af3_proxy(g,0,1)
    assert isinstance(value,float)

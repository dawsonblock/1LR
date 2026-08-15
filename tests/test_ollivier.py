import networkx as nx
import pytest
from lgae_v3.curvature import ollivier_edge

def test_k2_half_idle_identical_measures():
    g=nx.path_graph(2)
    assert ollivier_edge(g,0,1,p=.5)==pytest.approx(1.0)

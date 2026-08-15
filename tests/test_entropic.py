import networkx as nx
import pytest
from lgae_v3.curvature import weak_entropic_node

def test_path_weak_entropic_near_zero():
    g=nx.path_graph(4)
    assert weak_entropic_node(g,1)==pytest.approx(0.0,abs=1e-6)

def test_cycle4_positive_weak_entropic():
    g=nx.cycle_graph(4)
    assert weak_entropic_node(g,0)>0

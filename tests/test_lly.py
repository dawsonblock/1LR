import networkx as nx
import pytest
from lgae_v3.curvature import lly_laplacian_lp,lly_half_idleness,crosscheck_lly,integral_lly_deficit

@pytest.mark.parametrize("g",[nx.path_graph(4),nx.cycle_graph(4),nx.complete_graph(3)])
def test_two_exact_lly_paths_agree(g):
    result=crosscheck_lly(g)
    assert result["ok"],result


def test_known_lly_values():
    g=nx.path_graph(2)
    assert lly_laplacian_lp(g,0,1)==pytest.approx(2.0)
    assert lly_half_idleness(g,0,1)==pytest.approx(2.0)
    k3=nx.complete_graph(3)
    assert lly_laplacian_lp(k3,0,1)==pytest.approx(1.5)


def test_integral_lly_is_deficit_lower_is_better():
    assert integral_lly_deficit([1.0,0.5],0.0)==0.0
    assert integral_lly_deficit([-1.0,0.5],0.0)==1.0

import networkx as nx
import torch
import pytest
from lgae_v3.curvature import bakry_emery_curvature,sampled_cde_prime_residual

def generator(g):
    n=len(g); P=torch.zeros(n,n,dtype=torch.float64)
    for u in g:
        for v in g.neighbors(u): P[u,v]=1/g.degree[u]
    return P-torch.eye(n,dtype=torch.float64)

def test_k2_bakry():
    Q=generator(nx.path_graph(2))
    assert bakry_emery_curvature(Q,0)==pytest.approx(2.0)

def test_sampled_cde_nonnegative_violation_metric():
    Q=generator(nx.cycle_graph(4))
    assert sampled_cde_prime_residual(Q,samples=8)>=0

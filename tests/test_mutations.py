from lgae_v3.types import make_graph_buffers
from lgae_v3.mutations import AddEdge,PruneEdge,ReweightEdge

def test_fixed_capacity_mutations():
    g=make_graph_buffers(4,[(0,1),(1,2)],capacity=4)
    AddEdge(2,3).apply(g); assert g.edge_count==3
    ReweightEdge(2,3,factor=2).apply(g)
    assert float(g.weight[g.valid][-1])==2.0
    PruneEdge(2,3).apply(g); assert g.edge_count==2

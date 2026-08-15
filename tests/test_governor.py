import torch
from lgae_v3 import LGAEConfig
from lgae_v3.types import make_graph_buffers,MutationDecision
from lgae_v3.governor import GeometryGovernor
from lgae_v3.mutations import AddEdge

def small_cfg():
    c=LGAEConfig(); c.fiber.d_base=2;c.fiber.d_max=4
    c.audit.exact_lly_top_k=64;c.audit.entropic_nodes=3;c.audit.bakry_nodes=2;c.audit.cde_nodes=2;c.audit.cde_samples=4
    c.audit.max_integral_lly_deficit=1e9;c.audit.max_cde_residual=1e9;c.audit.max_topology_drift=1e9;c.audit.entropic_drop_tolerance=1e9
    c.operator.diagnostic_k=3
    return c

def test_shadow_mutation_does_not_touch_original():
    graph=make_graph_buffers(4,[(0,1),(1,2),(2,3)],capacity=8)
    z=torch.randn(4,4)
    gov=GeometryGovernor(small_cfg())
    result,shadow=gov.evaluate_mutation(graph,z,AddEdge(0,2))
    assert graph.edge_count==3
    assert shadow.edge_count==4
    assert result.decision in {MutationDecision.ACCEPT,MutationDecision.REJECT,MutationDecision.QUARANTINE}

def test_sampled_lly_causes_quarantine_when_not_global():
    c=small_cfg(); c.audit.exact_lly_top_k=1; c.mutation.quarantine_on_uncertainty=True
    graph=make_graph_buffers(5,[(0,1),(1,2),(2,3),(3,4)],capacity=8)
    result,_=GeometryGovernor(c).evaluate_mutation(graph,torch.randn(5,4),AddEdge(0,2))
    assert result.decision in {MutationDecision.QUARANTINE,MutationDecision.REJECT}

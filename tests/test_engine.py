import torch
from lgae_v3 import LGAEEngine,LGAEConfig
from lgae_v3.types import make_graph_buffers

def test_engine_smoke_and_compatibility_imports():
    from lgae_v3.core.geometry_field import GeometryField
    from lgae_v3.core.fiber import Fiber
    from lgae_v3.core.evolution import EvolutionEngine
    assert GeometryField is not None and Fiber is not None and EvolutionEngine is LGAEEngine
    cfg=LGAEConfig(); cfg.fiber.d_base=2;cfg.fiber.d_max=4;cfg.fiber.spawn_width=1;cfg.operator.diagnostic_k=3
    cfg.audit.exact_lly_top_k=16;cfg.audit.entropic_nodes=2;cfg.audit.bakry_nodes=1;cfg.audit.cde_nodes=1;cfg.audit.cde_samples=2
    g=make_graph_buffers(4,[(0,1),(1,2),(2,3)],capacity=8)
    e=LGAEEngine(g,cfg)
    z0=e().detach().clone(); e.diffuse_(.01); z1=e().detach()
    assert z0.shape==z1.shape==(4,4)
    out=e.fiber_tick(residual=z1)
    assert "capacity" in out
    audit=e.audit(); assert audit.lambda2>=-1e-6

def test_cli_demo_serializes(capsys):
    from lgae_v3.cli import main
    assert main(["demo","--nodes","6","--steps","1"])==0
    assert '"version": "5.0.0"' in capsys.readouterr().out

def test_quarantine_resolution_does_not_require_parameter_replacement():
    from lgae_v3.mutations import AddEdge
    cfg=LGAEConfig(); cfg.fiber.d_base=2;cfg.fiber.d_max=4;cfg.operator.diagnostic_k=3
    cfg.audit.exact_lly_top_k=1;cfg.audit.entropic_nodes=1;cfg.audit.bakry_nodes=1;cfg.audit.cde_nodes=1;cfg.audit.cde_samples=1
    e=LGAEEngine(make_graph_buffers(5,[(0,1),(1,2),(2,3),(3,4)],capacity=8),cfg)
    param_id=id(e.fibers.latent)
    r=e.evaluate_and_maybe_commit(AddEdge(0,2))
    if r.decision.value=="quarantine": e.resolve_quarantine(0,accept=False)
    assert id(e.fibers.latent)==param_id

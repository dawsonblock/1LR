from pathlib import Path

import torch

from lgae_v3 import LGAEConfig, LGAEEngine
from lgae_v3.mutations import AddEdge
from lgae_v3.types import MutationDecision, make_graph_buffers


def cfg_for_transactions():
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 4
    cfg.fiber.spawn_width = 1
    cfg.operator.diagnostic_k = 3
    cfg.audit.exact_lly_top_k = 64
    cfg.audit.entropic_nodes = 4
    cfg.audit.bakry_nodes = 2
    cfg.audit.cde_nodes = 1
    cfg.audit.cde_samples = 2
    cfg.audit.entropic_require_success = True
    return cfg


def test_shadow_steps_are_executed_and_reported():
    cfg = cfg_for_transactions()
    cfg.mutation.shadow_steps = 3
    engine = LGAEEngine(make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8), cfg)
    r = engine.evaluate_and_maybe_commit(AddEdge(0, 2))
    assert r.metadata["shadow_steps"] == 3
    assert r.metadata["shadow_latent_delta_norm"] >= 0.0


def test_stale_quarantine_cannot_overwrite_newer_graph():
    cfg = cfg_for_transactions()
    cfg.audit.exact_lly_top_k = 1  # forces uncertainty/quarantine on multi-edge graph
    engine = LGAEEngine(make_graph_buffers(5, [(0, 1), (1, 2), (2, 3), (3, 4)], capacity=10), cfg)
    r = engine.evaluate_and_maybe_commit(AddEdge(0, 2))
    assert r.decision in {MutationDecision.QUARANTINE, MutationDecision.REJECT}
    if r.decision is MutationDecision.REJECT:
        return
    AddEdge(0, 4).apply(engine.graph)  # simulate a newer authoritative commit
    resolved = engine.resolve_quarantine(0, accept=True)
    assert resolved.decision is MutationDecision.REJECT
    assert "stale_quarantine_base" in resolved.reasons


def test_governed_fiber_mutation_rolls_back_on_hard_failure():
    cfg = cfg_for_transactions()
    cfg.fiber.score_threshold = -999.0
    cfg.fiber.persistence_steps = 1
    cfg.fiber.gamma_quantile = 0.25
    cfg.audit.max_operator_discrepancy = 0.0  # any diagnostic change must reject
    engine = LGAEEngine(make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8), cfg)
    before = engine.fibers.active_mask.clone()
    out = engine.fiber_tick(residual=torch.ones(4, 4))
    if out["decision"] is not None:
        assert out["decision"] is MutationDecision.REJECT
        assert torch.equal(engine.fibers.active_mask, before)


def test_checkpoint_roundtrip_restores_graph_and_fibers(tmp_path: Path):
    cfg = cfg_for_transactions()
    engine = LGAEEngine(make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8), cfg)
    engine.diffuse_(0.02)
    AddEdge(0, 2).apply(engine.graph)
    path = tmp_path / "checkpoint.pt"
    engine.save_checkpoint(path)
    expected_graph_hash = engine.graph.state_hash()
    expected_fiber_hash = engine.fibers.state_hash()
    expected_step = engine.step_index

    restored = LGAEEngine(make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8), cfg)
    restored.load_checkpoint_(path)
    assert restored.graph.state_hash() == expected_graph_hash
    assert restored.fibers.state_hash() == expected_fiber_hash
    assert restored.step_index == expected_step

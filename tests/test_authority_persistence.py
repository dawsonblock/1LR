"""Tests for v3.3 Authority and Persistence Hardening.

Covers:
- P0: slot_generation included in graph state hash
- P0: graph/gauge generation synchronization
- P0: canonical authority hash
- P0: checkpoint config enforcement (structural + governance)
- P0: optimizer state semantics on checkpoint load
- P1: safetensors + JSON checkpoint format
- P1: optimizer-generic slot reset (Adagrad, etc.)
- P2: hash-chained receipts
- P2: receipts bind gauge authority hash
"""
from __future__ import annotations

from pathlib import Path
import json

import torch
from torch import nn
import pytest

from lgae_v3 import LGAEConfig, LGAEEngine, SOConnectionBank, make_graph_buffers
from lgae_v3.config import config_structural_hash, config_governance_hash
from lgae_v3.mutations import AddEdge, PruneEdge
from lgae_v3.receipts import mutation_receipt, append_receipt, verify_receipt_chain
from lgae_v3.types import MutationDecision, MutationResult


def _cfg():
    c = LGAEConfig()
    c.fiber.d_base = 2
    c.fiber.d_max = 4
    c.fiber.gauge_dim = 2
    c.fiber.gauge_parameterization = "cayley"
    c.audit.exact_lly_top_k = 16
    c.audit.entropic_nodes = 3
    c.audit.bakry_nodes = 1
    c.audit.cde_nodes = 1
    c.audit.cde_samples = 2
    return c


# ---------------------------------------------------------------------------
# P0: slot_generation in graph state hash
# ---------------------------------------------------------------------------

def test_slot_generation_included_in_graph_state_hash():
    """Altering slot_generation must change the graph state hash."""
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4)
    h_before = graph.state_hash()
    # Tamper with slot_generation
    assert graph.slot_generation is not None
    graph.slot_generation[0] += 100
    h_after = graph.state_hash()
    assert h_before != h_after, "slot_generation must be cryptographically committed in state_hash"


def test_state_hash_stable_under_unchanged_generation():
    """Unchanged slot_generation should produce the same hash."""
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4)
    h1 = graph.state_hash()
    h2 = graph.state_hash()
    assert h1 == h2


# ---------------------------------------------------------------------------
# P0: graph/gauge generation synchronization
# ---------------------------------------------------------------------------

def test_generation_sync_at_init():
    """Gauge bank generations should match graph generations after engine init."""
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4)
    # Set nonzero generations
    graph.slot_generation[0] = 5
    graph.slot_generation[1] = 3
    engine = LGAEEngine(graph, _cfg())
    assert engine.gauge_connections is not None
    assert torch.equal(engine.graph.slot_generation, engine.gauge_connections.slot_generation)


def test_generation_sync_after_mutation():
    """After a committed mutation, graph and gauge generations must match."""
    graph = make_graph_buffers(3, [(0, 1), (1, 2), (0, 2)], capacity=4, dtype=torch.float64)
    cfg = _cfg()
    engine = LGAEEngine(graph, cfg, dtype=torch.float64)
    engine.step_index += 25
    res = engine.evaluate_and_maybe_commit(PruneEdge(0, 1))
    assert res.decision.value == "accept"
    engine.assert_generation_sync()  # should not raise


def test_assert_generation_sync_detects_divergence():
    """Manually desyncing generations should cause assert_generation_sync to raise."""
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4)
    engine = LGAEEngine(graph, _cfg())
    assert engine.gauge_connections is not None
    engine.gauge_connections.slot_generation[0] += 999
    with pytest.raises(RuntimeError, match="divergence"):
        engine.assert_generation_sync()


# ---------------------------------------------------------------------------
# P0: Canonical authority hash
# ---------------------------------------------------------------------------

def test_authority_hash_changes_on_graph_mutation():
    """Authority hash should change after a committed graph mutation."""
    graph = make_graph_buffers(3, [(0, 1), (1, 2), (0, 2)], capacity=4, dtype=torch.float64)
    engine = LGAEEngine(graph, _cfg(), dtype=torch.float64)
    h_before = engine.authority_hash()
    engine.step_index += 25
    res = engine.evaluate_and_maybe_commit(PruneEdge(0, 1))
    assert res.decision.value == "accept"
    h_after = engine.authority_hash()
    assert h_before != h_after


def test_authority_hash_changes_on_governance_config():
    """Authority hash should differ for different governance configs."""
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4)
    cfg1 = _cfg()
    cfg2 = _cfg()
    cfg2.audit.min_lambda2 = 0.5
    e1 = LGAEEngine(graph.clone(), cfg1)
    e2 = LGAEEngine(graph.clone(), cfg2)
    assert e1.authority_hash() != e2.authority_hash()


# ---------------------------------------------------------------------------
# P0: Checkpoint config enforcement
# ---------------------------------------------------------------------------

def test_structural_config_mismatch_rejected(tmp_path: Path):
    """Loading a checkpoint with a structural config mismatch must fail."""
    cfg1 = _cfg()
    cfg2 = _cfg()
    cfg2.fiber.d_max = 8  # structural change
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4)
    e1 = LGAEEngine(graph, cfg1)
    path = tmp_path / "ckpt.pt"
    e1.save_checkpoint(path)
    e2 = LGAEEngine(make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4), cfg2)
    with pytest.raises(ValueError, match="structural config mismatch"):
        e2.load_checkpoint_(path)


def test_governance_config_mismatch_rejected_by_default(tmp_path: Path):
    """Loading a checkpoint with a governance mismatch must fail by default."""
    cfg1 = _cfg()
    cfg2 = _cfg()
    cfg2.audit.min_lambda2 = 0.987654  # governance change
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4)
    e1 = LGAEEngine(graph, cfg1)
    path = tmp_path / "ckpt.pt"
    e1.save_checkpoint(path)
    e2 = LGAEEngine(make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4), cfg2)
    with pytest.raises(ValueError, match="governance config mismatch"):
        e2.load_checkpoint_(path)


def test_governance_config_mismatch_allowed_with_flag(tmp_path: Path):
    """Governance mismatch should be accepted with allow_governance_mismatch=True."""
    cfg1 = _cfg()
    cfg2 = _cfg()
    cfg2.audit.min_lambda2 = 0.987654
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4)
    e1 = LGAEEngine(graph, cfg1)
    path = tmp_path / "ckpt.pt"
    e1.save_checkpoint(path)
    e2 = LGAEEngine(make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4), cfg2)
    e2.load_checkpoint_(path, allow_governance_mismatch=True)  # should not raise
    assert e2.step_index == e1.step_index


# ---------------------------------------------------------------------------
# P0: Optimizer state semantics on checkpoint load
# ---------------------------------------------------------------------------

def test_optimizer_state_restored_from_checkpoint(tmp_path: Path):
    """Optimizer state should be restored from checkpoint when policy='restore'."""
    cfg = _cfg()
    graph = make_graph_buffers(3, [(0, 1), (1, 2), (0, 2)], capacity=4, dtype=torch.float64)
    engine = LGAEEngine(graph, cfg, dtype=torch.float64)
    bank = engine.gauge_connections
    assert bank is not None
    decoder = nn.Linear(4, 2, dtype=torch.float64)
    core = nn.ModuleList([bank, decoder])
    opt = torch.optim.Adam(core.parameters(), lr=0.1)
    engine.register_optimizer(opt)
    # Accumulate momentum
    loss = (bank.raw_generators - 1.0).square().sum()
    loss.backward()
    opt.step()
    state = opt.state[bank.raw_generators]
    assert torch.count_nonzero(state["exp_avg"]).item() > 0
    path = tmp_path / "ckpt.pt"
    engine.save_checkpoint(path)
    # Load into a fresh engine with a fresh optimizer
    engine2 = LGAEEngine(make_graph_buffers(3, [(0, 1), (1, 2), (0, 2)], capacity=4, dtype=torch.float64), cfg, dtype=torch.float64)
    bank2 = engine2.gauge_connections
    decoder2 = nn.Linear(4, 2, dtype=torch.float64)
    core2 = nn.ModuleList([bank2, decoder2])
    opt2 = torch.optim.Adam(core2.parameters(), lr=0.1)
    engine2.register_optimizer(opt2)
    engine2.load_checkpoint_(path, optimizer_load_policy="restore")
    # The optimizer state for bank2 should be restored (nonzero exp_avg)
    state2 = opt2.state[bank2.raw_generators]
    assert "exp_avg" in state2
    assert torch.count_nonzero(state2["exp_avg"]).item() > 0


def test_optimizer_state_reset_when_checkpoint_lacks_it(tmp_path: Path):
    """If checkpoint has no optimizer state, registered optimizers should be reset."""
    cfg = _cfg()
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4, dtype=torch.float64)
    engine = LGAEEngine(graph, cfg, dtype=torch.float64)
    path = tmp_path / "ckpt.pt"
    engine.save_checkpoint(path)
    # Fresh engine with optimizer that has stale state
    engine2 = LGAEEngine(make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4, dtype=torch.float64), cfg, dtype=torch.float64)
    bank2 = engine2.gauge_connections
    assert bank2 is not None
    opt2 = torch.optim.Adam(bank2.parameters(), lr=0.1)
    engine2.register_optimizer(opt2)
    # Accumulate stale momentum
    loss = (bank2.raw_generators - 1.0).square().sum()
    loss.backward()
    opt2.step()
    assert torch.count_nonzero(opt2.state[bank2.raw_generators]["exp_avg"]).item() > 0
    # Load checkpoint (which has no optimizer state since engine had none registered)
    engine2.load_checkpoint_(path, optimizer_load_policy="restore")
    # Stale optimizer state should be cleared
    assert len(opt2.state) == 0 or all(
        len(s) == 0 for s in opt2.state.values()
    )


def test_optimizer_reject_policy_blocks_load(tmp_path: Path):
    """optimizer_load_policy='reject' should block load when optimizers are registered."""
    cfg = _cfg()
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4)
    engine = LGAEEngine(graph, cfg)
    path = tmp_path / "ckpt.pt"
    engine.save_checkpoint(path)
    engine2 = LGAEEngine(make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4), cfg)
    opt = torch.optim.Adam(engine2.fibers.parameters(), lr=0.1)
    engine2.register_optimizer(opt)
    with pytest.raises(RuntimeError, match="rejected"):
        engine2.load_checkpoint_(path, optimizer_load_policy="reject")


# ---------------------------------------------------------------------------
# P1: Safetensors + JSON checkpoint format
# ---------------------------------------------------------------------------

def test_safe_checkpoint_roundtrip(tmp_path: Path):
    """Safe checkpoint format (safetensors + JSON) should roundtrip correctly."""
    cfg = _cfg()
    graph = make_graph_buffers(3, [(0, 1), (1, 2), (0, 2)], capacity=4, dtype=torch.float64)
    engine = LGAEEngine(graph, cfg, dtype=torch.float64)
    engine.diffuse_(0.01)
    path = tmp_path / "safe_ckpt"
    engine.save_checkpoint(path)
    # Verify directory structure
    assert (path / "manifest.json").exists()
    assert (path / "tensors.safetensors").exists()
    assert (path / "graph.json").exists()
    assert (path / "controller.json").exists()
    assert (path / "governance.json").exists()
    # Load into fresh engine
    engine2 = LGAEEngine(make_graph_buffers(3, [(0, 1), (1, 2), (0, 2)], capacity=4, dtype=torch.float64), cfg, dtype=torch.float64)
    engine2.load_checkpoint_(path)
    assert engine2.graph.state_hash() == engine.graph.state_hash()
    assert engine2.fibers.state_hash() == engine.fibers.state_hash()
    assert engine2.step_index == engine.step_index


def test_safe_checkpoint_enforces_config_authority(tmp_path: Path):
    """Safe checkpoint should also enforce structural config authority."""
    cfg1 = _cfg()
    cfg2 = _cfg()
    cfg2.fiber.d_max = 8
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4)
    e1 = LGAEEngine(graph, cfg1)
    path = tmp_path / "safe_ckpt"
    e1.save_checkpoint(path)
    e2 = LGAEEngine(make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4), cfg2)
    with pytest.raises(ValueError, match="structural config mismatch"):
        e2.load_checkpoint_(path)


# ---------------------------------------------------------------------------
# P1: Optimizer-generic slot reset
# ---------------------------------------------------------------------------

def test_optimizer_generic_reset_handles_adagrad():
    """reset_slots should clear Adagrad 'sum' state, not just Adam keys."""
    bank = SOConnectionBank(4, 3, dtype=torch.float64)
    opt = torch.optim.Adagrad(bank.parameters(), lr=0.1)
    # Accumulate Adagrad state
    loss = (bank.raw_generators - 1.0).square().sum()
    loss.backward()
    opt.step()
    state = opt.state[bank.raw_generators]
    assert "sum" in state
    assert torch.count_nonzero(state["sum"][0]).item() > 0
    # Reset slot 0
    bank.reset_slots(torch.tensor([0]), optimizers=[opt])
    # Adagrad 'sum' for slot 0 should be zeroed
    assert torch.count_nonzero(state["sum"][0]).item() == 0
    # Slot 1 should be preserved
    assert torch.count_nonzero(state["sum"][1]).item() > 0


def test_optimizer_generic_reset_preserves_scalar_state():
    """reset_slots should not zero scalar state like step counters."""
    bank = SOConnectionBank(4, 3, dtype=torch.float64)
    opt = torch.optim.Adam(bank.parameters(), lr=0.1)
    loss = (bank.raw_generators - 1.0).square().sum()
    loss.backward()
    opt.step()
    state = opt.state[bank.raw_generators]
    assert "step" in state
    step_before = state["step"]
    bank.reset_slots(torch.tensor([0]), optimizers=[opt])
    # step counter should be preserved (it's a scalar, not slot-indexed)
    assert state["step"] == step_before


# ---------------------------------------------------------------------------
# P2: Hash-chained receipts
# ---------------------------------------------------------------------------

def test_receipt_hash_chain(tmp_path: Path):
    """Receipts should form a tamper-evident hash chain."""
    path = tmp_path / "ledger.jsonl"
    r1 = mutation_receipt(
        MutationResult(MutationDecision.ACCEPT, ["test"]),
        authority_state_hash_before="abc",
        authority_state_hash_after="def",
    )
    append_receipt(path, r1)
    r2 = mutation_receipt(
        MutationResult(MutationDecision.REJECT, ["test2"]),
        authority_state_hash_before="def",
        authority_state_hash_after="ghi",
    )
    append_receipt(path, r2)
    # Verify chain
    is_valid, errors = verify_receipt_chain(path)
    assert is_valid, f"chain errors: {errors}"
    # Read back and verify linkage
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["receipt_index"] == 0
    assert first["previous_receipt_hash"] is None
    assert second["receipt_index"] == 1
    assert second["previous_receipt_hash"] == first["sha256"]


def test_receipt_chain_detects_tampering(tmp_path: Path):
    """Tampering with a receipt should break the chain verification."""
    path = tmp_path / "ledger.jsonl"
    r1 = mutation_receipt(MutationResult(MutationDecision.ACCEPT, ["test"]))
    append_receipt(path, r1)
    r2 = mutation_receipt(MutationResult(MutationDecision.REJECT, ["test2"]))
    append_receipt(path, r2)
    # Tamper: delete the first receipt
    lines = path.read_text().strip().split("\n")
    path.write_text(lines[1] + "\n")
    is_valid, errors = verify_receipt_chain(path)
    assert not is_valid
    assert len(errors) > 0


def test_receipt_binds_gauge_authority_hash():
    """Mutation receipts should include gauge authority hash."""
    result = MutationResult(MutationDecision.ACCEPT, ["test"])
    receipt = mutation_receipt(
        result,
        gauge_authority_hash="gauge_hash_abc123",
        authority_state_hash_before="before",
        authority_state_hash_after="after",
    )
    assert receipt["gauge_authority_hash"] == "gauge_hash_abc123"
    assert receipt["authority_state_hash_before"] == "before"
    assert receipt["authority_state_hash_after"] == "after"


# ---------------------------------------------------------------------------
# P2: Receipts bind gauge authority in engine mutations
# ---------------------------------------------------------------------------

def test_engine_mutation_receipt_binds_gauge_hash():
    """Accepted mutations should record gauge hash in metadata."""
    cfg = _cfg()
    graph = make_graph_buffers(3, [(0, 1), (1, 2), (0, 2)], capacity=4, dtype=torch.float64)
    engine = LGAEEngine(graph, cfg, dtype=torch.float64)
    engine.step_index += 25
    res = engine.evaluate_and_maybe_commit(PruneEdge(0, 1))
    if res.decision.value == "accept":
        assert "base_gauge_hash" in res.metadata
        assert "authority_hash_after" in res.metadata

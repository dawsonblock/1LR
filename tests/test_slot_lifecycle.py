import torch
from torch import nn
import pytest

from lgae_v3 import LGAEConfig, LGAEEngine, LGAETrainCore, SOConnectionBank, make_graph_buffers
from lgae_v3.mutations import AddEdge, PruneEdge
from lgae_v3.training import padded_markov_edges_with_slots, train_step


def test_reused_slot_resets_adam_exp_avg_and_prevents_momentum_leak():
    """Verify that pruning/reusing a slot resets Adam momentum and avoids drift on zero gradient."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 4
    cfg.fiber.gauge_dim = 2
    cfg.fiber.gauge_parameterization = "cayley"
    cfg.audit.exact_lly_top_k = 16
    cfg.audit.entropic_nodes = 3
    cfg.audit.bakry_nodes = 1
    cfg.audit.cde_nodes = 1
    cfg.audit.cde_samples = 2

    # Graph with 3 nodes and 3 edges (triangle cycle) in capacity 4
    graph = make_graph_buffers(3, [(0, 1), (1, 2), (0, 2)], capacity=4, dtype=torch.float64)
    engine = LGAEEngine(graph, cfg, dtype=torch.float64)
    assert engine.gauge_connections is not None
    bank = engine.gauge_connections

    decoder = nn.Linear(4, 2, dtype=torch.float64)
    core = LGAETrainCore(engine.fibers, decoder, gauge_bank=bank, gauge_dim=2)
    optimizer = torch.optim.Adam(core.parameters(), lr=0.1)
    engine.register_optimizer(optimizer)

    # 1. Train slot 0 with gradients
    src, dst, w, valid, slot, reverse = padded_markov_edges_with_slots(engine.graph, max_edges=8)
    target = torch.ones(3, 2, dtype=torch.float64)
    pressure = torch.zeros(3, dtype=torch.float64)
    out = train_step(
        core, engine, optimizer,
        target=target, src=src, dst=dst, weight=w, valid=valid,
        bottleneck_pressure=pressure, edge_slot=slot, reverse=reverse,
    )
    assert out["loss"] is not None

    # Verify Adam accumulated momentum for slot 0
    state = optimizer.state[bank.raw_generators]
    assert "exp_avg" in state and "exp_avg_sq" in state
    assert torch.count_nonzero(state["exp_avg"][0]).item() > 0
    assert torch.count_nonzero(state["exp_avg_sq"][0]).item() > 0

    # 2. Prune edge (0, 1) which occupies slot 0
    gen_before = int(engine.graph.slot_generation[0].item())
    res_prune = engine.evaluate_and_maybe_commit(PruneEdge(0, 1))
    assert res_prune.decision.value == "accept"
    assert int(engine.graph.slot_generation[0].item()) > gen_before

    # Verify slot 0 generator and Adam momentum are completely zeroed
    assert torch.count_nonzero(bank.raw_generators[0]).item() == 0
    assert torch.count_nonzero(state["exp_avg"][0]).item() == 0
    assert torch.count_nonzero(state["exp_avg_sq"][0]).item() == 0

    # 3. Advance step beyond cooldown and add back edge (0, 1) that reuses slot 0
    engine.step_index += 25
    gen_before_reuse = int(engine.graph.slot_generation[0].item())
    res_add = engine.evaluate_and_maybe_commit(AddEdge(0, 1))
    assert res_add.decision.value == "accept"
    assert int(engine.graph.slot_generation[0].item()) > gen_before_reuse

    # 4. Take a dummy step with zero gradient on the connection bank
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()

    # Verify slot 0 connection matrix remains strictly identity without historical momentum drift
    eye2 = torch.eye(2, dtype=torch.float64)
    assert torch.allclose(bank.matrices()[0], eye2, atol=1e-12)


def test_unchanged_slot_preserves_optimizer_state():
    """Verify that resetting slot 0 preserves optimizer state for untouched slot 1."""
    bank = SOConnectionBank(4, 3, dtype=torch.float64)
    opt = torch.optim.Adam(bank.parameters(), lr=0.1)

    # Accumulate fake momentum across all slots
    loss = (bank.raw_generators - 1.0).square().sum()
    loss.backward()
    opt.step()

    state = opt.state[bank.raw_generators]
    m0_before = state["exp_avg"][0].clone()
    m1_before = state["exp_avg"][1].clone()
    assert torch.count_nonzero(m0_before).item() > 0
    assert torch.count_nonzero(m1_before).item() > 0

    # Reset slot 0 only
    bank.reset_slots(torch.tensor([0]), optimizers=[opt])

    # Slot 0 is zeroed
    assert torch.count_nonzero(state["exp_avg"][0]).item() == 0
    assert torch.count_nonzero(state["exp_avg_sq"][0]).item() == 0

    # Slot 1 is completely preserved
    assert torch.allclose(state["exp_avg"][1], m1_before)
    assert torch.count_nonzero(state["exp_avg"][1]).item() > 0


def test_slot_generation_monotonic_increment():
    """Verify that slot generations monotonically increase and survive state dict roundtrips."""
    graph = make_graph_buffers(3, [(0, 1)], capacity=4)
    assert graph.slot_generation is not None
    assert graph.slot_generation[0].item() == 1
    assert graph.slot_generation[1].item() == 0

    PruneEdge(0, 1).apply(graph)
    assert graph.slot_generation[0].item() == 2

    AddEdge(0, 2).apply(graph)
    # Reused slot 0
    assert graph.slot_generation[0].item() == 3

    # Roundtrip through state dict
    sd = graph.to_state_dict()
    restored = graph.from_state_dict(sd)
    assert restored.slot_generation is not None
    assert restored.slot_generation[0].item() == 3

import torch
from torch import nn

from lgae_v3 import LGAEConfig, LGAEEngine, LGAETrainCore, padded_markov_edges
from lgae_v3.compile_utils import compile_if_enabled
from lgae_v3.config import CompileConfig
from lgae_v3.training.loop import train_step
from lgae_v3.types import make_graph_buffers


def test_merged_sparse_training_core_and_loop():
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 4
    cfg.operator.diagnostic_k = 3
    cfg.audit.exact_lly_top_k = 32
    cfg.audit.entropic_nodes = 2
    cfg.audit.bakry_nodes = 1
    cfg.audit.cde_nodes = 1
    cfg.audit.cde_samples = 1
    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    core = LGAETrainCore(engine.fibers, nn.Linear(4, 2))
    opt = torch.optim.AdamW(core.parameters(), lr=1e-3)
    src, dst, weight, valid = padded_markov_edges(graph, max_edges=16)
    target = torch.randn(4, 2)
    pressure = torch.zeros(4)
    before = id(engine.fibers.latent)
    out = train_step(
        core, engine, opt,
        target=target, src=src, dst=dst, weight=weight, valid=valid,
        bottleneck_pressure=pressure, step=1, spawn_interval=100,
    )
    assert torch.isfinite(out["loss"])
    assert id(engine.fibers.latent) == before


def test_compile_utils_disabled():
    m = nn.Linear(4, 2)
    res = compile_if_enabled(m, CompileConfig(enabled=False))
    assert res is m

"""LGAE-v3: geometry-governed self-evolving graph/latent controller."""
from .config import LGAEConfig, load_config, config_structural_hash, config_governance_hash
from .evolution import LGAEEngine
from .fibers import FixedWidthFiberLatent, FiberController, SOConnectionBank, project_to_so_d
from .operators import DualOperatorState, SparseDualOperatorState
from .types import (
    EdgeRole,
    GraphBuffers,
    make_graph_buffers,
    make_bucketed_graph_buffers,
    round_edge_capacity,
    MutationDecision,
    MutationResult,
)
from .training import (
    LGAETrainCore, train_step, padded_markov_edges, refresh_padded_markov_edges_,
    padded_markov_edges_with_slots, refresh_padded_markov_edges_with_slots_,
)
from .mutations import RicciFlowReweight, MutationCooldownTracker

__all__ = [
    "LGAEConfig", "load_config", "config_structural_hash", "config_governance_hash",
    "LGAEEngine", "FixedWidthFiberLatent", "FiberController", "SOConnectionBank", "project_to_so_d",
    "DualOperatorState", "SparseDualOperatorState", "EdgeRole", "GraphBuffers", "make_graph_buffers", "make_bucketed_graph_buffers", "round_edge_capacity",
    "MutationDecision", "MutationResult", "RicciFlowReweight", "MutationCooldownTracker", "LGAETrainCore", "train_step", "padded_markov_edges", "refresh_padded_markov_edges_", "padded_markov_edges_with_slots", "refresh_padded_markov_edges_with_slots_",
]
__version__ = "4.1.0"

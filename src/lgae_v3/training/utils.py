from __future__ import annotations

import torch
from torch import Tensor

from ..operators import actuation_markov_edges, actuation_markov_edges_with_slots
from ..types import GraphBuffers, round_edge_capacity


def padded_markov_edges(
    graph: GraphBuffers,
    max_edges: int | None = None,
    *,
    bucket_size: int = 256,
    reserve_buckets: int = 1,
):
    """Build fixed-shape directed Markov buffers for compiled training kernels."""
    src, dst, weight = actuation_markov_edges(graph)
    e = src.numel()
    if max_edges is None:
        cap = round_edge_capacity(e, bucket_size=bucket_size, reserve_buckets=reserve_buckets)
    else:
        cap = int(max_edges)
        if cap < e:
            raise ValueError("max_edges is smaller than active directed edge count")
    src_pad = torch.zeros(cap, dtype=torch.long, device=src.device)
    dst_pad = torch.zeros(cap, dtype=torch.long, device=dst.device)
    w_pad = torch.zeros(cap, dtype=weight.dtype, device=weight.device)
    valid = torch.zeros(cap, dtype=torch.bool, device=weight.device)
    src_pad[:e] = src
    dst_pad[:e] = dst
    w_pad[:e] = weight
    valid[:e] = True
    return src_pad, dst_pad, w_pad, valid


@torch.no_grad()
def refresh_padded_markov_edges_(
    graph: GraphBuffers,
    src_pad: Tensor,
    dst_pad: Tensor,
    weight_pad: Tensor,
    valid: Tensor,
) -> int:
    """Refresh values in fixed-capacity edge buffers without changing tensor metadata."""
    if not (src_pad.ndim == dst_pad.ndim == weight_pad.ndim == valid.ndim == 1):
        raise ValueError("padded edge buffers must be vectors")
    if not (src_pad.numel() == dst_pad.numel() == weight_pad.numel() == valid.numel()):
        raise ValueError("padded edge buffers must have identical capacity")
    src, dst, weight = actuation_markov_edges(graph)
    e = src.numel()
    if e > src_pad.numel():
        raise RuntimeError("compiled edge bucket exhausted; switch buckets at a macro-boundary")
    src_pad.zero_(); dst_pad.zero_(); weight_pad.zero_(); valid.zero_()
    src_pad[:e].copy_(src); dst_pad[:e].copy_(dst); weight_pad[:e].copy_(weight); valid[:e] = True
    return int(e)


def padded_markov_edges_with_slots(
    graph: GraphBuffers,
    max_edges: int | None = None,
    *,
    bucket_size: int = 256,
    reserve_buckets: int = 1,
):
    """Fixed-shape directed Markov buffers including gauge slot/orientation."""
    src, dst, weight, slot, reverse = actuation_markov_edges_with_slots(graph)
    e = src.numel()
    cap = round_edge_capacity(e, bucket_size=bucket_size, reserve_buckets=reserve_buckets) if max_edges is None else int(max_edges)
    if cap < e:
        raise ValueError("max_edges is smaller than active directed edge count")
    src_pad = torch.zeros(cap, dtype=torch.long, device=src.device)
    dst_pad = torch.zeros(cap, dtype=torch.long, device=dst.device)
    w_pad = torch.zeros(cap, dtype=weight.dtype, device=weight.device)
    valid = torch.zeros(cap, dtype=torch.bool, device=weight.device)
    slot_pad = torch.full((cap,), -1, dtype=torch.long, device=src.device)
    reverse_pad = torch.zeros(cap, dtype=torch.bool, device=src.device)
    src_pad[:e] = src; dst_pad[:e] = dst; w_pad[:e] = weight; valid[:e] = True
    slot_pad[:e] = slot; reverse_pad[:e] = reverse
    return src_pad, dst_pad, w_pad, valid, slot_pad, reverse_pad


@torch.no_grad()
def refresh_padded_markov_edges_with_slots_(
    graph: GraphBuffers,
    src_pad: Tensor,
    dst_pad: Tensor,
    weight_pad: Tensor,
    valid: Tensor,
    slot_pad: Tensor,
    reverse_pad: Tensor,
) -> int:
    """Refresh fixed gauge-aware edge buffers without metadata changes."""
    cap = src_pad.numel()
    if not all(x.ndim == 1 and x.numel() == cap for x in (dst_pad, weight_pad, valid, slot_pad, reverse_pad)):
        raise ValueError("all padded edge buffers must be equal-length vectors")
    src, dst, weight, slot, reverse = actuation_markov_edges_with_slots(graph)
    e = src.numel()
    if e > cap:
        raise RuntimeError("compiled edge bucket exhausted; switch buckets at a macro-boundary")
    src_pad.zero_(); dst_pad.zero_(); weight_pad.zero_(); valid.zero_(); slot_pad.fill_(-1); reverse_pad.zero_()
    src_pad[:e].copy_(src); dst_pad[:e].copy_(dst); weight_pad[:e].copy_(weight); valid[:e] = True
    slot_pad[:e].copy_(slot); reverse_pad[:e].copy_(reverse)
    return int(e)

from __future__ import annotations

import inspect
import torch
from torch import nn

from .config import CompileConfig


def compile_if_enabled(module: nn.Module, cfg: CompileConfig) -> nn.Module:
    if not cfg.enabled or not hasattr(torch,"compile"):
        return module
    kwargs={"backend":cfg.backend,"dynamic":cfg.dynamic,"fullgraph":cfg.fullgraph,"mode":cfg.mode}
    try:
        sig=inspect.signature(torch.compile)
        if cfg.isolate_recompiles and "isolate_recompiles" in sig.parameters:
            kwargs["isolate_recompiles"]=True
    except Exception:
        pass
    return torch.compile(module,**kwargs)

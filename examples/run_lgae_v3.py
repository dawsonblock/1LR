from __future__ import annotations

import networkx as nx
import torch

from lgae_v3 import LGAEEngine, LGAEConfig
from lgae_v3.types import make_graph_buffers


def main():
    g=nx.barbell_graph(4,2)
    buffers=make_graph_buffers(g.number_of_nodes(),[(u,v,1.0) for u,v in g.edges()],capacity=32)
    cfg=LGAEConfig()
    cfg.fiber.d_base=8; cfg.fiber.d_max=16; cfg.fiber.spawn_width=2
    cfg.audit.exact_lly_top_k=64  # demo: make integral LLY global on this small graph
    cfg.audit.cde_samples=8
    engine=LGAEEngine(buffers,cfg)
    opt=torch.optim.AdamW(engine.parameters(),lr=1e-3)

    for step in range(6):
        opt.zero_grad(set_to_none=True)
        z=engine()
        target=torch.zeros_like(z)
        loss=(z-target).square().mean()+engine.fibers.regularization()["fiber_loss"]
        loss.backward()
        engine.fiber_controller.update_utility(engine.fibers.latent.grad)
        opt.step()
        engine.diffuse_(eta=0.01)

    ctl=engine.fiber_tick(residual=engine().detach())
    candidate=engine.propose_midpoint_edge()
    result=engine.evaluate_and_maybe_commit(candidate) if candidate else None
    print("mean capacity:",float(ctl["capacity"].float().mean()))
    if result: print("mutation:",result.decision.value,result.reasons)
    print("audit:",engine.audit())

if __name__=="__main__": main()

import torch
from lgae_v3.config import FiberConfig
from lgae_v3.fibers import FixedWidthFiberLatent,FiberController

def test_gates_are_trainable_and_shape_stays_fixed():
    cfg=FiberConfig(d_base=2,d_max=4,spawn_width=1,max_births_per_event=2,score_threshold=-999,persistence_steps=1,gamma_quantile=.5)
    m=FixedWidthFiberLatent(3,cfg)
    assert isinstance(m.gate_logits,torch.nn.Parameter)
    before=m().shape
    ctl=FiberController(m)
    event=ctl.activate(torch.tensor([0]))
    assert m().shape==before
    assert m.capacity[0]==3

def test_candidate_gamma_threshold_is_quantile_not_zero():
    cfg=FiberConfig(d_base=1,d_max=2,score_threshold=-10,persistence_steps=1,gamma_quantile=.5)
    m=FixedWidthFiberLatent(4,cfg); ctl=FiberController(m)
    persistent=ctl.persistent_candidates(torch.ones(4),torch.tensor([0.,0.,1.,2.]))
    assert int(persistent.sum())==2

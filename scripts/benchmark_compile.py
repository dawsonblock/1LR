"""Sparse static/dynamic torch.compile benchmark with unambiguous modes."""
from __future__ import annotations
import argparse, time, torch
from torch import nn
from lgae_v3.compile_utils import compile_if_enabled
from lgae_v3.config import CompileConfig
from lgae_v3.kernels import SparseFieldKernel

class Kernel(nn.Module):
    def __init__(self,n=4096,d=128):
        super().__init__(); self.z=nn.Parameter(torch.randn(n,d)*.01); self.field=SparseFieldKernel()
    def forward(self,src,dst,w):
        z_next,gamma,_,_=self.field(self.z,src,dst,w,0.01)
        return gamma.mean()+1e-4*z_next.square().mean()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["eager","static","auto","dynamic"],default="eager"); ap.add_argument("--n",type=int,default=4096); ap.add_argument("--d",type=int,default=128); ap.add_argument("--e",type=int,default=100000); ap.add_argument("--runs",type=int,default=20); a=ap.parse_args()
    m=Kernel(a.n,a.d)
    src=torch.randint(0,a.n,(a.e,)); dst=torch.randint(0,a.n,(a.e,)); w=torch.rand(a.e)
    mass=torch.zeros(a.n); mass.index_add_(0,src,w); w=w/mass[src].clamp_min(1e-8)
    if a.mode!="eager":
        dyn={"static":False,"auto":None,"dynamic":True}[a.mode]
        m=compile_if_enabled(m,CompileConfig(enabled=True,dynamic=dyn,mode="default"))
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    for _ in range(3): opt.zero_grad(); m(src,dst,w).backward(); opt.step()
    t=time.perf_counter()
    for _ in range(a.runs): opt.zero_grad(); m(src,dst,w).backward(); opt.step()
    print({"mode":a.mode,"ms_per_step":(time.perf_counter()-t)*1000/a.runs})
if __name__=="__main__": main()

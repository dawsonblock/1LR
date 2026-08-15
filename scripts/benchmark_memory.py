"""Compare eager/static/auto/dynamic compile modes without overloading None semantics."""
from __future__ import annotations
import argparse,gc,json,os,time
import torch
from torch import nn

class Core(nn.Module):
    def __init__(self,n,d): super().__init__(); self.z=nn.Parameter(.01*torch.randn(n,d))
    def forward(self,src,dst,w):
        delta=self.z[dst]-self.z[src]; energy=delta.square().sum(-1)
        gamma=torch.zeros(self.z.shape[0],device=self.z.device,dtype=self.z.dtype)
        gamma.index_add_(0,src,w*energy)
        return gamma.mean()+1e-4*self.z.square().mean()

def compile_mode(m,mode):
    if mode=="eager": return m
    dynamic={"static":False,"auto":None,"dynamic":True}[mode]
    kw={"dynamic":dynamic,"mode":"default"}
    import inspect
    if "isolate_recompiles" in inspect.signature(torch.compile).parameters: kw["isolate_recompiles"]=True
    return torch.compile(m,**kw)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["eager","static","auto","dynamic"],required=True); ap.add_argument("--n",type=int,default=4096);ap.add_argument("--d",type=int,default=128);ap.add_argument("--e",type=int,default=100000);ap.add_argument("--runs",type=int,default=50);a=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    m=Core(a.n,a.d).to(device); m=compile_mode(m,a.mode); opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
    src=torch.randint(0,a.n,(a.e,),device=device);dst=torch.randint(0,a.n,(a.e,),device=device);w=torch.rand(a.e,device=device)
    if device=="cuda": torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter(); opt.zero_grad();m(src,dst,w).backward();opt.step();
    if device=="cuda":torch.cuda.synchronize()
    cold=(time.perf_counter()-t0)*1000
    for _ in range(5):opt.zero_grad();m(src,dst,w).backward();opt.step()
    if device=="cuda":torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(a.runs):opt.zero_grad();m(src,dst,w).backward();opt.step()
    if device=="cuda":torch.cuda.synchronize()
    out={"mode":a.mode,"device":device,"cold_ms":cold,"warm_ms":(time.perf_counter()-t)*1000/a.runs}
    if device=="cuda":out.update(peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()

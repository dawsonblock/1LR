from __future__ import annotations

import argparse, json, math
import networkx as nx
import torch

from .config import load_config
from .evolution import LGAEEngine
from .types import make_graph_buffers
from .curvature import crosscheck_lly
from .version import VERSION


def _demo_graph(n: int, capacity_factor: float = 2.0):
    g=nx.barbell_graph(max(2,n//3),max(0,n-2*max(2,n//3))) if n>=6 else nx.path_graph(n)
    edges=[(int(u),int(v),1.0) for u,v in g.edges()]
    return g, make_graph_buffers(g.number_of_nodes(),edges,capacity=max(len(edges)+8,int(len(edges)*capacity_factor)))


def _json_safe(x):
    if isinstance(x,dict): return {str(k):_json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_json_safe(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):
        return "Infinity" if x > 0 else ("-Infinity" if x < 0 else "NaN")
    return x

def _snapshot_dict(s):
    return _json_safe({
        "lambda2":s.lambda2,"operator_discrepancy":s.operator_discrepancy,
        "integral_lly_deficit":s.integral_lly_deficit,"weak_entropic_min":s.weak_entropic_min,
        "bakry_min":s.bakry_min,"cde_residual":s.cde_residual,
        "topology_signature":s.topology_signature,"details":s.details,
    })


def main(argv=None):
    p=argparse.ArgumentParser(prog="lgae-v3")
    p.add_argument("--config",default=None)
    sub=p.add_subparsers(dest="cmd",required=True)
    d=sub.add_parser("demo"); d.add_argument("--nodes",type=int,default=10); d.add_argument("--steps",type=int,default=4)
    q=sub.add_parser("qualify-lly"); q.add_argument("--graph",choices=["path","cycle","complete"],default="cycle"); q.add_argument("--nodes",type=int,default=4)
    args=p.parse_args(argv); cfg=load_config(args.config)
    if args.cmd=="qualify-lly":
        g={"path":nx.path_graph,"cycle":nx.cycle_graph,"complete":nx.complete_graph}[args.graph](args.nodes)
        print(json.dumps(crosscheck_lly(g),indent=2,default=str)); return 0
    g,buffers=_demo_graph(args.nodes)
    eng=LGAEEngine(buffers,cfg)
    for _ in range(args.steps): eng.diffuse_(eta=0.02)
    fiber=eng.fiber_tick()
    mutation=eng.propose_midpoint_edge()
    result=eng.evaluate_and_maybe_commit(mutation) if mutation else None
    out={"version":VERSION,"nodes":buffers.num_nodes,"edges_before":len(g.edges()),"capacity_mean":float(fiber["capacity"].float().mean()),"audit":_snapshot_dict(eng.audit())}
    if result:
        out["mutation"]={"decision":result.decision.value,"reasons":result.reasons,"metadata":result.metadata}
    print(json.dumps(out,indent=2,default=str)); return 0

if __name__=="__main__": raise SystemExit(main())

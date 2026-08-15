from __future__ import annotations
from dataclasses import asdict,is_dataclass
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
from typing import Any


def _safe(x:Any):
    if is_dataclass(x): return _safe(asdict(x))
    if isinstance(x,dict): return {str(k):_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_safe(v) for v in x]
    if hasattr(x,"value") and isinstance(getattr(x,"value"),str): return x.value
    return x


def mutation_receipt(result, *, build_version="3.2.0") -> dict:
    payload={"schema":"LGAE_MUTATION_RECEIPT_V2","build_version":build_version,"created_at":datetime.now(timezone.utc).isoformat(),"result":_safe(result)}
    canonical=json.dumps(payload,sort_keys=True,separators=(",",":"),default=str).encode()
    payload["sha256"]=hashlib.sha256(canonical).hexdigest()
    return payload


def append_receipt(path:str|Path, receipt:dict)->None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf-8") as f: f.write(json.dumps(receipt,sort_keys=True,default=str)+"\n")

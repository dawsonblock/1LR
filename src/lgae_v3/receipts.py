from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
from typing import Any


def _safe(x: Any):
    if is_dataclass(x): return _safe(asdict(x))
    if isinstance(x, dict): return {str(k): _safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_safe(v) for v in x]
    if hasattr(x, "value") and isinstance(getattr(x, "value"), str): return x.value
    return x


def mutation_receipt(
    result,
    *,
    build_version: str = "3.3.0",
    receipt_index: int = 0,
    previous_receipt_hash: str | None = None,
    authority_state_hash_before: str | None = None,
    authority_state_hash_after: str | None = None,
    gauge_authority_hash: str | None = None,
) -> dict:
    """Create a hash-chained mutation receipt.

    The receipt binds the full authority identity: graph state, gauge
    connections, fiber state, and governance config. Each receipt links to
    the previous receipt via ``previous_receipt_hash``, forming a tamper-evident
    hash chain H_i = SHA256(H_{i-1} || R_i).
    """
    payload = {
        "schema": "LGAE_MUTATION_RECEIPT_V3",
        "build_version": build_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "receipt_index": int(receipt_index),
        "previous_receipt_hash": previous_receipt_hash,
        "authority_state_hash_before": authority_state_hash_before,
        "authority_state_hash_after": authority_state_hash_after,
        "gauge_authority_hash": gauge_authority_hash,
        "result": _safe(result),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def append_receipt(path: str | Path, receipt: dict) -> None:
    """Append a receipt to the JSONL ledger, maintaining the hash chain."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # If the receipt doesn't already have chain fields populated, read the
    # last receipt's hash from the file and populate them.
    if receipt.get("previous_receipt_hash") is None and receipt.get("receipt_index", 0) == 0:
        last_hash, last_index = _read_last_receipt_hash(p)
        # If no prior receipts exist, this is the genesis receipt (index 0).
        # Otherwise, chain to the last receipt.
        if last_hash is None:
            receipt["receipt_index"] = 0
            receipt["previous_receipt_hash"] = None
        else:
            receipt["receipt_index"] = last_index + 1
            receipt["previous_receipt_hash"] = last_hash
        # Recompute sha256 with updated chain fields
        chain_keys = {"sha256"}
        payload_for_hash = {k: v for k, v in receipt.items() if k not in chain_keys}
        canonical = json.dumps(payload_for_hash, sort_keys=True, separators=(",", ":"), default=str).encode()
        receipt["sha256"] = hashlib.sha256(canonical).hexdigest()
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(receipt, sort_keys=True, default=str) + "\n")


def _read_last_receipt_hash(path: Path) -> tuple[str | None, int]:
    """Read the hash and index of the last receipt in the ledger."""
    if not path.exists():
        return None, 0
    last_hash: str | None = None
    last_index: int = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                last_hash = r.get("sha256")
                last_index = int(r.get("receipt_index", 0))
            except json.JSONDecodeError:
                continue
    return last_hash, last_index


def verify_receipt_chain(path: str | Path) -> tuple[bool, list[str]]:
    """Verify the integrity of a receipt ledger hash chain.

    Returns (is_valid, errors). Each error describes a broken chain link.
    """
    p = Path(path)
    if not p.exists():
        return True, []
    errors: list[str] = []
    expected_prev: str | None = None
    expected_index: int = 0
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            # Verify chain linkage
            actual_prev = r.get("previous_receipt_hash")
            actual_index = int(r.get("receipt_index", 0))
            if actual_index != expected_index:
                errors.append(f"line {line_no}: receipt_index {actual_index} != expected {expected_index}")
            if actual_prev != expected_prev:
                errors.append(f"line {line_no}: previous_receipt_hash mismatch")
            # Verify self-hash
            stored_hash = r.get("sha256")
            payload_for_hash = {k: v for k, v in r.items() if k != "sha256"}
            canonical = json.dumps(payload_for_hash, sort_keys=True, separators=(",", ":"), default=str).encode()
            computed = hashlib.sha256(canonical).hexdigest()
            if stored_hash != computed:
                errors.append(f"line {line_no}: sha256 mismatch (stored={stored_hash}, computed={computed})")
            expected_prev = stored_hash
            expected_index = actual_index + 1
    return (len(errors) == 0), errors

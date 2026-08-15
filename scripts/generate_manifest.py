#!/usr/bin/env python3
"""Generate a SHA-256 manifest over all tracked repository files.

The manifest is an exact integrity contract: every file in the repository
must either be listed in the manifest or explicitly declared as excluded.
This script enforces that contract and fails if any file is uncovered.

Usage:
    python scripts/generate_manifest.py [--check]
        --check: verify the existing manifest without writing a new one.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from pathlib import Path


# Files that are intentionally excluded from the manifest.
# The manifest itself is always excluded (it would be self-referential).
MANIFEST_EXCLUDES = {
    "MANIFEST.sha256.json",
    ".gitignore",
}

# Directories that are not part of the source contract.
DIR_EXCLUDES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "dist",
    "build",
    "htmlcov",
    ".venv",
    "venv",
    "ENV",
}

# File patterns that are not part of the source contract.
FILE_EXCLUDES = {
    "*.egg-info",  # setuptools build metadata
    "*.pyc",
    "*.pyo",
    "*.so",
}


def should_exclude(path: Path, repo_root: Path) -> bool:
    """Check if a path should be excluded from the manifest."""
    rel = path.relative_to(repo_root)
    parts = rel.parts
    for part in parts:
        for pattern in DIR_EXCLUDES:
            if part == pattern:
                return True
        for pattern in FILE_EXCLUDES:
            if fnmatch.fnmatch(part, pattern):
                return True
    if rel.name in MANIFEST_EXCLUDES:
        return True
    # Exclude common transient files
    if rel.name.startswith(".coverage"):
        return True
    return False


def collect_files(repo_root: Path) -> list[Path]:
    """Collect all files that should be in the manifest."""
    files: list[Path] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if should_exclude(path, repo_root):
            continue
        files.append(path)
    return files


def hash_file(path: Path) -> tuple[int, str]:
    """Return (byte_count, sha256_hex) for a file."""
    h = hashlib.sha256()
    count = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
            count += len(chunk)
    return count, h.hexdigest()


def generate_manifest(repo_root: Path) -> dict:
    """Generate the manifest dictionary."""
    files = collect_files(repo_root)
    entries = []
    for path in files:
        rel = str(path.relative_to(repo_root))
        byte_count, sha = hash_file(path)
        entries.append({"path": rel, "bytes": byte_count, "sha256": sha})
    entries.sort(key=lambda e: e["path"])
    return {
        "schema": "LGAE_V3_MANIFEST_V4",
        "version": "3.3.0",
        "manifest_excludes": sorted(MANIFEST_EXCLUDES),
        "file_count": len(entries),
        "files": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or verify SHA-256 manifest")
    parser.add_argument("--check", action="store_true", help="verify existing manifest without writing")
    parser.add_argument("--root", type=str, default=None, help="repository root (default: cwd)")
    args = parser.parse_args()

    repo_root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    manifest_path = repo_root / "MANIFEST.sha256.json"

    new_manifest = generate_manifest(repo_root)

    if args.check:
        if not manifest_path.exists():
            print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
            return 1
        existing = json.loads(manifest_path.read_text())
        if existing.get("file_count") != new_manifest["file_count"]:
            print(
                f"ERROR: file count mismatch: manifest={existing['file_count']}, actual={new_manifest['file_count']}",
                file=sys.stderr,
            )
            return 1
        # Check each entry
        existing_map = {e["path"]: e for e in existing.get("files", [])}
        new_map = {e["path"]: e for e in new_manifest["files"]}
        missing = set(existing_map) - set(new_map)
        extra = set(new_map) - set(existing_map)
        if missing:
            print(f"ERROR: files in manifest but not on disk: {sorted(missing)}", file=sys.stderr)
            return 1
        if extra:
            print(f"ERROR: files on disk but not in manifest: {sorted(extra)}", file=sys.stderr)
            return 1
        for path, entry in new_map.items():
            old = existing_map[path]
            if old["sha256"] != entry["sha256"]:
                print(f"ERROR: hash mismatch for {path}", file=sys.stderr)
                return 1
            if old["bytes"] != entry["bytes"]:
                print(f"ERROR: byte count mismatch for {path}", file=sys.stderr)
                return 1
        print(f"OK: {new_manifest['file_count']} files verified")
        return 0

    manifest_path.write_text(json.dumps(new_manifest, indent=2, sort_keys=False) + "\n")
    print(f"Manifest written: {new_manifest['file_count']} files, {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

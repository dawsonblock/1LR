"""Canonical version and schema constants for LGAE.

All version-sensitive code (package, CLI, checkpoints, receipts, manifest,
qualification, release report) should import from this module rather than
hardcoding version strings.
"""

VERSION = "4.1.1"
SCHEMA_VERSION = "LGAE_GEOMETRY_V4_1_1"
QUALIFICATION_SCHEMA = "LGAE_QUALIFICATION_V4_1_1"
CHECKPOINT_SCHEMA = "LGAE_V3_CHECKPOINT_V4"
SAFE_CHECKPOINT_SCHEMA = "LGAE_V3_SAFE_CHECKPOINT_V2"
RECEIPT_SCHEMA = "LGAE_V3_RECEIPT_V4"
GRAPH_STATE_SCHEMA = "LGAE_GRAPH_STATE_V4"

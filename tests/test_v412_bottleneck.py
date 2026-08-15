"""v4.1.2 PH bottleneck distance tests."""
from __future__ import annotations

import math
import pytest
import torch
import numpy as np

from lgae_v3.topology import (
    persistent_homology_diagrams,
    bottleneck_distance,
    persistent_homology_bottleneck_drift,
)


def test_bottleneck_distance_identical_diagrams():
    """Bottleneck distance between identical diagrams is 0."""
    dgm = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
    assert bottleneck_distance(dgm, dgm) == 0.0


def test_bottleneck_distance_empty_diagrams():
    """Bottleneck distance between empty diagrams is 0."""
    empty = np.empty((0, 2))
    assert bottleneck_distance(empty, empty) == 0.0


def test_bottleneck_distance_single_point_shift():
    """A shifted point should give bottleneck distance equal to the shift."""
    dgm_a = np.array([[0.0, 1.0]])
    dgm_b = np.array([[0.1, 1.1]])
    # L-infinity distance between (0,1) and (0.1,1.1) is 0.1
    bd = bottleneck_distance(dgm_a, dgm_b)
    assert bd == pytest.approx(0.1, abs=1e-6)


def test_bottleneck_distance_to_diagonal():
    """A point matched to diagonal has distance = (death-birth)/2."""
    dgm_a = np.array([[0.0, 2.0]])  # persistence = 2, diag dist = 1
    dgm_b = np.empty((0, 2))
    bd = bottleneck_distance(dgm_a, dgm_b)
    assert bd == pytest.approx(1.0, abs=1e-6)


def test_bottleneck_distance_symmetric():
    """Bottleneck distance should be symmetric."""
    dgm_a = np.array([[0.0, 1.0], [0.5, 2.0]])
    dgm_b = np.array([[0.1, 1.5], [0.3, 0.8]])
    bd_ab = bottleneck_distance(dgm_a, dgm_b)
    bd_ba = bottleneck_distance(dgm_b, dgm_a)
    assert bd_ab == pytest.approx(bd_ba, abs=1e-6)


def test_bottleneck_distance_strips_infinite():
    """Infinite-death points should be stripped before computation."""
    dgm_a = np.array([[0.0, 1.0], [0.0, float("inf")]])
    dgm_b = np.array([[0.0, 1.0]])
    bd = bottleneck_distance(dgm_a, dgm_b)
    assert bd == 0.0  # only the finite point matters


def test_persistent_homology_bottleneck_drift():
    """Bottleneck drift between two latent clouds should be non-negative."""
    torch.manual_seed(42)
    z_a = torch.randn(10, 3)
    z_b = z_a + 0.5 * torch.randn(10, 3)
    bd = persistent_homology_bottleneck_drift(z_a, z_b)
    if bd is not None:  # ripser may not be available
        assert bd >= 0.0
        assert math.isfinite(bd)


def test_persistent_homology_bottleneck_drift_identical():
    """Bottleneck drift between identical latent clouds should be 0."""
    torch.manual_seed(42)
    z = torch.randn(10, 3)
    bd = persistent_homology_bottleneck_drift(z, z)
    if bd is not None:
        assert bd == pytest.approx(0.0, abs=1e-6)


def test_persistent_homology_diagrams_returns_list():
    """PH diagrams should return a list of arrays."""
    torch.manual_seed(42)
    z = torch.randn(8, 3)
    dgms = persistent_homology_diagrams(z, maxdim=1)
    if dgms is not None:
        assert isinstance(dgms, list)
        assert len(dgms) >= 1
        for d in dgms:
            assert isinstance(d, np.ndarray)
            assert d.shape[1] == 2 if len(d) > 0 else True

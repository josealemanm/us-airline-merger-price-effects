"""Shared fixtures. Everything is seeded, so a failure is reproducible."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def rng():
    return np.random.default_rng(20260101)


@pytest.fixture
def panel(rng):
    """A balanced panel with two-way effects and a known slope."""
    n_unit, n_time = 60, 20
    unit = np.repeat(np.arange(n_unit), n_time)
    time = np.tile(np.arange(n_time), n_unit)
    n = unit.size
    au = rng.normal(0, 2, n_unit)[unit]
    at = rng.normal(0, 1, n_time)[time]
    x1 = rng.normal(0, 1, n) + 0.5 * au
    x2 = rng.normal(0, 1, n)
    y = 1.7 * x1 - 0.9 * x2 + au + at + rng.normal(0, 1, n)
    w = rng.uniform(0.5, 3.0, n)
    return dict(unit=unit, time=time, x1=x1, x2=x2, y=y, w=w,
                n_unit=n_unit, n_time=n_time, beta=(1.7, -0.9))


@pytest.fixture
def staggered(rng):
    """Staggered adoption with an effect that grows after treatment.

    This is the case that breaks two-way fixed effects, which is what the
    Goodman-Bacon tests need.
    """
    T = 24
    cohorts = {8: 40, 14: 40, 19: 40, 10_000: 60}   # last one never treated
    rows, uid = [], 0
    for g, count in cohorts.items():
        for _ in range(count):
            au = rng.normal(0, 1)
            for t in range(T):
                k = t - g
                eff = 0.0 if k < 0 else 0.05 * min(k, 8)
                rows.append((uid, t, g, au + 0.02 * t + eff + rng.normal(0, 0.3)))
            uid += 1
    u, t, g, y = (np.array(a) for a in zip(*rows))
    return dict(unit=u, period=t, gvar=g, y=y, never=10_000)

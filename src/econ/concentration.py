"""Market shares, concentration, and the structural screens in the Guidelines.

These are the first numbers computed in any merger review and the cheapest. A
share table and an HHI need nothing but quantities; they require no demand
estimation, no cost recovery and no assumption about how firms compete. That is
their appeal and also their limit: concentration is a statement about how sales
are distributed today, not about how much a fare would rise tomorrow.

The 2023 Merger Guidelines set out the structural presumption in Section 2.1. A
merger is presumed to substantially lessen competition when it produces a
highly concentrated market - an HHI above 1,800 - and increases the HHI by more
than 100. A second screen catches mergers producing a combined share above 30%
with an HHI increase above 100. This module computes both, so the retrospective
can ask what the screens would have flagged before the fact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def shares_from_quantities(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    total = q.sum()
    return q / total if total > 0 else np.zeros_like(q)


def hhi(shares: np.ndarray) -> float:
    """Herfindahl-Hirschman index on the agency's 0-10,000 scale."""
    s = np.asarray(shares, float)
    return float(np.sum((100.0 * s) ** 2))


def delta_hhi(share_a: float, share_b: float) -> float:
    """Increase in HHI from combining two firms, holding shares fixed.

    The identity is exact: (a+b)^2 - a^2 - b^2 = 2ab. It assumes the merged firm
    simply inherits both shares, which is the convention the Guidelines use and
    is deliberately static - it is a screen, not a prediction.
    """
    return float(2.0 * (100.0 * share_a) * (100.0 * share_b))


def concentration_ratio(shares: np.ndarray, n: int = 4) -> float:
    s = np.sort(np.asarray(shares, float))[::-1]
    return float(s[:n].sum())


def num_effective_competitors(shares: np.ndarray, floor: float = 0.01) -> int:
    """Count of firms holding at least ``floor`` of the market."""
    return int(np.sum(np.asarray(shares, float) >= floor))


@dataclass(frozen=True)
class GuidelinesScreen:
    hhi_pre: float
    hhi_post: float
    delta: float
    combined_share: float
    highly_concentrated: bool
    presumption_hhi: bool          # Section 2.1, first structural screen
    presumption_share: bool        # Section 2.1, second structural screen

    @property
    def flagged(self) -> bool:
        return self.presumption_hhi or self.presumption_share


def screen_merger(shares: np.ndarray, idx_a: int, idx_b: int,
                  hhi_threshold: float = 1800.0, delta_threshold: float = 100.0,
                  share_threshold: float = 0.30) -> GuidelinesScreen:
    """Apply the 2023 Guidelines structural screens to one market."""
    s = np.asarray(shares, float)
    pre = hhi(s)
    post_shares = s.copy()
    post_shares[idx_a] = s[idx_a] + s[idx_b]
    post_shares = np.delete(post_shares, idx_b)
    post = hhi(post_shares)
    d = post - pre
    combined = float(s[idx_a] + s[idx_b])
    concentrated = post > hhi_threshold
    return GuidelinesScreen(
        hhi_pre=pre, hhi_post=post, delta=d, combined_share=combined,
        highly_concentrated=concentrated,
        presumption_hhi=bool(concentrated and d > delta_threshold),
        presumption_share=bool(combined > share_threshold and d > delta_threshold),
    )

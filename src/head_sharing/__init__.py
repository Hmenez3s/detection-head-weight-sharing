"""Inert namespace for controlled detection-head sharing experiments."""

LEGACY_ARMS = ("native", "independent_k2", "shared_k2")
LOOPED_ARMS = ("loop_ab2", "loop_a3")
ALL_ARMS = LEGACY_ARMS + LOOPED_ARMS

__all__ = ["LEGACY_ARMS", "LOOPED_ARMS", "ALL_ARMS"]

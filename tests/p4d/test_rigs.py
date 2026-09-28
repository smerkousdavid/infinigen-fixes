"""Curriculum rig acceptance in an analytic room, without Blender."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from infinigen.p4d import cameras as C
from infinigen.p4d.rigs import sample_rig


def test_rigs_have_measured_tiers_and_independent_paths():
    lo, hi = np.zeros(3), np.array([6., 7., 3.])
    def clearance(p):
        p = np.asarray(p)
        return np.minimum(p - lo, hi - p).min(1)
    def inside(p):
        p = np.asarray(p)
        return ((p > lo + .3) & (p < hi - .3)).all(1)
    def ray(o, d):
        o, d = np.asarray(o), np.asarray(d)
        with np.errstate(divide="ignore", invalid="ignore"):
            exits = np.maximum((lo - o) / d, (hi - o) / d)
        exits[exits <= 0] = np.inf
        return exits.min(1)
    def anchor(rng):
        return rng.uniform([.6, .6, 1.2], [5.4, 6.4, 2.])
    intr = C.Intrinsics.from_blender(24, 36, 320, 180)
    scores = []
    for tier in ("high", "medium", "low"):
        views = sample_rig(np.random.default_rng(104), 24, 24, [intr] * 4, anchor,
                           np.tile([3., 3.5, 1.], (24, 1)), tier, clearance, ray, inside)
        assert len({v["meta"]["path_type"] for v in views}) == 4
        assert sum(v["meta"]["jitter"] for v in views) == 1
        assert all(not v["meta"].get("fallback") for v in views)
        scores.append(views[0]["meta"]["preview_overlap"])
    assert scores[0] >= .65 and .3 <= scores[1] < .55 and scores[2] < .22

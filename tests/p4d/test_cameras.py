"""Unit tests for infinigen.p4d.cameras (numpy only; no Blender needed)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "p4d_cameras", Path(__file__).resolve().parents[2] / "src/infinigen/p4d/cameras.py")
cams = importlib.util.module_from_spec(_spec)
sys.modules["p4d_cameras"] = cams
_spec.loader.exec_module(cams)

INTR = cams.Intrinsics.from_blender(24, 36, 640, 360)
T, FPS = 48, 24.0


def ground_and_wall():
    """ground plane z=0 and a wall x=6 (for clearance / depth rejection tests)."""

    def clearance(p):
        p = np.asarray(p)
        return np.minimum(np.abs(p[:, 2]), np.abs(6.0 - p[:, 0]))

    def ray(o, d):
        o, d = np.asarray(o), np.asarray(d)
        out = np.full(len(o), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            tz = -o[:, 2] / d[:, 2]
            tx = (6.0 - o[:, 0]) / d[:, 0]
        for t in (tz, tx):
            ok = np.isfinite(t) & (t > 0)
            out[ok] = np.minimum(out[ok], t[ok])
        return out

    return clearance, ray


@pytest.mark.parametrize("pt", cams.PATH_TYPES)
def test_path_shapes_and_rotations(pt):
    rng = np.random.default_rng(0)
    tgt = np.stack([np.linspace(0, 2, T), np.zeros(T), np.full(T, 0.5)], 1)
    kw = dict(fps=FPS) if pt in ("handheld", "static") else {}
    pos, R, params = cams.PATHS[pt](rng, T, np.array([-4.0, -3.0, 1.6]), tgt, 5.0, **kw)
    assert pos.shape == (T, 3) and R.shape == (T, 3, 3)
    assert np.allclose(R @ np.swapaxes(R, 1, 2), np.eye(3), atol=1e-9)
    assert np.allclose(np.linalg.det(R), 1, atol=1e-9)
    assert np.isfinite(pos).all()
    if pt in cams.NEEDS_TARGET:  # target projects near the image centre (look-at paths)
        uv, z = cams.project(tgt[:, None], pos, R, INTR)
        assert (z > 0).all()
        tol = 40.0 if pt == "follow" else 1.0  # follow looks at a lagged (smoothed) target
        assert np.abs(uv[..., 0] - INTR.cx).max() < tol and np.abs(uv[..., 1] - INTR.cy).max() < tol


def test_orbit_arc_and_radius():
    rng = np.random.default_rng(1)
    c = np.array([1.0, 2.0, 0.0])
    pos, R, p = cams.path_orbit(rng, T, np.array([5.0, 2.0, 2.0]), c, 4.0, arc_deg=180, direction=1)
    r = np.linalg.norm(pos[:, :2] - c[:2], axis=1)
    assert np.allclose(r.mean(), p["radius"], rtol=0.12)
    a0 = math.atan2(*(pos[0, 1::-1] - c[1::-1]))
    a1 = math.atan2(*(pos[-1, 1::-1] - c[1::-1]))
    assert abs(abs(math.degrees((a1 - a0 + math.pi) % (2 * math.pi) - math.pi)) - 180) < 1.0


def test_tracking_pan_keeps_moving_target_centered_without_translation():
    target = np.stack([np.linspace(-2, 2, T), np.zeros(T), np.ones(T)], -1)
    anchor = np.array([0., -5., 2.])
    pos, R, _ = cams.path_spin(np.random.default_rng(0), T, anchor, target, 5., track_target=True)
    uv, z = cams.project(target[:, None], pos, R, INTR)
    assert np.allclose(pos, anchor)
    assert (z > 0).all()
    assert np.allclose(uv, [INTR.cx, INTR.cy])
    assert not np.allclose(R[0], R[-1])


def test_extrinsics_are_opencv():
    pos = np.array([[0.0, -5.0, 1.0]])
    R = cams.look_rotation(np.array([[0.0, 1.0, 0.0]]))  # looking +y
    E = cams.extrinsics_cv(pos, R)[0]
    pw = np.array([0.0, 0.0, 2.0, 1.0])  # 5 m in front, 1 m above the camera
    pc = E @ pw
    assert np.isclose(pc[2], 5.0) and pc[1] < 0  # z forward, y down (point above -> negative y)
    pr = E @ np.array([1.0, 0.0, 1.0, 1.0])
    assert pr[0] > 0  # x right


def test_jitter_statistics():
    rng = np.random.default_rng(2)
    n = 4000
    pos = np.zeros((n, 3))
    R = np.tile(np.eye(3), (n, 1, 1))
    _, _, prm, dpos, drot = cams.apply_jitter(rng, pos, R, FPS, trans_sigma=0.02, rot_sigma_deg=1.0)
    assert 0.014 < dpos.std() < 0.028
    assert math.radians(0.7) < drot.std() < math.radians(1.4)
    # smooth (OU, not white): lag-1 autocorrelation high
    a = np.corrcoef(dpos[:-1, 0], dpos[1:, 0])[0, 1]
    assert a > 0.8
    assert 0.01 <= prm["trans_sigma_m"] <= 0.04


def test_jitter_fraction():
    rng = np.random.default_rng(3)
    cl, ray = ground_and_wall()
    flags = []
    for i in range(60):
        v = cams.sample_view(rng, T, FPS, INTR, anchor=[-3, 0, 1.5], scale=4.0, target=np.array([0, 0, 0.5]),
                             clearance_fn=cl, ray_fn=ray)
        flags.append(v["meta"]["jitter"])
    assert 0.1 < np.mean(flags) < 0.45


def test_clearance_rejection():
    cl, ray = ground_and_wall()
    pos = np.tile([5.9, 0.0, 1.5], (T, 1))  # 0.1 m from the wall
    R = cams.look_rotation(np.tile([-1.0, 0, 0], (T, 1)))
    ok, rep = cams.validate_path(pos, R, INTR, clearance_fn=cl, ray_fn=ray)
    assert not ok and rep["reason"] == "clearance"


def test_median_depth_rejection():
    cl, ray = ground_and_wall()
    pos = np.tile([5.5, 0.0, 1.5], (T, 1))  # 0.5 m from the wall, looking at it -> median depth 0.5 ok
    R = cams.look_rotation(np.tile([1.0, 0, 0], (T, 1)))
    ok, rep = cams.validate_path(pos, R, INTR, clearance_fn=cl, ray_fn=ray, min_clearance=0.3)
    assert ok, rep
    ok, rep = cams.validate_path(pos, R, INTR, clearance_fn=None, ray_fn=ray, min_median_depth=0.6)
    assert not ok and rep["reason"] == "median_depth"


def test_target_frustum_rejection():
    pos = np.tile([0.0, -5.0, 1.0], (T, 1))
    R = cams.look_rotation(np.tile([0.0, 1.0, 0.0], (T, 1)))
    tgt = np.stack([np.linspace(-1, 30, T), np.zeros(T), np.ones(T)], 1)  # leaves the frame early
    ok, rep = cams.validate_path(pos, R, INTR, target=tgt)
    assert not ok and rep["reason"] == "target_frustum" and rep["target_in_frame_frac"] < 0.8


def test_sample_view_valid_and_fallback():
    cl, ray = ground_and_wall()
    rng = np.random.default_rng(4)
    tgt = np.stack([np.linspace(0, 1.5, T), np.zeros(T), np.full(T, 0.4)], 1)
    v = cams.sample_view(rng, T, FPS, INTR, anchor=[-3, -1, 1.4], scale=3.0, target=tgt, clearance_fn=cl, ray_fn=ray)
    assert v["meta"]["validation"]["min_clearance_m"] >= 0.3
    # impossible: anchor inside the wall clearance band for every path -> static fallback, flagged
    v2 = cams.sample_view(rng, T, FPS, INTR, anchor=[5.95, 0, 1.4], scale=0.01, target=None, clearance_fn=cl,
                          ray_fn=ray, max_tries=4)
    assert v2["meta"]["path_type"] == "static" and v2["meta"].get("fallback")


def test_assign_path_types_distinct():
    rng = np.random.default_rng(5)
    for _ in range(20):
        t = cams.assign_path_types(rng, 4, True)
        assert len(set(t)) == 4
    t = cams.assign_path_types(rng, 4, False)
    assert not set(t) & cams.NEEDS_TARGET


def test_inside_region_rejection():
    pos = np.stack([np.linspace(0.5, 3.5, T), np.full(T, 1.0), np.full(T, 1.5)], 1)  # leaves a 3 m room
    R = cams.look_rotation(np.tile([0.0, 1.0, 0.0], (T, 1)))
    inside = lambda P: np.all((np.asarray(P) >= 0.25) & (np.asarray(P) <= 2.75), axis=-1)  # noqa: E731
    ok, rep = cams.validate_path(pos, R, INTR, inside_fn=inside)
    assert not ok and rep["reason"] == "outside_region" and rep["inside_frac"] < 1


def test_follow_does_not_flip_when_target_stops():
    rng = np.random.default_rng(6)
    x = np.concatenate([np.linspace(0, 2, T // 2), np.full(T - T // 2, 2.0)])  # moves, then stops
    tgt = np.stack([x, np.zeros(T), np.full(T, 0.3)], 1)
    pos, R, p = cams.path_follow(rng, T, np.array([-2.0, -1.0, 1.2]), tgt, 2.0, fps=FPS)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    assert step.max() < 0.25, step.max()  # no jumps (> ~6 m/s)


def test_projection_matrix_shifted_anisotropic():
    P = np.array([[2., 0, .2, 0], [0, 3., -.3, 0], [0, 0, -1.01, -.1], [0, 0, -1., 0]])
    intr = cams.Intrinsics.from_projection(P, 800, 600)
    point_bl = np.array([1., .4, -5., 1.])
    ndc = P @ point_bl
    uv = np.array([(ndc[0] / ndc[3] + 1) * 400, (1 - ndc[1] / ndc[3]) * 300])
    pc = point_bl[:3] * [1, -1, -1]
    expected = intr.K() @ pc
    assert np.allclose(uv, expected[:2] / expected[2])
    assert intr.fx != intr.fy and intr.cx != 400 and intr.cy != 300


def test_strict_invalid_path_raises_without_fallback():
    cl, ray = ground_and_wall()
    with pytest.raises(ValueError, match="no valid static camera"):
        cams.sample_view(np.random.default_rng(3), T, FPS, INTR, anchor=[5.99, 0, 1.5], scale=.01,
                         path_type="static", strict=True, jitter=True, clearance_fn=cl, ray_fn=ray, max_tries=2)


def test_full_rotation_jitter_metadata():
    view = cams.sample_view(np.random.default_rng(8), T, FPS, INTR, [0, -4, 2], 4,
                            target=np.array([0, 0, 1]), path_type="static", jitter=True)
    p = view["meta"]["jitter_params"]
    relative = np.swapaxes(np.asarray(p["pre_jitter_R_cw"]), 1, 2) @ view["R"]
    theta = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
    assert np.isclose(np.degrees(np.sqrt(np.mean(theta**2))), p["applied_rot_rms_deg"])

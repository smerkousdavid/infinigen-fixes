"""Strict acceptance checks for generated pilot clips (separate from codec validity)."""
import argparse
import json
import tarfile
from pathlib import Path
import numpy as np


def check(raw, clip_tar, expected_frames=96, expected_size=(640, 360), expected_fps=24):
    raw = Path(raw)
    scene = json.loads((raw / "scene_meta.json").read_text())
    with tarfile.open(clip_tar) as tar:
        meta_member = next(m for m in tar if m.name.endswith(".meta.json"))
        meta = json.loads(tar.extractfile(meta_member).read())
    quality = meta["quality"]
    failures = []

    def require(ok, message):
        if not ok:
            failures.append(message)

    require(scene.get("frames") == expected_frames, "unexpected pilot frame count")
    require((scene.get("width"), scene.get("height")) == expected_size, "unexpected pilot resolution")
    require(scene.get("fps") == expected_fps, "unexpected pilot frame rate")
    require(meta.get("num_views") == 4, "requires exactly four synchronized views")
    for name in ("rgb", "depth", "normals", "instance", "semantic", "flow", "tracks"):
        require(meta["modalities"].get(name, {}).get("present"), f"missing modality: {name}")
    require(quality.get("validated"), "codec validation failed")
    views = quality.get("views", [])
    require(len(views) == 4 and all((v.get("reprojection_agree_mean") or 0) >= .9 for v in views),
            "per-view reprojection below 0.90")
    # No overlap is valid data, but missing evidence is never reported as a passing agreement score.
    cross = quality.get("cross_view_agree")
    require(cross is None or cross >= .9, "cross-view agreement below 0.90")
    calibration = quality.get("render_calibration", [])
    require(len(calibration) == 4 and all(c.get("position_pass") and
            c["reprojection_p95_max_px"] <= 1.5 for c in calibration), "independent rendered calibration missing/failed")
    require(len(calibration) == 4 and all(c.get("min_median_depth_m", 0) >= .3 for c in calibration),
            "rendered median valid depth below 0.3m")
    paths = scene["views"]
    require(len({v.get("path_type") for v in paths}) == 4, "camera path types are not distinct")
    require(not any(v.get("fallback") for v in paths), "camera fallback is forbidden")
    require(sum(bool(v.get("jitter")) for v in paths) == 1, "requires exactly one jittered camera")
    geometry = scene.get("geometry_checks", {})
    require(geometry.get("frames_checked") == scene["frames"] and
            min(geometry.get("min_per_view_m", [0])) >= .3, "full-clip camera clearance missing/failed")
    for i, view in enumerate(paths):
        camera = np.load(raw / f"view_{i:02d}" / "camera.npz")
        E, K = camera["E_world2cv"], camera["K"]
        R = E[:, :3, :3]
        require(np.isfinite(E).all() and np.isfinite(K).all() and
                np.allclose(R @ np.swapaxes(R, 1, 2), np.eye(3), atol=1e-5) and
                np.allclose(np.linalg.det(R), 1, atol=1e-5), f"invalid camera matrices: view {i}")
        if view.get("jitter"):
            p = view.get("jitter_params") or {}
            if "pre_jitter_R_cw" not in p:
                require(False, f"missing full pre-jitter rotations: view {i}")
                continue
            cw = np.linalg.inv(E)
            trans = np.sqrt(np.mean(np.sum((cw[:, :3, 3] - p["pre_jitter_pos"])**2, axis=1)))
            R_bl = cw[:, :3, :3] @ np.diag([1, -1, -1])
            rel = np.swapaxes(np.asarray(p["pre_jitter_R_cw"]), 1, 2) @ R_bl
            angles = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
            rot = np.sqrt(np.mean(angles**2))
            require(abs(trans - p["applied_trans_rms_m"]) < 1e-3 and
                    abs(rot - p["applied_rot_rms_deg"]) < .02 and trans > .005 and rot > .05,
                    f"rendered jitter differs from authored jitter: view {i}")
    overlap = meta.get("extra_meta", {}).get("camera_overlap", {})
    require(overlap.get("valid_pair_frame_fraction") == 1., "overlap measurements have missing evidence")
    require(overlap.get("achieved") == scene.get("overlap_target"), "achieved overlap differs from requested tier")
    per_kind = quality.get("tracks_per_kind", {})
    family = scene["family"]
    expected = {"nature_creatures": ("creature",), "nature_wind": ("wind", "particle"),
                "indoor_physics": ("rigid",), "indoor_artic": ("articulated",)}.get(family, ())
    for kind in expected:
        q = per_kind.get(kind, {})
        require(q.get("n", 0) >= 16 and q.get("dynamic_fraction", 0) > 0,
                f"missing moving {kind} tracks")
        require(q.get("n_dynamic_visible", 0) >= 16, f"too few visible moving {kind} tracks")
        flow = quality.get("flow_dynamic_per_kind", {}).get(kind, {})
        require(flow.get("n", 0) >= 16 and flow.get("median_epe_px", float("inf")) <= 1.,
                f"missing/failed independent flow evidence for {kind}")
    def visible_motion(name):
        tracks = quality.get("tracks_per_object", {}).get(name, {})
        flow = quality.get("flow_dynamic_per_object", {}).get(name, {})
        return tracks.get("n_dynamic_visible", 0) >= 16 and flow.get("n", 0) >= 16 and flow.get("median_epe_px", float("inf")) <= 1.
    require(not scene.get("tracks", {}).get("report", {}).get("dropped_topology"), "tracked topology changed")
    if family == "nature_creatures":
        gaits = scene["motion"].get("gaits", {})
        require({g.get("gait") for g in gaits.values()} >= {"walk", "run"}, "walk and run gaits not both present")
        for gait in gaits.values():
            contact = gait.get("evaluated_contacts", {})
            require(contact.get("frames_checked") == scene["frames"] and contact.get("stance_samples", 0) > 0
                    and contact.get("stable_steps", 0) > 0
                    and contact.get("endpoint_target_error_p95_m") is not None
                    and contact.get("stance_step_slip_p95_m") is not None
                    and contact["endpoint_target_error_p95_m"] <= .05
                    and contact["stance_step_slip_p95_m"] <= .02
                    and contact.get("terrain_height_error_p95_m") is not None
                    and contact["terrain_height_error_p95_m"] <= .03,
                    f"evaluated {gait.get('gait')} foot contacts failed/missing")
        deformation = scene["motion"].get("vertex_motion", {})
        require(any(v.get("max_local_deform_m", 0) > .02 for v in deformation.values()), "no nonrigid creature gait")
    if family == "indoor_artic":
        joints = [j for a in scene["motion"].get("articulated", []) for j in a.get("joints", [])]
        require({j["kind"] for j in joints if np.ptp(j.get("q", [0])) > .02} >= {"hinge", "slide"},
                "both hinge and sliding motion required")
        for asset in scene["motion"].get("articulated", []):
            require(visible_motion(asset.get("object")), f"articulated asset motion is not visibly verified: {asset.get('object')}")
    if family == "indoor_physics":
        physics = scene["motion"].get("physics", {})
        stats = physics.get("stats", {})
        for motion in ("drops", "slides", "rolls", "chairs"):
            require(any(stats.get(record["object"], {}).get("max_disp_m", 0) > .05
                        and visible_motion(record["object"])
                        for record in physics.get(motion, [])), f"missing visible effective physics motion: {motion}")
        require(physics.get("simulation", {}).get("initial_velocity") == "explicit", "unverified physics impulse")
    result = dict(PASS=not failures, failures=failures, family=family, clip=meta["key"],
                  overlap=overlap, cross_view_evidence=quality.get("cross_view_pairs", 0))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check(args.raw, args.clip)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["PASS"] else 1)

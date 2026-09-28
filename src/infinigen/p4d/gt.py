# Copyright (C) 2026. This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.
"""Multi-view render + ground truth + unified mesh-surface point tracker (p4d raw format v1).

`export_scene(out_dir, cameras, ...)` renders every camera (Cycles; RGB PNG + multilayer EXR with Depth (camera z),
Normal (world), IndexOB, Vector) and writes one world-track bank shared by all views:

  source 0  mesh-surface samples: (triangle, barycentric) fixed on the first-frame evaluated mesh of every
            dynamic object (creature / wind / rigid / articulated / deform); world position + triangle normal
            re-evaluated every frame (constant topology checked per object; objects that change topology are
            dropped and reported).
  source 2  instance samples: GN / particle instances of dynamic instancers (falling leaves, wind-swayed grass
            and tree leaves), keyed by the depsgraph persistent_id, local surface point pushed through the
            instance matrix every frame.
  source 3  interior points (surface=0) inside closed rigid meshes (~interior_frac of rigid samples).
  source 1  static background: rendered depth of static pixels unprojected (several frames of every view).

Per-view visibility is NOT decided here (the p4d converter does depth + instance tests per view).

Raw layout (consumed by point4d-datasets convert/infinigen.py, kind `p4d_multiview`):
  scene_meta.json, objects.json, tracks_raw.npz, view_VV/{rgb_TTTT.png, passes_TTTT.exr, camera.npz}

Infinigen 1.x entry point: gin configurable `p4d_render_image` for `render.render_image_func`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import time
from pathlib import Path

import gin
import numpy as np

logger = logging.getLogger(__name__)

DYNAMIC_KINDS = ("creature", "wind", "rigid", "articulated", "deform", "particle")
CREATURE_RE = re.compile(r"(herbivore|carnivore|flying_bird|bird|fish|beetle|dragonfly|snake|crab|lizard|frog)\(",
                         re.I)


# ------------------------------------------------------------------------------------------ scene helpers
def _collection_parents():
    import bpy

    par = {}
    for c in bpy.data.collections:
        for ch in c.children:
            par.setdefault(ch.name, []).append(c)
    for ch in bpy.context.scene.collection.children:
        par.setdefault(ch.name, []).append(bpy.context.scene.collection)
    return par


def renderable(o, _par=None):
    if o.hide_render:
        return False
    par = _par if _par is not None else _collection_parents()

    def col_ok(c, depth=0):
        if c.hide_render:
            return False
        ps = par.get(c.name, [])
        return depth > 32 or not ps or any(col_ok(p, depth + 1) for p in ps)

    return any(col_ok(c) for c in o.users_collection)


def unhide_renderables():
    """Make every render-visible object/collection also viewport-visible, so the (viewport) evaluated depsgraph
    contains exactly the rendered geometry; sync subdivision/particle display settings to render values."""
    import bpy

    par = _collection_parents()
    for c in bpy.data.collections:
        if not c.hide_render:
            c.hide_viewport = False

    def walk(lc):
        if not lc.collection.hide_render:
            lc.hide_viewport = False
        for ch in lc.children:
            walk(ch)

    walk(bpy.context.view_layer.layer_collection)
    n = 0
    for o in bpy.data.objects:
        if renderable(o, par):
            o.hide_viewport = False
            try:
                o.hide_set(False)
            except RuntimeError:
                pass
            n += 1
        for m in getattr(o, "modifiers", []):
            if m.type == "SUBSURF":
                m.levels = m.render_levels
            elif m.type == "MULTIRES":
                m.levels = m.render_levels
    for ps in bpy.data.particles:
        ps.display_percentage = 100
    bpy.context.scene.render.use_simplify = False
    for mat in bpy.data.materials:  # keep rendered geometry == evaluated mesh (no render-time displacement)
        try:
            mat.cycles.displacement_method = "BUMP"
        except AttributeError:
            pass
    return n


def _chain(o):
    while o is not None:
        yield o
        o = o.parent


def object_kind(o):
    for a in _chain(o):
        k = a.get("p4d_kind") if hasattr(a, "get") else None
        if k:
            return str(k)
    if CREATURE_RE.search(o.name):
        return "creature"
    if any(m.type == "ARMATURE" for m in o.modifiers):
        return "creature" if CREATURE_RE.search(o.name) else "deform"
    for a in _chain(o):
        # passive (non-kinematic) colliders never move: static; actives / animated kinematic ones are rigid
        if a.rigid_body is not None and (a.rigid_body.type == "ACTIVE" or a.rigid_body.kinematic):
            return "rigid"
        if a.animation_data is not None and (a.animation_data.action is not None or len(a.animation_data.drivers)):
            return "rigid"
        if len(a.constraints):
            return "rigid"
    for m in o.modifiers:
        if m.type == "NODES" and m.node_group is not None and m.node_group.animation_data is not None:
            return "articulated" if "joint" in m.node_group.name.lower() else "deform"
    return "static"


def class_hint(o):
    n = o.name
    m = re.match(r"^(?:scatter:)?([A-Za-z_]+?)(?:Factory)?[\(\.:]", n)
    return (o.get("p4d_class") if hasattr(o, "get") and o.get("p4d_class") else (m.group(1) if m else n))


def assign_pass_indices(objects):
    """Unique pass_index 1..N per renderable object (0 = none/sky). -> {pass_index: info}."""
    import bpy
    maximum = bpy.types.Object.bl_rna.properties["pass_index"].hard_max
    if len(objects) > maximum:
        raise ValueError(f"too many objects for unique Blender pass indices: {len(objects)} > {maximum}")
    table = {}
    for i, o in enumerate(objects, start=1):
        o.pass_index = i
        table[int(o.pass_index)] = dict(name=o.name, factory=_factory(o), **{"class": class_hint(o)},
                                        kind=object_kind(o),
                                        actor=next((a.name for a in _chain(o) if a.get("p4d_gait_report")), None))
    return table


def _factory(o):
    for a in _chain(o):
        m = re.search(r"([A-Za-z]+Factory)\(", a.name)
        if m:
            return m.group(1)
        for c in a.users_collection:
            m = re.search(r"([A-Za-z]+Factory)\(", c.name)
            if m:
                return m.group(1)
    return None


# ------------------------------------------------------------------------------------------ mesh evaluation
def eval_mesh(o, dg):
    eo = o.evaluated_get(dg)
    try:
        me = eo.to_mesh()
    except RuntimeError:
        return None, None
    if me is None:
        return None, None
    me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    tri = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    M = np.array(eo.matrix_world)
    eo.to_mesh_clear()
    return co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3], tri.reshape(-1, 3)


def local_mesh(o_eval):
    me = o_eval.to_mesh()
    me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    tri = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    o_eval.to_mesh_clear()
    return co.reshape(-1, 3), tri.reshape(-1, 3)


def tri_areas(co, tri):
    return 0.5 * np.linalg.norm(np.cross(co[tri[:, 1]] - co[tri[:, 0]], co[tri[:, 2]] - co[tri[:, 0]]), axis=1)


def closed_mesh(tri):
    """Every undirected edge must have two oppositely oriented incident faces."""
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    undirected, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    return bool(len(undirected) and np.all(counts == 2) and np.all(np.bincount(inverse, weights=signs) == 0))


def sample_bary(rng, n):
    u, v = rng.random(n), rng.random(n)
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    return np.stack([1 - u - v, u, v], 1)


def eval_points(co, tri, ti, bary):
    p = co[tri[ti]]
    x = (p * bary[:, :, None]).sum(1)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return x, n


# ------------------------------------------------------------------------------------------ render setup
def setup_render(samples=128, device="OPTIX", width=None, height=None):
    import bpy

    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.samples = samples
    s.cycles.use_adaptive_sampling = True
    s.cycles.adaptive_threshold = 0.01
    s.cycles.use_denoising = True
    # Containers expose the host's 96+ cores even when the pod has 9 vCPUs.
    # Bound render/compositor workers to avoid starving GPU synchronization.
    s.render.threads_mode = "FIXED"
    s.render.threads = int(os.environ.get("P4D_RENDER_THREADS", "8"))
    if hasattr(s.cycles, "denoising_use_gpu"):
        s.cycles.denoising_use_gpu = True
    if width:
        s.render.resolution_x, s.render.resolution_y = width, height
    s.render.resolution_percentage = 100
    s.render.use_motion_blur = False
    s.view_settings.view_transform = "AgX"
    s.render.use_persistent_data = True  # keep the synced scene between frames (per-frame sync dominated)
    s.render.film_transparent = False
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev in (device, "CUDA"):
        try:
            prefs.compute_device_type = dev
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type != "CPU"
                s.cycles.device = "GPU"
                break
        except (TypeError, ValueError) as e:
            logger.warning(f"cycles device {dev}: {e}")
    vl = s.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_normal = True
    vl.use_pass_object_index = True
    vl.use_pass_vector = True
    vl.use_pass_position = True
    s.use_nodes = True
    nt = s.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
    fo = nt.nodes.new("CompositorNodeOutputFile")
    fo.format.file_format = "OPEN_EXR_MULTILAYER"
    fo.format.color_depth = "32"
    fo.format.exr_codec = "ZIP"
    fo.file_slots.clear()
    for name in ("Depth", "Normal", "IndexOB", "Vector", "Position"):
        fo.file_slots.new(name)
        nt.links.new(rl.outputs[name], fo.inputs[name])
    # the File Output stores the 4D Vector socket as XYZ only; W (next-frame v) goes through Separate Color
    sep = nt.nodes.new("CompositorNodeSeparateColor")
    sep.mode = "RGB"
    nt.links.new(rl.outputs["Vector"], sep.inputs[0])
    fo.file_slots.new("VectorW")
    nt.links.new(sep.outputs["Alpha"], fo.inputs["VectorW"])
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGB"
    s.render.image_settings.color_depth = "8"
    return fo, s.cycles.device, prefs.compute_device_type


def camera_matrices(cam, frames):
    import bpy
    from infinigen.p4d.cameras import Intrinsics

    s = bpy.context.scene
    Ks, Es = [], []
    for f in frames:
        s.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        Ks.append(Intrinsics.from_camera(cam, s, dg).K())
        M = np.array(cam.evaluated_get(dg).matrix_world) @ np.diag([1.0, -1, -1, 1])
        # Camera scale is not part of a Euclidean world-to-camera pose.
        M[:3, :3] /= np.linalg.norm(M[:3, :3], axis=0)
        if not np.allclose(M[:3, :3].T @ M[:3, :3], np.eye(3), atol=1e-6) or np.linalg.det(M[:3, :3]) < 0:
            raise ValueError("camera transform contains shear or reflection")
        Es.append(np.linalg.inv(M))
    return np.stack(Ks), np.stack(Es)


# ------------------------------------------------------------------------------------------ tracker
class Tracker:
    """Collects the sampled tracks; `step(frame)` evaluates every sample at the current frame."""

    def __init__(self, rng, dyn_objects, instancers, n_mesh=10000, n_inst=3000, pts_per_inst=4,
                 interior_frac=0.05, cameras=None, frames=None):
        import bpy

        self.rng = rng
        self.dg = bpy.context.evaluated_depsgraph_get()
        self.frames = []
        self.report = {"dropped_topology": [], "objects": {}}
        # --- mesh samples (area weighted, sqrt-balanced between objects so small creatures get enough points)
        meshes = []
        for o in dyn_objects:
            co, tri = eval_mesh(o, self.dg)
            if co is None or len(tri) == 0:
                continue
            a = tri_areas(co, tri)
            if a.sum() <= 0:
                continue
            meshes.append((o, co, tri, a))
        w = np.array([math.sqrt(m[3].sum()) for m in meshes]) if meshes else np.zeros(0)
        counts = rng.multinomial(n_mesh, w / w.sum()) if len(w) else []
        self.mesh = []
        for (o, co, tri, a), n in zip(meshes, counts):
            n = int(max(n, 24))
            ti = rng.choice(len(tri), n, p=a / a.sum())
            self.mesh.append(dict(obj=o, nv=len(co), nt=len(tri), topology=tri.copy(), ti=ti,
                                  bary=sample_bary(rng, n), alive=True))
            self.report["objects"][o.name] = dict(n=n, kind=object_kind(o))
        # --- interior samples for closed rigid meshes (surface = 0): inside test by ray parity (+x) in world
        self.interior = []
        n_int = int(interior_frac * n_mesh)
        rig = [m for m in meshes if object_kind(m[0]) == "rigid"]
        if n_int and rig:
            from mathutils import Vector
            from mathutils.bvhtree import BVHTree

            per = max(4, n_int // len(rig))
            for o, co, tri, a in rig:
                if not closed_mesh(tri):
                    continue
                bvh = BVHTree.FromPolygons(co.tolist(), tri.tolist(), all_triangles=True)
                lo, hi = co.min(0), co.max(0)
                pts = []
                for p in rng.uniform(lo, hi, (per * 8, 3)):
                    hits, org = 0, Vector(p)
                    d = Vector((1.0, 0.0137, 0.0071)).normalized()
                    for _ in range(64):
                        loc, _, _, dist = bvh.ray_cast(org, d)
                        if loc is None:
                            break
                        hits += 1
                        org = loc + d * 1e-5
                    if hits % 2 == 1:
                        pts.append(p)
                    if len(pts) >= per:
                        break
                if pts:
                    M0 = np.array(o.evaluated_get(self.dg).matrix_world)
                    loc = (np.array(pts) - M0[:3, 3]) @ np.linalg.inv(M0[:3, :3]).T  # object-local
                    self.interior.append(dict(obj=o, local=loc))
        # --- instance samples (particles / GN instances of dynamic instancers)
        self.inst = []
        self._inst_keys = set()
        if instancers and n_inst > 0:
            names = {o.name for o in instancers}
            candidates, srcmesh = {}, {}
            scene = bpy.context.scene
            initial = scene.frame_current
            # Discover the whole lifetime, not just particles alive at frame zero.
            for frame in frames or [initial]:
                scene.frame_set(frame)
                dg = bpy.context.evaluated_depsgraph_get()
                for ins in dg.object_instances:
                    if not ins.is_instance or ins.parent is None or ins.parent.original.name not in names:
                        continue
                    src = ins.instance_object
                    if src is None or src.type != "MESH":
                        continue
                    srcname = src.original.name
                    key = (ins.parent.original.name, srcname, tuple(ins.persistent_id))
                    if key not in candidates:
                        candidates[key] = srcname
                    if srcname not in srcmesh:
                        co, tri = local_mesh(src)
                        srcmesh[srcname] = co, tri, tri_areas(co, tri)
            keys = sorted(candidates)
            if keys:
                # Dense grass scatters can outnumber falling leaves by 1000:1.
                # Reserve equal budgets for particles and other instances so
                # leaf supervision survives camera-independent sampling.
                strata = {}
                for i, key in enumerate(keys):
                    parent = bpy.data.objects[key[0]]
                    label = "particle" if len(parent.particle_systems) else "other_instance"
                    strata.setdefault(label, []).append(i)
                budget = min(len(keys), max(1, n_inst // pts_per_inst))
                pick = []
                groups = sorted(strata.values(), key=len)
                for j, group in enumerate(groups):
                    count = min(len(group), (budget - len(pick)) // (len(groups) - j))
                    pick.extend(rng.choice(group, count, replace=False).tolist())
                for i in sorted(pick):
                    key = keys[i]
                    srcname = candidates[key]
                    co, tri, a = srcmesh[srcname]
                    if not len(tri) or a.sum() <= 0:
                        continue
                    ti = rng.choice(len(tri), pts_per_inst, p=a / a.sum())
                    self.inst.append(dict(key=key, src=srcname, ti=ti, bary=sample_bary(rng, pts_per_inst),
                                          parent=key[0], topology=tri.copy(), nv=len(co)))
                    self._inst_keys.add(key)
            scene.frame_set(initial)
            self.report["instances"] = dict(candidates=len(keys), tracked=len(self.inst),
                                              lifetime_discovery_frames=len(frames or [initial]))
        self.xyz, self.nrm = [], []

    def step(self):
        import bpy

        dg = bpy.context.evaluated_depsgraph_get()
        X, Nn = [], []
        for m in self.mesh:
            n = len(m["ti"])
            if not m["alive"]:
                X.append(np.full((n, 3), np.nan))
                Nn.append(np.full((n, 3), np.nan))
                continue
            co, tri = eval_mesh(m["obj"], dg)
            if co is None or len(co) != m["nv"] or not np.array_equal(tri, m["topology"]):
                m["alive"] = False
                self.report["dropped_topology"].append(m["obj"].name)
                X.append(np.full((n, 3), np.nan))
                Nn.append(np.full((n, 3), np.nan))
                continue
            x, nn = eval_points(co, tri, m["ti"], m["bary"])
            X.append(x)
            Nn.append(nn)
        for it in self.interior:
            M = np.array(it["obj"].evaluated_get(dg).matrix_world)
            X.append(it["local"] @ M[:3, :3].T + M[:3, 3])
            Nn.append(np.full((len(it["local"]), 3), np.nan))
        if self.inst:
            mats, srcs = {}, {}
            for ins in dg.object_instances:
                if not ins.is_instance or ins.parent is None:
                    continue
                key = (ins.parent.original.name, ins.instance_object.original.name, tuple(ins.persistent_id))
                if key in self._inst_keys:
                    mats[key] = np.array(ins.matrix_world)
            cache = {}
            for it in self.inst:
                n = len(it["ti"])
                M = mats.get(it["key"])
                if M is None:
                    X.append(np.full((n, 3), np.nan))
                    Nn.append(np.full((n, 3), np.nan))
                    continue
                if it["src"] not in cache:
                    cache[it["src"]] = local_mesh(bpy.data.objects[it["src"]].evaluated_get(dg))
                co, tri = cache[it["src"]]
                if len(co) != it["nv"] or not np.array_equal(tri, it["topology"]):
                    raise ValueError(f"instance source changes topology: {it['src']}")
                x, nn = eval_points(co, tri, it["ti"], it["bary"])
                X.append(x @ M[:3, :3].T + M[:3, 3])
                nw = nn @ np.linalg.inv(M[:3, :3])  # normals transform with inverse-transpose
                Nn.append(nw / np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-12))
        self.xyz.append(np.concatenate(X) if X else np.zeros((0, 3)))
        self.nrm.append(np.concatenate(Nn) if Nn else np.zeros((0, 3)))

    def arrays(self):
        pidx, surf, src = [], [], []
        for m in self.mesh:
            n = len(m["ti"])
            pidx.append(np.full(n, m["obj"].pass_index))
            surf.append(np.ones(n, np.int8))
            src.append(np.zeros(n, np.int8))
        for it in self.interior:
            n = len(it["local"])
            pidx.append(np.full(n, it["obj"].pass_index))
            surf.append(np.zeros(n, np.int8))
            src.append(np.full(n, 3, np.int8))
        import bpy

        for it in self.inst:
            n = len(it["ti"])
            # Cycles Object Index belongs to the source object, not the emitter.
            pidx.append(np.full(n, bpy.data.objects[it["src"]].pass_index))
            surf.append(np.ones(n, np.int8))
            src.append(np.full(n, 2, np.int8))
        cat = (lambda L, dt: np.concatenate(L).astype(dt) if L else np.zeros(0, dt))
        return (np.stack(self.xyz).astype(np.float32), np.stack(self.nrm).astype(np.float16),
                cat(pidx, np.int32), cat(surf, np.int8), cat(src, np.int8))


def _exr(path):
    import OpenEXR

    with OpenEXR.File(str(path), separate_channels=True) as f:
        return {k: np.array(v.pixels) for k, v in f.channels().items()}


def static_background(out, views, frames_idx, T, kinds_by_pidx, n_total=6000, rng=None):
    """Unproject rendered depth of static-object pixels (a few frames per view) into constant world tracks."""
    rng = rng or np.random.default_rng(0)
    per = max(1, n_total // (len(views) * len(frames_idx)))
    xyz, nrm, pid = [], [], []
    for v in views:
        cam = np.load(out / f"view_{v:02d}" / "camera.npz")
        for t in frames_idx:
            ch = _exr(out / f"view_{v:02d}" / f"passes_{t + 1:04d}.exr")
            z = ch["Depth.V"].astype(np.float64)
            idx = np.round(ch["IndexOB.V"]).astype(np.int64)
            H, W = z.shape
            ok = np.isfinite(z) & (z > 0) & (z < 1e4) & (idx > 0)
            stat = np.array([kinds_by_pidx.get(int(i), "static") == "static" for i in range(int(idx.max()) + 1)])
            ok &= stat[idx]
            ys, xs = np.nonzero(ok)
            if len(ys) == 0:
                continue
            pick = rng.choice(len(ys), min(per, len(ys)), replace=False)
            ys, xs = ys[pick], xs[pick]
            zz = z[ys, xs]
            K, E = cam["K"][t], cam["E_world2cv"][t]
            pc = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * zz, (ys + 0.5 - K[1, 2]) / K[1, 1] * zz, zz], 1)
            Ecw = np.linalg.inv(E)
            if all(f"Position.{c}" in ch for c in "XYZ"):
                xyz.append(np.stack([ch[f"Position.{c}"] for c in "XYZ"], -1)[ys, xs])
            else:
                xyz.append(pc @ Ecw[:3, :3].T + Ecw[:3, 3])
            n = np.stack([ch["Normal.X"], ch["Normal.Y"], ch["Normal.Z"]], -1)[ys, xs].astype(np.float64)
            nrm.append(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9))
            pid.append(idx[ys, xs])
    if not xyz:
        return np.zeros((T, 0, 3), np.float32), np.zeros((T, 0, 3), np.float16), np.zeros(0, np.int32)
    X, Nn, P = np.concatenate(xyz), np.concatenate(nrm), np.concatenate(pid)
    return (np.repeat(X[None], T, 0).astype(np.float32), np.repeat(Nn[None], T, 0).astype(np.float16),
            P.astype(np.int32))


def _git_commit():
    if os.environ.get("INFINIGEN_FORK_REVISION"):
        return os.environ["INFINIGEN_FORK_REVISION"]
    try:
        root = Path(__file__).resolve().parents[3]
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


UPSTREAM_COMMIT = "3f58bb886bb1bda681d41240344fe3126ac0e9bd"

from infinigen.p4d.runtime import source_identity
SOURCE_IDENTITY = source_identity()


def clearance_report(objects, cameras, frames):
    """Check every evaluated camera frame against static and moving mesh geometry."""
    import bpy
    from infinigen.p4d.cameras import blender_bvh_callbacks

    scene = bpy.context.scene
    scene.frame_set(frames[0])
    static = [o for o in objects if o.type == "MESH" and object_kind(o) == "static"]
    moving = [o for o in objects if o.type == "MESH" and object_kind(o) != "static"]
    static_clear, _, _ = blender_bvh_callbacks(static)
    result = np.full((len(frames), len(cameras)), np.inf)
    for t, frame in enumerate(frames):
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        p = np.array([c.evaluated_get(dg).matrix_world.translation[:] for c in cameras])
        result[t] = static_clear(p)
        if moving:
            dynamic_clear, _, _ = blender_bvh_callbacks(moving, dg)
            result[t] = np.minimum(result[t], dynamic_clear(p))
    if np.any(result < .3):
        t, v = np.unravel_index(np.argmin(result), result.shape)
        raise ValueError(f"camera {v} violates 0.3m clearance at frame {frames[t]}: {result[t, v]:.4f}m")
    return dict(min_per_view_m=result.min(0).tolist(), frames_checked=len(frames),
                geometry="evaluated mesh objects; rendered depth additionally checks visible instances")


def export_scene(out, cameras, family, seed, views_meta=None, motion=None, samples=128, n_mesh=10000,
                 n_inst=3000, n_static=6000, interior_frac=0.05, extra_timing=None, device="OPTIX",
                 instancer_filter=None, vertex_check=True):
    """Render all cameras + write the raw p4d multi-view scene. Returns scene_meta dict."""
    import bpy

    from infinigen.p4d.motion.creatures import vertex_motion_report

    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    s = bpy.context.scene
    fs, fe = s.frame_start, s.frame_end
    T = fe - fs + 1
    frames = list(range(fs, fe + 1))
    rng = np.random.default_rng(seed)
    n_vis = unhide_renderables()
    par = _collection_parents()
    rend = [o for o in s.objects if o.type in ("MESH", "CURVE", "CURVES", "FONT", "META", "VOLUME", "POINTCLOUD")
            and renderable(o, par)]
    table = assign_pass_indices(rend)
    kinds = {pi: d["kind"] for pi, d in table.items()}
    # instancers of dynamic instances: particle emitters + wind-swayed scatters / trees with GN instances
    instancers = []
    for o in rend:
        if o.particle_systems and len(o.particle_systems):
            instancers.append(o)
            table[o.pass_index]["kind"] = "particle"
            kinds[o.pass_index] = "particle"
        elif instancer_filter is not None and instancer_filter(o):
            instancers.append(o)
    s.frame_set(fs)
    # mesh samples on every dynamic mesh except pure particle emitters (wind trees are tracked both as meshes and
    # through their GN instances, e.g. leaves)
    dyn = [o for o in rend if o.type == "MESH" and kinds[o.pass_index] in DYNAMIC_KINDS
           and not (o.particle_systems and len(o.particle_systems))]
    geometry_checks = clearance_report(rend, cameras, frames)
    s.frame_set(fs)
    t1 = time.time()
    tracker = Tracker(rng, dyn, instancers, n_mesh=n_mesh, n_inst=n_inst, interior_frac=interior_frac,
                      cameras=cameras, frames=frames)
    for instance in tracker.inst:
        obj = bpy.data.objects[instance["src"]]
        parent = bpy.data.objects[instance["parent"]]
        if obj.pass_index == 0 or table.get(obj.pass_index, {}).get("name") != obj.name:
            index = max(table, default=0) + 1
            if index > bpy.types.Object.bl_rna.properties["pass_index"].hard_max:
                raise ValueError("instance source exceeds Blender's unique pass-index capacity")
            obj.pass_index = index
            table[obj.pass_index] = dict(name=obj.name, factory=_factory(obj), **{"class": class_hint(obj)})
        kind = kinds[parent.pass_index]
        kinds[obj.pass_index] = kind
        table[obj.pass_index].update(kind=kind, instance_granularity="source_template")
    if tracker.inst:
        tracker.report["instances"]["identity"] = "parent/source/persistent_id; segmentation uses source template"
        tracker.report["instances"]["keys"] = [dict(parent=i["parent"], source=i["src"],
            persistent_id=list(i["key"][2]), points=len(i["ti"])) for i in tracker.inst]
    for f in frames:
        s.frame_set(f)
        tracker.step()
    if tracker.report["dropped_topology"]:
        raise ValueError(f"tracked topology changed: {tracker.report['dropped_topology']}")
    xyz, nrm, pidx, surf, src = tracker.arrays()
    t_track = time.time() - t1
    motion = dict(motion or {})
    motion["gaits"] = {o.name: json.loads(o["p4d_gait_report"]) for o in s.objects if o.get("p4d_gait_report")}
    if motion["gaits"]:
        from infinigen.p4d.motion.creatures import evaluate_contacts, terrain_bvh
        ground = terrain_bvh(render_only=True)
        for gait in motion["gaits"].values():
            gait["evaluated_contacts"] = evaluate_contacts(gait, frames, bvh=ground)
    if vertex_check:
        chk = [o for o in dyn if kinds[o.pass_index] in ("creature", "wind", "articulated", "deform")]
        chk = sorted(chk, key=lambda o: -tracker.report["objects"].get(o.name, {}).get("n", 0))[:60]
        motion["vertex_motion"] = vertex_motion_report(chk, sorted({fs, fs + (fe - fs) // 4, (fs + fe) // 2,
                                                                     fs + 3 * (fe - fs) // 4, fe}))
    # render
    fo, dev, devtype = setup_render(samples=samples, device=device)
    s.cycles.seed = int(seed)
    s.cycles.use_animated_seed = False
    rtimes = []
    for v, cam in enumerate(cameras):
        cam.data.dof.use_dof = False
        vd = out / f"view_{v:02d}"
        vd.mkdir(exist_ok=True)
        K, E = camera_matrices(cam, frames)
        np.savez(vd / "camera.npz", K=K, E_world2cv=E,
                 timestamps_s=np.arange(T, dtype=np.float64) / (s.render.fps / s.render.fps_base))
        s.camera = cam
        fo.base_path = str(vd / "passes_")
        s.render.filepath = str(vd / "rgb_")
        tv = time.time()
        bpy.ops.render.render(animation=True)
        rtimes.append(time.time() - tv)
        logger.info(f"p4d rendered view {v} in {rtimes[-1]:.1f}s")
    # static background from rendered depth
    t2 = time.time()
    fidx = sorted({0, T // 2, T - 1})
    bx, bn, bp = static_background(out, range(len(cameras)), fidx, T, kinds, n_total=n_static, rng=rng)
    xyz = np.concatenate([xyz, bx], 1)
    nrm = np.concatenate([nrm, bn], 1)
    pidx = np.concatenate([pidx, bp])
    surf = np.concatenate([surf, np.ones(len(bp), np.int8)])
    src = np.concatenate([src, np.ones(len(bp), np.int8)])
    np.savez_compressed(out / "tracks_raw.npz", xyz_world=xyz, normal_world=nrm, surface=surf, pass_index=pidx,
                        source=src)
    (out / "objects.json").write_text(json.dumps({str(k): v for k, v in table.items()}, indent=0))
    W, H = (int(s.render.resolution_x * s.render.resolution_percentage / 100),
            int(s.render.resolution_y * s.render.resolution_percentage / 100))
    counts = {k: int((np.array([kinds.get(int(p), "static") for p in pidx]) == k).sum()) for k in
              ("static",) + DYNAMIC_KINDS}
    meta = dict(format="p4d_multiview_raw_v1", family=family, seed=int(seed), frames=T, width=W, height=H,
                fps=float(s.render.fps / s.render.fps_base), infinigen_commit=UPSTREAM_COMMIT,
                fork_commit=SOURCE_IDENTITY.get("fork_commit") or _git_commit(),
                render_source=SOURCE_IDENTITY, world_frame="Blender scene world (Z up, metres)",
                views=[dict(m, view_id=i) for i, m in enumerate(views_meta or [{} for _ in cameras])],
                motion=motion,
                geometry_checks=geometry_checks,
                tracks=dict(n=int(len(pidx)), by_source={str(k): int((src == k).sum()) for k in range(4)},
                            by_kind=counts, report=tracker.report),
                render=dict(samples=samples, device=dev, device_type=devtype, n_renderable_objects=len(rend),
                            n_viewport_unhidden=n_vis, adaptive_threshold=s.cycles.adaptive_threshold,
                            view_transform=s.view_settings.view_transform, exposure=s.view_settings.exposure,
                            cycles_film_exposure=s.cycles.film_exposure, motion_blur=False, depth_of_field=False,
                            cpu_threads=s.render.threads,
                            blender_version=bpy.app.version_string, seed=int(seed)),
                timing=dict(extra_timing or {}, track_sample_eval_s=t_track, render_per_view_s=rtimes,
                            render_per_frame_s=float(np.sum(rtimes) / (T * len(cameras))),
                            static_bg_s=time.time() - t2, export_total_s=time.time() - t0))
    (out / "scene_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    logger.info(f"p4d export done: {meta['tracks']['n']} tracks, {len(cameras)} views, {time.time() - t0:.0f}s")
    return meta


# ------------------------------------------------------------------------------------------ Infinigen 1.x hook
@gin.configurable
def p4d_render_image(frames_folder, camera=None, family="nature", seed=0, out_dir=None, samples=128,
                     wind_strength=0.0, wind_gust=0.5, wind_flutter=0.0, wind_scatters=True, n_mesh=10000,
                     n_inst=3000, n_static=6000, **_):
    """render.render_image_func replacement: renders ALL camera rigs (subcam 0) + unified tracks."""
    import bpy

    from infinigen.core.placement import camera as cam_util
    from infinigen.p4d.motion import wind

    s = bpy.context.scene
    # Saved populated scenes may predate the body-height correction. Reuse the
    # rig and its exact world-space foot targets, then verify evaluated contacts.
    unhide_renderables()
    from infinigen.p4d.motion.creatures import bake_contact_gait, terrain_bvh
    has_gaits = any(o.get("p4d_gait_report") for o in s.objects)
    ground = terrain_bvh(render_only=True) if has_gaits else None
    for obj in list(s.objects):
        if obj.get("p4d_gait_report") and not obj.get("p4d_final_terrain_contacts"):
            gait = json.loads(obj["p4d_gait_report"])
            if obj.get("p4d_contact_original_location") is not None:
                obj.location = obj["p4d_contact_original_location"]
            bake_contact_gait(obj, bpy.data.objects[gait["armature"]],
                              [bpy.data.objects[f["target"]] for f in gait["feet"]], ground)
            obj["p4d_final_terrain_contacts"] = True
        if obj.get("p4d_gait_report"):
            gait = json.loads(obj["p4d_gait_report"])
            contact = gait["evaluated_contacts"]
            if (contact["endpoint_target_error_p95_m"] > .05 or contact["stance_step_slip_p95_m"] > .02
                    or contact["terrain_height_error_p95_m"] is None or contact["terrain_height_error_p95_m"] > .03):
                raise ValueError(f"evaluated foot contacts failed before rendering: {obj.name}: {contact}")
    rigs = cam_util.get_camera_rigs()
    cams = [next(c for c in r.children if c.type == "CAMERA") for r in rigs]
    views = [json.loads(r["p4d_view"]) if "p4d_view" in r else {"path_type": "infinigen_default"} for r in rigs]
    motion = json.loads(s["p4d_motion"]) if "p4d_motion" in s else {}
    tw = time.time()
    wind_objs = set()
    if wind_strength > 0:
        uniq, sources = wind.vegetation_targets(s)
        targets = uniq + (sources if wind_scatters else [])
        motion["wind"] = wind.apply_wind(targets, strength=wind_strength, gust=wind_gust, flutter=wind_flutter,
                                         seed=seed)
        wind_objs = {o.name for o in targets}
    for o in s.objects:  # scatter instancers of wind sources are dynamic
        if o.name.startswith("scatter:") and wind_strength > 0 and wind_scatters and wind.VEGETATION.search(o.name):
            o["p4d_kind"] = "wind"
    out = Path(out_dir) if out_dir else Path(frames_folder) / "p4d"
    tiers = list(dict.fromkeys(v.get("overlap_target", "none") for v in views))
    results = []
    for tier in tiers:
        indices = [i for i, v in enumerate(views) if v.get("overlap_target", "none") == tier]
        destination = out.with_name(out.name + "_" + tier) if tier != "none" else out
        result = export_scene(destination, [cams[i] for i in indices], family, seed,
                              views_meta=[views[i] for i in indices], motion=motion, samples=samples, n_mesh=n_mesh,
                              n_inst=n_inst, n_static=n_static, extra_timing={"wind_setup_s": time.time() - tw},
                              instancer_filter=lambda o: o.get("p4d_kind") == "wind")
        result.update(scene_id=out.name, overlap_target=tier if tier != "none" else None)
        (destination / "scene_meta.json").write_text(json.dumps(result, indent=1, default=str))
        results.append(result)
    return results

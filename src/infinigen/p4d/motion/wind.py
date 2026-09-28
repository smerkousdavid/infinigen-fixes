# Copyright (C) 2026. This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.
"""Wind on vegetation with constant topology (geometry nodes, time from the Scene Time node).

A shared node group `p4d_wind` is appended as the LAST modifier of every vegetation object (and of the source
objects of vegetation scatters, so every instance of a grass tuft / fern sways). It only moves things:

  mesh points      Set Position(offset)
  GN instances     Translate Instances(offset at the instance position) + small flutter rotation (leaves)

    h      = clamp((P.z - z_min) / height, 0, 1) ** 1.5         (base fixed, tip moves most)
    phase  = 2 pi f t + phi_obj + k . P_xy                      (travelling wave across the object)
    sway   = sin(phase) + 0.35 sin(2.3 phase + 1.7)
    gust   = 2 * (noise4(P * 0.05, t * 0.25 + phi_obj) - 0.5)   (shared low-frequency wind field)
    offset = dir * strength * height * h * (sway + gust_amp * gust)  +  flutter * height * h * (noise3(P*6, 2t) - .5)

`dir` is the scene wind direction expressed in each object's local frame, so all objects lean the same way.
No vertices are added/removed, so evaluated meshes keep their topology and surface tracks stay valid.
"""

from __future__ import annotations

import logging
import math
import re

import numpy as np

logger = logging.getLogger(__name__)

VEGETATION = re.compile(
    r"(Tree|Bush|Grass|Fern|Monocot|Flower|Plant|Leaf|Leaves|Branch|Kelp|Coral|Cactus|Mushroom|Twig|Shrub|Ivy|"
    r"Pine|Palm|Agave|Banana|Veratrum|Reed|Wheat)", re.I)
NOT_WIND = re.compile(r"(chopped_tree|Pinecone|Rock|Boulder|Stone|Log|stump|Terrain|placeholder)", re.I)
# cacti and mushrooms are stiff: much smaller amplitude
STIFF = re.compile(r"(Cactus|Mushroom|Coral)", re.I)

GROUP_NAME = "p4d_wind"


def _new_socket(ng, name, in_out, stype, default=None):
    s = ng.interface.new_socket(name, in_out=in_out, socket_type=stype)
    if default is not None and hasattr(s, "default_value"):
        s.default_value = default
    return s


def build_group():
    import bpy

    if GROUP_NAME in bpy.data.node_groups:
        return bpy.data.node_groups[GROUP_NAME]
    ng = bpy.data.node_groups.new(GROUP_NAME, "GeometryNodeTree")
    _new_socket(ng, "Geometry", "INPUT", "NodeSocketGeometry")
    _new_socket(ng, "Geometry", "OUTPUT", "NodeSocketGeometry")
    for nm, d in (("Strength", 0.04), ("ZMin", 0.0), ("Height", 1.0), ("Phase", 0.0), ("Freq", 0.4),
                  ("Gust", 0.5), ("Flutter", 0.0), ("Wave", 0.3)):
        _new_socket(ng, nm, "INPUT", "NodeSocketFloat", d)
    _new_socket(ng, "Direction", "INPUT", "NodeSocketVector", (1.0, 0.0, 0.0))
    N, L = ng.nodes, ng.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    self_object = N.new("GeometryNodeSelfObject")
    object_info = N.new("GeometryNodeObjectInfo")
    object_info.transform_space = "ORIGINAL"
    L.new(self_object.outputs["Self Object"], object_info.inputs["Object"])

    def math_(op, a=None, b=None, c=None):
        n = N.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = float(v)
            else:
                L.new(v, n.inputs[i])
        return n.outputs[0]

    def vmath(op, a=None, b=None, scale=None):
        n = N.new("ShaderNodeVectorMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (tuple, list)):
                n.inputs[i].default_value = v
            else:
                L.new(v, n.inputs[i])
        if scale is not None:
            if isinstance(scale, (int, float)):
                n.inputs["Scale"].default_value = scale
            else:
                L.new(scale, n.inputs["Scale"])
        return n.outputs["Vector"] if op != "DOT_PRODUCT" else n.outputs["Value"]

    def field(domain_pos):
        """offset field (vector) given a position output socket."""
        sep = N.new("ShaderNodeSeparateXYZ")
        L.new(domain_pos, sep.inputs[0])
        h = math_("DIVIDE", math_("SUBTRACT", sep.outputs["Z"], gi.outputs["ZMin"]), gi.outputs["Height"])
        h = math_("POWER", math_("MINIMUM", math_("MAXIMUM", h, 0.0), 1.0), 1.5)
        t = N.new("GeometryNodeInputSceneTime").outputs["Seconds"]
        rotated = N.new("ShaderNodeVectorRotate")
        rotated.rotation_type = "EULER_XYZ"
        L.new(vmath("MULTIPLY", domain_pos, object_info.outputs["Scale"]), rotated.inputs["Vector"])
        L.new(object_info.outputs["Rotation"], rotated.inputs["Rotation"])
        world_pos = vmath("ADD", rotated.outputs["Vector"], object_info.outputs["Location"])
        world_sep = N.new("ShaderNodeSeparateXYZ")
        L.new(world_pos, world_sep.inputs[0])
        wave = math_("MULTIPLY", math_("ADD", world_sep.outputs["X"], world_sep.outputs["Y"]), gi.outputs["Wave"])
        ph = math_("ADD", math_("ADD", math_("MULTIPLY", math_("MULTIPLY", t, gi.outputs["Freq"]), 2 * math.pi),
                                    gi.outputs["Phase"]), wave)
        sway = math_("ADD", math_("SINE", ph), math_("MULTIPLY", math_("SINE", math_("ADD", math_("MULTIPLY", ph, 2.3),
                                                                                         1.7)), 0.35))
        nz = N.new("ShaderNodeTexNoise")
        nz.noise_dimensions = "4D"
        nz.inputs["Scale"].default_value = 1.0
        L.new(vmath("SCALE", world_pos, scale=0.05), nz.inputs["Vector"])
        L.new(math_("MULTIPLY", t, 0.25), nz.inputs["W"])
        gust = math_("MULTIPLY", math_("SUBTRACT", nz.outputs["Fac"], 0.5), 2.0)
        amp = math_("MULTIPLY", math_("MULTIPLY", math_("MULTIPLY", gi.outputs["Strength"], gi.outputs["Height"]), h),
                    math_("ADD", sway, math_("MULTIPLY", gust, gi.outputs["Gust"])))
        main = vmath("SCALE", gi.outputs["Direction"], scale=amp)
        nf = N.new("ShaderNodeTexNoise")
        nf.noise_dimensions = "4D"
        nf.inputs["Scale"].default_value = 6.0
        L.new(domain_pos, nf.inputs["Vector"])
        L.new(math_("MULTIPLY", t, 2.0), nf.inputs["W"])
        fl = vmath("SUBTRACT", nf.outputs["Color"], (0.5, 0.5, 0.5))
        fl_amt = math_("MULTIPLY", math_("MULTIPLY", gi.outputs["Flutter"], gi.outputs["Height"]), h)
        return vmath("ADD", main, vmath("SCALE", fl, scale=fl_amt)), fl, fl_amt

    pos = N.new("GeometryNodeInputPosition").outputs[0]
    off_pts, _, _ = field(pos)
    setp = N.new("GeometryNodeSetPosition")
    L.new(gi.outputs["Geometry"], setp.inputs["Geometry"])
    L.new(off_pts, setp.inputs["Offset"])
    # instances: evaluate the same field at the instance position (Position on the instance domain)
    ipos = N.new("GeometryNodeInputPosition").outputs[0]
    off_inst, fl_i, fl_amt_i = field(ipos)
    tr = N.new("GeometryNodeTranslateInstances")
    L.new(setp.outputs["Geometry"], tr.inputs["Instances"])
    L.new(off_inst, tr.inputs["Translation"])
    tr.inputs["Local Space"].default_value = False
    rot = N.new("GeometryNodeRotateInstances")
    L.new(tr.outputs["Instances"], rot.inputs["Instances"])
    # flutter rotation (rad) ~ flutter noise * 4 (only when Flutter > 0)
    L.new(vmath("SCALE", fl_i, scale=math_("MULTIPLY", gi.outputs["Flutter"], 4.0)), rot.inputs["Rotation"])
    rot.inputs["Local Space"].default_value = True
    L.new(rot.outputs["Instances"], go.inputs["Geometry"])
    return ng


def _local_bbox_z(obj):
    import bpy

    dg = bpy.context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(dg)
    bb = np.array([list(v) for v in eo.bound_box])
    return float(bb[:, 2].min()), float(max(bb[:, 2].max() - bb[:, 2].min(), 1e-3))


def is_vegetation(obj):
    n = obj.name
    for c in obj.users_collection:
        n += " " + c.name
    return bool(VEGETATION.search(n)) and not NOT_WIND.search(obj.name)


def apply_wind(objects, strength=0.04, gust=0.5, flutter=0.0, freq=0.4, direction_deg=None, seed=0):
    """Append the wind modifier to each object. Returns {object name: params}."""
    from mathutils import Vector

    rng = np.random.default_rng(seed)
    ng = build_group()
    yaw = math.radians(direction_deg if direction_deg is not None else rng.uniform(0, 360))
    wdir = Vector((math.cos(yaw), math.sin(yaw), 0.0))
    done = {}
    for o in objects:
        if o.type != "MESH" or any(m.name == "p4d_wind" for m in o.modifiers):
            continue
        z0, h = _local_bbox_z(o)
        transform = o.matrix_world.to_3x3()
        d = transform.inverted() @ wdir
        k = 0.25 if STIFF.search(o.name) else 1.0
        prm = dict(Strength=strength * k * transform.col[2].length, ZMin=z0, Height=h, Phase=float(rng.uniform(0, 2 * math.pi)),
                   Freq=freq * float(rng.uniform(0.8, 1.25)), Gust=gust, Flutter=flutter * k, Wave=0.3)
        m = o.modifiers.new("p4d_wind", "NODES")
        m.node_group = ng
        for item in ng.interface.items_tree:
            if item.in_out != "INPUT" or item.socket_type == "NodeSocketGeometry":
                continue
            if item.name == "Direction":
                m[item.identifier] = (d.x, d.y, d.z)
            elif item.name in prm:
                m[item.identifier] = prm[item.name]
        o["p4d_kind"] = "wind"
        done[o.name] = dict(prm, direction=(d.x, d.y, d.z))
    logger.info(f"p4d wind on {len(done)} objects (strength={strength}, gust={gust}, flutter={flutter})")
    return dict(direction_deg=math.degrees(yaw), field_frame="world", n_objects=len(done), objects=list(done)[:200],
                strength=strength, gust=gust, flutter=flutter)


def vegetation_targets(scene=None):
    """(unique vegetation mesh objects, scatter source objects of vegetation scatters)."""
    import bpy

    scene = scene or bpy.context.scene
    uniq = [o for o in scene.objects if o.type == "MESH" and is_vegetation(o) and not o.name.startswith("scatter:")]
    sources = set()
    for col in bpy.data.collections:
        if col.name.startswith("assets:") and VEGETATION.search(col.name) and not NOT_WIND.search(col.name):
            for o in col.all_objects:
                if o.type == "MESH":
                    sources.add(o)
    return uniq, sorted(sources, key=lambda o: o.name)

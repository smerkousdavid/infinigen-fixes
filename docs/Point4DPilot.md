# Point4D motion and camera pilot

The pilot configuration is `scripts/p4d/configs/pilot.json`: four underlying scenes,
each reused with high, medium and low camera overlap. Each of the twelve clips has
96 frames at 24 fps, four synchronized 640×360 views and 128 Cycles samples.
The 30-scene manifest is preparation for a later run; it is not launched by the pilot.

```bash
python -m infinigen.p4d.checks --out /root/out/calibration_checks
python -m infinigen.p4d.check_motion --out /root/out/motion_checks.json
python -m infinigen.p4d.batch --plan scripts/p4d/configs/pilot.txt \
  --out /root/out/pilot_v02 --jobs 1 --split pilot_v02
```

Run on a Linux x86 GPU host installed with `scripts/p4d/install_pod.sh`.
The converter checkout defaults to `/root/point4d-datasets`.
`--resume` requires matching recorded configuration and generator code. Indoor
`--reuse-room PATH` reuses the saved, unanimated room with the same seed while
rebuilding motion and cameras; the new run records that source path.
`--reuse-prepared PATH` reuses a compatible indoor scene's baked motion and rigs;
`--reuse-coarse PATH` reuses a successful nature coarse stage. Both record source
configuration and artifact checksums. `batch --collect-only` converts and validates
completed raw clips without running generation again.

When deploying source without Git metadata, run `scripts/p4d/stamp_source.py` from
a clean checkout and copy the generated `source_manifest.json` with the package.
Each render records the verified source checksum and commit independently of the
parent generation process. Set `P4D_CPU_THREADS` and `P4D_RENDER_THREADS` to share
the container's actual CPU allocation between concurrent scenes.

## Geometry and motion

Camera intrinsics come from Blender's evaluated projection, including sensor fit,
shift, pixel aspect and resolution scaling. Extrinsics are evaluated world-to-camera
matrices in OpenCV axes. Exported depth is camera-z in metres, derived from the
rendered world Position pass; its independent reprojection is checked before encoding.
Motion blur, depth of field and lens distortion are disabled for the sharp GT pilot.

Furniture joint limits follow linked geometry-node expressions. Every joint records
its full trajectory, units, limits, profile and velocity cap. Placement measures
rendered faces rather than loose simulation metadata and checks swept bounds on every
frame. Room generation currently admits cabinet and drawer; doors require authored
wall openings and countertop appliances require support-aware placement before they
can be added to this pool.

Indoor impulses use PyBullet 3.2.7 with explicit initial linear/angular velocities,
gravity and 20 simulation substeps per frame. Its poses are baked into Blender for
both rendering and tracking. Chairs use collision-checked external actuation.
Blender's native kinematic handoff was tested and reset the intended launch velocity.

Creature feet use distance-phased walk/run contact targets, rebound to the final
rendered terrain. Body height is fitted against evaluated IK endpoints. Acceptance
checks stance foot height, foot slip, nonrigid mesh motion, and visible tracks and
flow separately for the walking and running actors. Wind uses a shared world-space
gust field on unique geometry; repeated instances use a source-template field.
Particle tracks use parent/source/persistent IDs discovered
over the full clip. Cycles Object Index labels repeated instances by source template,
so particle segmentation is at source-template granularity; tracks retain separate
temporal identities. Particle samples receive a separate quota so dense vegetation
cannot consume their entire track budget. Changing tracked connectivity fails
before rendering, and unique segmentation IDs must fit Blender's pass-index range.

## Overlap curriculum

Each rig has four distinct path types and exactly one jittered camera. Coarse BVH
overlap is only a proposal filter. The converter measures rendered surface coverage,
track and dynamic-track Jaccard, baseline, normalized baseline, optical-axis angle and
triangulation angle per frame and camera pair. These are stored in the optional
`extra.camera_overlap.npz` member; summaries also enter the clip metadata and index.

The measured median surface overlap labels a clip high (≥0.60), medium (≥0.25) or
low (<0.25). Zero overlap is valid; missing evidence is NaN. A requested tier is
never substituted for the measured label. Low-overlap views can look at different
actions/regions of the same scene. Keep all variants sharing `scene_id` in the same
train/validation split.

The reader's `select_camera_pairs` accepts numeric overlap and normalized-baseline
filters. Training chooses epoch boundaries; dataset generation imposes no schedule.

## Acceptance and storage

Codec checks are followed by `infinigen.p4d.qa`: required modalities, all-view
calibration and clearance, camera path/jitter checks, measured overlap, visible motion
and per-motion-type flow evidence. Motion-specific checks require both hinge/slide,
walk/run, wind/particles, or drop/slide/roll/chair as appropriate. Rendered previews
must also be reviewed; passing the codec alone is insufficient.

Validated compressed clips and SHA256 manifests remain in `ready/`. Raw files stay
on the generation host until a transfer has been independently verified. Store the
pilot under the new `pilot_v02` split and preserve existing `smoke` shards. Full runs
must record timing, actual spend, source revisions and final pod state.

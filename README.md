
<div align="center">
<img src="docs/images/infinigen.png" width="300"></img>
</div>

# [Infinigen: Infinite Photorealistic Worlds Using Procedural Generation](https://infinigen.org)

[**Getting Started**](#getting-started)
| [**Website**](https://infinigen.org/)
| [**Intro Video**](https://www.youtube.com/watch?v=6tgspeI-GHY)
| [**Papers**](#papers)
| [**Documentation**](#documentation)
| [**Contributing**](#contributing)

<div align="center">
<a href="https://youtu.be/6tgspeI-GHY"> <img src="docs/images/infinigen_teaser.png" width="100%"></a>
</div>

## infinigen-fixes (branch `p4d-fixes`): bug fixes + realistic-motion multi-view 4D ground truth

This fork is based on upstream `princeton-vl/infinigen` at `3f58bb8` (v2.0.0a2), with upstream history kept.
It adds a few bug fixes, plus an opt-in package `infinigen.p4d` that produces multi-view video with 4D point
tracks for point-tracking and 4D-reconstruction training. Upstream files change only where a fix requires it.

### Fixes (one commit each)
1. **customgt glad sources.** `customgt/dependencies/glad/src/glad.c` and `glad_egl.c` were dropped by the `*.c`
   `.gitignore` rule, so `INFINIGEN_INSTALL_CUSTOMGT=True` failed ("Cannot find source file ... glad.c").
   The fix restores them from v1.19.0 and adds `.gitignore` exceptions.
2. **SnakeFactory crash.** `reptile_postprocessing()` and `chameleon_postprocessing()` had the wrong arity for
   `join_and_rig_parts(postprocess_func(root))`, raising `TypeError` for every snake (e.g. `desert.gin`).
3. **Animated hairy creatures.** Herbivore and carnivore raised `NotImplementedError` whenever hair and animation
   were both set, and `compose_nature` always animates them. The fix drops the hair with a warning instead.
4. **SLURM probe hang.** Container images ship `sinfo` without a cluster, and `sinfo()` retried forever, stalling
   `manage_jobs` after coarse. The fix adds a cached `slurm_available()` probe with a timeout, bounded retries, and
   an `INFINIGEN_DISABLE_SLURM=1` switch.

### Features (`src/infinigen/p4d/`)

See [the Point4D pilot guide](docs/Point4DPilot.md) for calibrated cameras, explicit
physics impulses, measured overlap curriculum, acceptance gates and pilot commands.
- **`cameras.py`: 4 synchronized cameras per scene.** Each view gets an independent path type: orbit, spin, crane,
  dolly, follow-object, handheld or near-static.
  - About 25 % of views get Ornstein-Uhlenbeck jitter on translation (1-4 cm) and rotation (0.3-1.5°), plus a
    small shake.
  - Validity checks: clearance ≥ 0.3 m, target in frustum ≥ 80 % of frames, median depth ≥ 0.3 m, the camera
    stays inside the room, and a sky-fraction limit.
  - Unit tests: `tests/p4d/test_cameras.py`.
- **`motion/creatures.py`: walking creatures.** Upstream `populate_all` re-created creature factories without the
  coarse-stage `animation_mode`, so creatures were never rigged. With this, herbivores and carnivores use the
  distance-phased walk/run gaits on terrain paths. Evaluated foot contacts and nonrigid vertex motion are checked.
- **`motion/wind.py`: wind on vegetation.** A geometry-nodes wind (height-weighted sway, a shared gust field and
  leaf flutter) on trees, bushes, grass and scatter sources. Topology stays constant.
- **`motion/objects.py`: indoor rigid-body motion.** Bullet rigid-body drops, tosses, slides and rolling balls,
  plus keyframed pushed chairs.
- **`motion/articulation.py`: articulated Infinigen-Sim assets in v2 rooms.** Cabinets and drawers
  replace static storage against walls, and every joint is animated with an independent
  profile: open, close, partial, open-close or repeated.
- **`gt.py`: multi-view render and point tracks.** V-view Cycles render: RGB, z-depth, normals, object index and
  Vector flow. It also runs a unified mesh-surface point tracker that produces one world-track bank shared by
  all views:
  - (triangle, barycentric) samples on constant-topology evaluated meshes;
  - instance samples for particles, such as falling leaves;
  - about 5 % interior points;
  - static background points.

  The output is the raw `p4d_multiview_raw_v1` format, converted to the p4d-1.1 schema by the point4d-datasets
  pipeline (`convert/infinigen.py`).
- **`run_scene.py` and `batch.py`: scene drivers.** Families: `nature_creatures`, `nature_wind`,
  `indoor_physics`, `indoor_artic` and `indoor_flying`. `nature_driver.py` wraps `generate_nature` unchanged.

```bash
bash scripts/p4d/install_pod.sh        # fork install on a Linux x86 GPU box (no patches needed)
python -m infinigen.p4d.run_scene --family indoor_physics --seed 0 --out outputs/p4d/phys0
python -m infinigen.p4d.batch --plan scripts/p4d/configs/pilot.txt --out outputs/p4d/pilot --jobs 1
```

## Getting Started

### Getting Started with Infinigen V2

<p align="center">
  <img src="docs/images/hello_v2/brick_sphere_0.png" width="13%" />
  <img src="docs/images/hello_v2/brick_sphere_2.png" width="13%" />
  <img src="docs/images/hello_v2/sofa_2.png" width="13%" />
  <img src="docs/images/hello_v2/sofa_4.png" width="13%" />
  <img src="docs/images/hello_v2/livingroom_rgb.png" width="13%" />
  <img src="docs/images/hello_v2/livingroom_depth.png" width="13%" />
</p>

See getting-started instructions for Infinigen-V2 in our [hosted documentation](https://infinigen.cs.princeton.edu/docs/latest/index.html)

### Generate Articulated Sim Assets: Getting Started with Infinigen Articulated

<p align="center">
  <img src="docs/images/infinigen_articulated.gif" width="80%" />
</p>

See Installation and Exporting-to-Simulators instructions on our [articulated-stable](https://github.com/princeton-vl/infinigen/blob/articulated-stable/README.md) version, or on [initial release](https://github.com/princeton-vl/infinigen/blob/articulated-initial/README.md) or [latest](docs/simulation/ExportingToSimulators.md)

### Hello Room: Getting Started with Infinigen Indoors

<p align="center">
  <img src="docs/images/hello_room/dining_blender.png" width="19.5%" />
  <img src="docs/images/hello_room/dining.png" width="19.5%" />
  <img src="docs/images/hello_room/dining_depth.png" width="19.5%" />
  <img src="docs/images/hello_room/dining_obj.png" width="19.5%" />
</p>

See Installation and Hello-Room instructions on our [indoors-stable](https://github.com/princeton-vl/infinigen/blob/indoors-stable/README.md) version, or on [initial release](https://github.com/princeton-vl/infinigen/blob/indoors-initial/README.md) or [latest](docs/HelloRoom.md)

### Hello World: Getting Started with Infinigen Nature

<p align="center">
  <img src="docs/images/hello_world/Image0048_00_00.png" width="19.5%" />
  <img src="docs/images/hello_world/Depth0048_00_00.png" width="19.5%" />
  <img src="docs/images/hello_world/SurfaceNormal_0001_00_00.png" width="19.5%" />
  <img src="docs/images/hello_world/InstanceSegmentation_0001_00_00.png" width="19.5%" />
</p>

See Installation and Hello-World instructions on our [nature-stable](https://github.com/princeton-vl/infinigen/blob/nature-stable/README.md) version, or on [initial release](https://github.com/princeton-vl/infinigen/blob/nature-initial/README.md) or [latest](docs/HelloWorld.md)

## Papers

If you use Infinigen in your work, please cite our academic papers:

<h3 align="center"><a href="https://arxiv.org/pdf/2306.09310">Infinite Photorealistic Worlds using Procedural Generation</a></h3>
<p align="center">
<a href="http://araistrick.com/">Alexander Raistrick</a>*, 
<a href="https://www.lahavlipson.com/">Lahav Lipson</a>*, 
<a href="https://mazeyu.github.io/">Zeyu Ma</a>* (*equal contribution, alphabetical order) <br>
<a href="https://www.cs.princeton.edu/~lm5483/">Lingjie Mei</a>, 
<a href="https://www.cs.princeton.edu/~mingzhew">Mingzhe Wang</a>, 
<a href="https://zuoym15.github.io/">Yiming Zuo</a>, 
<a href="https://kkayan.com/">Karhan Kayan</a>, 
<a href="https://hermera.github.io/">Hongyu Wen</a>, 
<a href="https://pvl.cs.princeton.edu/people.html">Beining Han</a>, <br>
<a href="https://pvl.cs.princeton.edu/people.html">Yihan Wang</a>, 
<a href="http://www-personal.umich.edu/~alnewell/index.html">Alejandro Newell</a>, 
<a href="https://heilaw.github.io/">Hei Law</a>, 
<a href="https://imankgoyal.github.io/">Ankit Goyal</a>, 
<a href="https://yangky11.github.io/">Kaiyu Yang</a>, 
<a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
Conference on Computer Vision and Pattern Recognition (CVPR) 2023
</p>

</p>

```
@inproceedings{infinigen2023infinite,
  title={Infinite Photorealistic Worlds Using Procedural Generation},
  author={Raistrick, Alexander and Lipson, Lahav and Ma, Zeyu and Mei, Lingjie and Wang, Mingzhe and Zuo, Yiming and Kayan, Karhan and Wen, Hongyu and Han, Beining and Wang, Yihan and Newell, Alejandro and Law, Hei and Goyal, Ankit and Yang, Kaiyu and Deng, Jia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12630--12641},
  year={2023}
}
```

<h3 align="center">
  <a href="https://arxiv.org/abs/2406.11824">Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation</a>
</h3>
<p align="center">
  <a href="http://araistrick.com/">Alexander Raistrick</a>*, 
  <a href="https://www.cs.princeton.edu/~lm5483/">Lingjie Mei</a>*, 
  <a href="https://kkayan.com/">Karhan Kayan</a>*, (*equal contribution, random order) <br>
  <a href="https://david-yan1.github.io/">David Yan</a>, 
  <a href="https://zuoym15.github.io/">Yiming Zuo</a>, 
  <a href="https://pvl.cs.princeton.edu/people.html">Beining Han</a>, 
  <a href="https://hermera.github.io/">Hongyu Wen</a>, 
  <a href="https://scholar.google.com/citations?user=q38OfTQAAAAJ&hl=en">Meenal Parakh</a>, <br>
  <a href="https://stamatisalex.github.io/">Stamatis Alexandropoulos</a>, 
  <a href="https://www.lahavlipson.com/">Lahav Lipson</a>, 
  <a href="https://mazeyu.github.io/">Zeyu Ma</a>,
  <a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
  Conference on Computer Vision and Pattern Recognition (CVPR) 2024
</p>

```
@inproceedings{infinigen2024indoors,
    author    = {Raistrick, Alexander and Mei, Lingjie and Kayan, Karhan and Yan, David and Zuo, Yiming and Han, Beining and Wen, Hongyu and Parakh, Meenal and Alexandropoulos, Stamatis and Lipson, Lahav and Ma, Zeyu and Deng, Jia},
    title     = {Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {21783-21794}
}
```

<h3 align="center">
  <a href="https://arxiv.org/abs/2505.10755">Infinigen-Articulated: Procedural Generation of Articulated Simulation Assets</a>
</h3>
<p align="center">
  <a href="https://abhihjoshi.github.io/"> Abhishek Joshi</a>,
  <a href="https://beininghan.github.io/">Beining Han</a>,
  <a href="https://pvl.cs.princeton.edu/people.html">Jack Nugent</a>,
  <a href="https://pvl.cs.princeton.edu/people.html">Max Gonzalez Saez-Diez</a>,
  <br>
  <a href="https://zuoym15.github.io">Yiming Zuo</a>,
  <a href="https://pvl.cs.princeton.edu/people.html">Jonathan Liu</a>,
  <a href="https://hermera.github.io">Hongyu Wen</a>,
  <a href=https://stamatisalex.github.io/>Stamatis Alexandropoulos</a>,
  <a href="https://kkayan.com/">Karhan Kayan</a>,
  <a href="https://pvl.cs.princeton.edu/people.html">Anna Calveri</a>,
  <br>
  <a href="https://pvl.cs.princeton.edu/people.html">Tao Sun</a>,
  <a href="https://pvl.cs.princeton.edu/people.html">Gaowen Liu</a>,
  <a href="https://www.mcgill.ca/civil/yi-shao">Yi Shao</a>,
  <a href="http://araistrick.com">Alexander Raistrick</a>,
  <a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a>
  <br>
</p>

```
@misc{joshi2025articulated,
      title={Procedural Generation of Articulated Simulation-Ready Assets}, 
      author={Abhishek Joshi and Beining Han and Jack Nugent and Max Gonzalez Saez-Diez and Yiming Zuo and Jonathan Liu and Hongyu Wen and Stamatis Alexandropoulos and Karhan Kayan and Anna Calveri and Tao Sun and Gaowen Liu and Yi Shao and Alexander Raistrick and Jia Deng},
      year={2025},
      eprint={2505.10755},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.10755}, 
}
```

<h3 align="center">
  <a href="https://arxiv.org/abs/2604.26943">ProcFunc: Function-Oriented Abstractions for Procedural 3D Generation in Python</a>
</h3>
<p align="center">
  <a href="http://araistrick.com/">Alexander Raistrick</a>, 
  <a href="https://kkayan.com/">Karhan Kayan</a>, 
  <a href="https://pvl.cs.princeton.edu/people.html">Jack Nugent</a>, 
  <a href="https://david-yan1.github.io/">David Yan</a>, 
  <a href="https://www.cs.princeton.edu/~lm5483/">Lingjie Mei</a>, 
  <a href="https://scholar.google.com/citations?user=q38OfTQAAAAJ&hl=en">Meenal Parakh</a>, <br>
  <a href="https://hermera.github.io/">Hongyu Wen</a>, 
  <a href="https://pvl.cs.princeton.edu/people.html">Dylan Li</a>, 
  <a href="https://zuoym15.github.io/">Yiming Zuo</a>, 
  <a href="https://pvl.cs.princeton.edu/people.html">Erich Liang</a>, 
  <a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
  arXiv 2026
</p>

```
@misc{raistrick2026procfunc,
      title={ProcFunc: Function-Oriented Abstractions for Procedural 3D Generation in Python}, 
      author={Alexander Raistrick and Karhan Kayan and Jack Nugent and David Yan and Lingjie Mei and Meenal Parakh and Hongyu Wen and Dylan Li and Yiming Zuo and Erich Liang and Jia Deng},
      year={2026},
      eprint={2604.26943},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.26943}, 
}
```

## Documentation

- [Installation Guide](docs/Installation.md)
- ["Hello World": Generate your first Infinigen-Nature scene](docs/HelloWorld.md)
- ["Hello Room": Generate your first Infinigen-Indoors scene](docs/HelloRoom.md)
- [Configuring Infinigen](docs/ConfiguringInfinigen.md)
- [Configuring Cameras](docs/ConfiguringCameras.md)
- [Downloading pre-generated data](docs/PreGeneratedData.md)
- [Generating individual assets](docs/GeneratingIndividualAssets.md)
- [Exporting to external fileformats (OBJ, OpenUSD, etc)](docs/ExportingToExternalFileFormats.md)
- [Add external assets to indoor scenes](docs/StaticAssets.md)
- [Extended ground-truth](docs/GroundTruthAnnotations.md)
- [Implementing new materials & assets](docs/ImplementingAssets.md)
- [Generating fluid simulations](docs/GeneratingFluidSimulations.md)
- [Exporting to physics simulation](docs/ExportingToSimulators.md)

Please see our [project roadmap](https://infinigen.org/roadmap) and follow us at [https://twitter.com/PrincetonVL](https://twitter.com/PrincetonVL) for updates. 

## Contributing

We welcome contributions! You can contribute in many ways:
- **Contribute code to this repository** - We welcome code contributions. More guidelines coming soon.
- **Contribute procedural generators** - `infinigen/nodes/node_transpiler/dev_script.py` provides tools to convert artist-friendly [Blender Nodes](https://docs.blender.org/manual/en/2.79/render/blender_render/materials/nodes/introduction.html) into python code. Tutorials and guidelines coming soon.
- **Contribute pre-generated data** - Anyone can contribute their computing power to create data and share it with the community. Please stay tuned for a repository of pre-generated data.

### Getting Help

Please post this repository's Github Issues page for help. Please run your command with `--debug`, and let us know:
- What is your computing setup, including OS version, CPU, RAM, GPU(s) and any drivers?
- What version of the code are you using (link a commit hash), and what if any modifications have you made (new configs, code edits)
- What exact command did you run?
- What were the output logs of the command you ran? 
    - If using `manage_jobs`, look in `outputs/MYJOB/MYSEED/logs/` to find the right one.
    - What was the exact python error and stacktrace, if applicable?

### Acknowledgements

Infinigen wouldn't be possible without the fantastic work of the [Blender Foundation](https://www.blender.org/) and it's open-source contributors. Infinigen uses many open source projects, with special thanks to [Land-Lab](https://github.com/landlab/landlab), [BlenderProc](https://github.com/DLR-RM/BlenderProc) [Blender-FLIP-Fluids](https://github.com/rlguy/Blender-FLIP-Fluids) and [Blender-Differential-Growth](https://github.com/inca/blender-differential-growth).

We thank [Thomas Kole](https://blenderartists.org/u/ThomasKole) for providing procedural clouds.

We learned tremendously from online tutorials of 
[Andrew Price](https://www.youtube.com/channel/UCOKHwx1VCdgnxwbjyb9Iu1g),
[Artisans of Vaul](https://www.youtube.com/@ArtisansofVaul),
[Bad Normals](https://www.youtube.com/@BadNormals),
[Blender Tutorial Channel](https://www.youtube.com/@BlenderTutorialChannel),
[blenderbitesize](https://www.youtube.com/@blenderbitesize),
[Blendini](http://www.youtube.com/watch?v=sHr8LjfX09c),
[Bradley Animation](https://www.youtube.com/@bradleyanimation120),
[CGCookie](https://www.youtube.com/watch?v=lPAYX8z9i8M),
[CGRogue](https://www.youtube.com/@PixelicaCG),
[Creative Shrimp](https://www.youtube.com/@CreativeShrimp),
[CrowdRender](https://www.youtube.com/@Crowdrender),
[Dr. Blender](https://www.youtube.com/@DrBlender),
[HEY Pictures](https://www.youtube.com/channel/UCo5rv1z-PPrCh-C7OvO2VAA),
[Ian Hubert](https://www.youtube.com/@IanHubert2),
[Kev Binge](https://www.youtube.com/@KevBinge),
[Lance Phan](https://www.youtube.com/@LancePhan),
[MaxEdge](https://www.youtube.com/@MaxEdge420),
[Mr. Cheebs](https://www.youtube.com/@MrCheebs),
[PixelicaCG](https://www.youtube.com/@PixelicaCG),
[Polyfjord](https://www.youtube.com/@Polyfjord),
[Robbie Tilton](https://www.youtube.com/@RobbieTilton),
[Ryan King Art](https://www.youtube.com/@RyanKingArt),
[Sam Bowman](https://www.youtube.com/@snow_mamba) and
[yogigraphics](https://www.youtube.com/@yojigraphics).
These tutorials provided procedural generators for our early experimentation and served as inspiration for our own implementations in the official release of Infinigen. They are acknowledged in file header comments where applicable. 

Infinigen has evolved significantly since the version described in our CVPR paper. It now features some procedural code obtained from the internet under CC-0 licenses, which are marked with code comments where applicable - no such code was present in the system for the CVPR version.

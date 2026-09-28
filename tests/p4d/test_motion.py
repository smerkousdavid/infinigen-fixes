import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("p4d_motion", Path(__file__).resolve().parents[2] /
                                             "src/infinigen/p4d/motion/creatures.py")
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)


def test_rigid_turn_is_not_deformation():
    points = np.random.default_rng(9).normal(size=(400, 3))
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    assert motion.rigid_residual(points, points @ rotation + [3, 5, -2]).max() < 1e-10
    transformed = points @ rotation + [3, 5, -2]
    transformed[0] += [0, 0, .4]
    assert motion.rigid_residual(points, transformed).max() > .35


def test_missing_terrain_does_not_move_into_void():
    def terrain(x, y):
        return (0., [0, 0, 1]) if x < .2 else None
    pos, _, _ = motion.walk_path(np.random.default_rng(1), [0, 0, 0], 96, 24, terrain, heading0=0)
    assert np.max(pos[:, 0]) < .2

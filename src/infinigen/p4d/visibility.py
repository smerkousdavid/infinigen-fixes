"""Exact batching of static placeholder visibility across the complete camera union."""
import numpy as np


def patch_population_visibility():
    from infinigen.core.placement import placement, split_in_view
    original = placement.filter_populate_targets
    if getattr(original, "_p4d_batched", False):
        return

    def animated(obj):
        while obj is not None:
            if obj.animation_data is not None or obj.constraints:
                return True
            obj = obj.parent
        return False

    def filter_targets(placeholders, cameras, dist_cull, vis_cull, verbose):
        if not placeholders or any(animated(p) for p in placeholders):
            return original(placeholders, cameras, dist_cull, vis_cull, verbose)
        # The original traverses every camera and every frame separately for
        # every placeholder. Joining static vertices gives identical tests with
        # one traversal; no time samples or cameras are dropped.
        points = [placement.get_placeholder_points(p).reshape(-1, 3) for p in placeholders]
        ends = np.cumsum([len(p) for p in points])
        mask, distances, visible_distances = split_in_view.compute_inview_distances(
            np.concatenate(points), cameras, dist_max=dist_cull, vis_margin=vis_cull, verbose=verbose)
        result, start = [], 0
        for obj, end in zip(placeholders, ends):
            if mask[start:end].any():
                result.append((obj, distances[start:end].min(), visible_distances[start:end].min()))
            start = end
        return result

    filter_targets._p4d_batched = True
    placement.filter_populate_targets = filter_targets

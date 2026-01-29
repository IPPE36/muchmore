from __future__ import annotations

from typing import List

import numpy as np
from skimage.measure import marching_cubes
import pyvista as pv


def periodic_iso_surfaces(field, min_area=5, snap_tol=0) -> List[pv.PolyData]:

    tiled = np.tile(field, (3, 3, 3))

    # extract iso-surface on tiled domain
    verts_zyx, faces, _, _ = marching_cubes(tiled.astype(np.float32), level=0.5)
    # marching_cubes outputs vertices in (z, y, x) index coordinates by default
    # reorder to (x, y, z) to match your 2D pixel-space convention
    verts_xyz = verts_zyx[:, [2, 1, 0]].astype(np.float64)

    # PyVista mesh, face format: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    surf = pv.PolyData(verts_xyz, faces_pv)

    # central tile bounds in pixel space:
    ny, nx, nz = field.shape
    x0, x1 = nx, 2 * nx
    y0, y1 = ny, 2 * ny
    z0, z1 = nz, 2 * nz

    # clip to axis-aligned box (robust triangle clipping via VTK)
    clipped = surf.clip_box(bounds=(x0, x1, y0, y1, z0, z1), invert=False)
    if clipped.n_cells == 0:
        return []

    # Optional snap: push vertices within snap_tol to exact box planes
    if snap_tol and snap_tol > 0:
        pts = clipped.points.copy()
        # snap to x planes
        pts[np.abs(pts[:, 0] - x0) <= snap_tol, 0] = x0
        pts[np.abs(pts[:, 0] - x1) <= snap_tol, 0] = x1
        # snap to y planes
        pts[np.abs(pts[:, 1] - y0) <= snap_tol, 1] = y0
        pts[np.abs(pts[:, 1] - y1) <= snap_tol, 1] = y1
        # snap to z planes
        pts[np.abs(pts[:, 2] - z0) <= snap_tol, 2] = z0
        pts[np.abs(pts[:, 2] - z1) <= snap_tol, 2] = z1
        clipped.points = pts

        # Clean coincident points / degenerate triangles that snapping can introduce
        clipped = clipped.clean(tolerance=0.0)

    # Split into connected components and filter by surface area
    comps = clipped.connectivity(extraction_mode="all")
    out: List[pv.PolyData] = []
    for i in range(comps.n_blocks):
        block = comps[i]
        if block is None or block.n_cells == 0:
            continue
        area = float(block.area)  # surface area
        if area >= min_area:
            out.append(block)

    return out


def mesh_3d(field: np.ndarray):
    polys_center = periodic_iso_surfaces(field)
    return


if __name__ == "__main__":
    exit()

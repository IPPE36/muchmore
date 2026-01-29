from __future__ import annotations

from typing import List

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union, snap
from skimage.measure import marching_cubes


def periodic_iso_surfaces(field, min_area=5, snap_tol=0) -> List[Polygon]:

    tiled = np.tile(field, (3, 3, 3))

    # extract iso-surface on tiled domain
    Vzyx, F, _, _ = marching_cubes(tiled.astype(np.float32), level=0.5)
    # marching_cubes outputs vertices in (z, y, x) index coordinates by default
    # reorder to (x, y, z) to match your 2D pixel-space convention
    V = Vzyx[:, [2, 1, 0]].astype(np.float64)

    # central tile bounds in pixel space:
    ny, nx, nz = field.shape
    x0, x1 = nx, 2 * nx
    y0, y1 = ny, 2 * ny
    z0, z1 = nz, 2 * nz

    return []


def mesh_3d(field: np.ndarray):
    polys_center = periodic_iso_surfaces(field)
    return


if __name__ == "__main__":
    exit()

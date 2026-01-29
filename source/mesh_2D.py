from typing import List

import gmsh
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from skimage.measure import find_contours

from source.plotting import plot_polygons, plot_contours


def periodic_contours(field, min_area=5) -> List[Polygon]:

    tiled = np.tile(field, (3, 3))
    contours = find_contours(tiled, level=0.5)
    plot_contours(tiled, contours, "contours.png")

    # central tile bounds in pixel space:
    ny, nx = field.shape
    x0, x1 = nx, 2 * nx
    y0, y1 = ny, 2 * ny
    center_tile = box(x0, y0, x1, y1)

    polys = []
    for c in contours:
        if c.shape[0] < 10:
            continue

        poly = Polygon(np.c_[c[:, 1], c[:, 0]])
        if (not poly.is_valid) or poly.area <= 0:
            continue

        # check intersection with center
        inter_area = poly.intersection(center_tile).area
        if inter_area < min_area:
            continue

        wrapped = Polygon(np.array(poly.exterior.coords))
        if wrapped.area >= min_area:
            polys.append(wrapped)

    # union overlapping polygons
    if polys:
        merged = unary_union(polys)
        if merged.geom_type == "Polygon":
            return [merged]
        elif merged.geom_type == "MultiPolygon":
            return list(merged.geoms)

    return []


def polygon_to_loop(gmsh_model, poly: Polygon, lc: float):
    """
    Adds a shapely polygon to gmsh as a curve loop.
    Returns curveLoopTag.
    """
    pts = list(poly.exterior.coords)
    # remove last point because shapely repeats start at end
    pts = pts[:-1]

    p_tags = []
    for x, y in pts:
        p_tags.append(gmsh_model.occ.addPoint(float(x), float(y), 0.0, lc))

    # create lines
    l_tags = []
    n = len(p_tags)
    for i in range(n):
        a = p_tags[i]
        b = p_tags[(i + 1) % n]
        l_tags.append(gmsh_model.occ.addLine(a, b))

    loop = gmsh_model.occ.addCurveLoop(l_tags)
    return loop


def mesh_microstructure_2d(field, lc=0.01):
    ny, nx = field.shape

    polys = periodic_contours(field)
    plot_polygons(field, polys, "poly.png")

    gmsh.initialize()
    gmsh.model.add("micro_2d")

    occ = gmsh.model.occ

    # Outer domain rectangle
    rect = occ.addRectangle(0, 0, 0, nx, ny)

    # Inclusion surfaces
    inc_surfs = []
    for poly in polys:
        # Skip tiny or invalid
        if poly.is_empty or not poly.is_valid or poly.area <= 0:
            continue
        loop = polygon_to_loop(gmsh.model, poly, lc=lc)
        surf = occ.addPlaneSurface([loop])
        inc_surfs.append(surf)

    occ.synchronize()

    # Boolean cut matrix = rect \ inclusions
    # If you also want inclusion surfaces kept, use removeObject=False
    if inc_surfs:
        cut = occ.cut([(2, rect)], [(2, s) for s in inc_surfs],
                      removeObject=True, removeTool=False)
        occ.synchronize()
        matrix_entities = cut[0]  # list of (dim, tag)
    else:
        matrix_entities = [(2, rect)]

    # Physical groups (nice for FEM)
    # inclusions:
    if inc_surfs:
        gmsh.model.addPhysicalGroup(2, inc_surfs, tag=1)
        gmsh.model.setPhysicalName(2, 1, "inclusions")
    # matrix:
    matrix_tags = [tag for (dim, tag) in matrix_entities if dim == 2]
    gmsh.model.addPhysicalGroup(2, matrix_tags, tag=2)
    gmsh.model.setPhysicalName(2, 2, "matrix")

    # --- Periodic boundary constraints ---
    # Identify boundary curves on rectangle by bounding boxes
    # (this is robust even after boolean ops)
    eps = 1e-9
    left = gmsh.model.getEntitiesInBoundingBox(-eps, -eps, -eps,  eps, ny+eps, eps, 1)
    right = gmsh.model.getEntitiesInBoundingBox(nx-eps, -eps, -eps, ny+eps, ny+eps, eps, 1)
    bot = gmsh.model.getEntitiesInBoundingBox(-eps, -eps, -eps, nx+eps,  eps, eps, 1)
    top = gmsh.model.getEntitiesInBoundingBox(-eps, ny-eps, -eps, nx+eps, ny+eps, eps, 1)

    left_tags = [c[1] for c in left]
    right_tags = [c[1] for c in right]
    bot_tags = [c[1] for c in bot]
    top_tags = [c[1] for c in top]

    # Pair right (slave) to left (master) via translation (-Lx, 0)
    gmsh.model.mesh.setPeriodic(1, right_tags, left_tags,
                                [1, 0, 0, -nx,
                                 0, 1, 0,  0,
                                 0, 0, 1,  0,
                                 0, 0, 0,  1])

    # Pair top (slave) to bottom (master) via translation (0, -Ly)
    gmsh.model.mesh.setPeriodic(1, top_tags, bot_tags,
                                [1, 0, 0,  0,
                                 0, 1, 0, -ny,
                                 0, 0, 1,  0,
                                 0, 0, 0,  1])

    # Mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    gmsh.model.mesh.generate(2)

    gmsh.finalize()
    return


if __name__ == "__main__":
    exit()

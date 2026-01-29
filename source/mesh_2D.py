from typing import List

import gmsh
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union, snap
from shapely.affinity import translate
from skimage.measure import find_contours


def periodic_contours(field, min_area=5, snap_tol=0) -> List[Polygon]:
    """
    Tile microstructure to consider inclusions outside and ensure consistent polygons for periodic meshing
    """

    tiled = np.tile(field, (3, 3))
    contours = find_contours(tiled, level=0.5)

    # central tile bounds in pixel space:
    ny, nx = field.shape
    x0, x1 = nx, 2 * nx
    y0, y1 = ny, 2 * ny
    rve = box(x0, y0, x1, y1)
    rve_bnd = rve.boundary

    polys = []
    for c in contours:
        if c.shape[0] < 10:
            continue

        poly = Polygon(np.c_[c[:, 1], c[:, 0]])
        if (not poly.is_valid) or poly.area <= 0:
            continue

        # fast reject: intersects (includes boundary) instead of area check
        if not poly.intersects(rve):
            continue

        clipped = poly.intersection(rve)
        if clipped.is_empty:
            continue

        # clean & snap to exact RVE edges
        clipped = clipped.buffer(0)
        clipped = snap(clipped, rve_bnd, snap_tol)

        if clipped.area >= min_area:
            if clipped.geom_type == "Polygon":
                polys.append(clipped)
            elif clipped.geom_type == "MultiPolygon":
                polys.extend(list(clipped.geoms))

    if polys:
        merged = unary_union(polys).buffer(0)
        if merged.geom_type == "Polygon":
            return [merged]
        elif merged.geom_type == "MultiPolygon":
            return list(merged.geoms)

    return []


def polygon_to_loop(gmsh_model, poly: Polygon):
    """
    Adds a shapely polygon to gmsh as a curve loop.
    Returns curveLoopTag.
    """
    pts = list(poly.exterior.coords)
    pts = pts[:-1]

    p_tags = []
    for x, y in pts:
        p_tags.append(gmsh_model.occ.addPoint(float(x), float(y), 0.0))

    # create lines
    l_tags = []
    n = len(p_tags)
    for i in range(n):
        a = p_tags[i]
        b = p_tags[(i + 1) % n]
        l_tags.append(gmsh_model.occ.addLine(a, b))

    loop = gmsh_model.occ.addCurveLoop(l_tags)
    return loop


def mesh_2d(field: np.ndarray):

    polygons = periodic_contours(field)
    ny, nx = field.shape

    # Shift polygons into origin tile
    polys = []
    for poly in polygons:
        p = translate(poly, xoff=-nx, yoff=-ny)
        p = p.simplify(0.5, preserve_topology=True)
        polys.append(p)

    gmsh.initialize()
    gmsh.model.add("ms2d")
    occ = gmsh.model.occ

    # RVE at origin
    rect = occ.addRectangle(0, 0, 0, nx, ny)

    # Inclusion surfaces (now also in origin coordinates)
    surfs = []
    for poly in polys:
        if poly.is_empty or (not poly.is_valid) or poly.area <= 0:
            continue
        loop = polygon_to_loop(gmsh.model, poly)
        surf = occ.addPlaneSurface([loop])
        surfs.append(surf)
    occ.synchronize()

    # Boolean fragment (conforming interfaces)
    outDimTags, outMap = occ.fragment([(2, rect)], [(2, s) for s in surfs])
    occ.synchronize()

    matrix_surfs = sorted({tag for (dim, tag) in outMap[0] if dim == 2})
    inclusion_surfs = sorted({
        tag
        for m in outMap[1:]
        for (dim, tag) in m
        if dim == 2
    })

    gmsh.model.addPhysicalGroup(2, inclusion_surfs, tag=1)
    gmsh.model.setPhysicalName(2, 1, "inclusions")
    gmsh.model.addPhysicalGroup(2, matrix_surfs, tag=2)
    gmsh.model.setPhysicalName(2, 2, "matrix")

    # --- Periodic boundary constraints (origin tile) ---
    eps = 1e-9
    xL, xR = 0.0, float(nx)
    yB, yT = 0.0, float(ny)

    left = gmsh.model.getEntitiesInBoundingBox(xL-eps, yB-eps, -eps, xL+eps, yT+eps,  eps, 1)
    right = gmsh.model.getEntitiesInBoundingBox(xR-eps, yB-eps, -eps, xR+eps, yT+eps,  eps, 1)
    bot = gmsh.model.getEntitiesInBoundingBox(xL-eps, yB-eps, -eps, xR+eps, yB+eps,  eps, 1)
    top = gmsh.model.getEntitiesInBoundingBox(xL-eps, yT-eps, -eps, xR+eps, yT+eps,  eps, 1)

    left_tags = [c[1] for c in left]
    right_tags = [c[1] for c in right]
    bot_tags = [c[1] for c in bot]
    top_tags = [c[1] for c in top]

    # Pair right -> left (translate by -nx)
    gmsh.model.mesh.setPeriodic(
        1, right_tags, left_tags,
        [1, 0, 0, -nx,
         0, 1, 0,  0,
         0, 0, 1,  0,
         0, 0, 0,  1]
    )

    # Pair top -> bottom (translate by -ny)
    gmsh.model.mesh.setPeriodic(
        1, top_tags, bot_tags,
        [1, 0, 0,  0,
         0, 1, 0, -ny,
         0, 0, 1,  0,
         0, 0, 0,  1]
    )

    # Mesh options (fast debug)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.33)
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.write("ms2d.msh")
    gmsh.fltk.run()  # interactive viewer
    gmsh.finalize()


if __name__ == "__main__":
    exit()

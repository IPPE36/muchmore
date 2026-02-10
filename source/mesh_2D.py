from typing import List, Literal

import gmsh
import numpy as np
from gmshModel.Model.GenericRVE import GenericRVE
from shapely.affinity import translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from skimage.measure import find_contours

from source.timer import timer


def periodic_contours(field, min_area=5) -> List[Polygon]:
    """
    Tile microstructure to consider inclusions outside and ensure consistent polygons for periodic meshing
    """

    tiled = np.tile(field, (3, 3)).astype(np.float32)
    contours = find_contours(tiled, level=0.5)

    # central tile bounds in pixel space:
    ny, nx = field.shape
    x0, x1 = nx, 2 * nx
    y0, y1 = ny, 2 * ny
    rve = box(x0, y0, x1, y1)

    polys = []
    for c in contours:
        if c.shape[0] < 10:
            continue

        poly = Polygon(np.c_[c[:, 1], c[:, 0]])
        if (not poly.is_valid) or poly.area <= 0:
            continue

        if not poly.intersects(rve):
            continue

        clipped = poly.intersection(rve)
        if clipped.is_empty:
            continue

        if clipped.area >= min_area:
            polys.append(poly)

    if polys:
        merged = unary_union(polys).buffer(0)
        if merged.geom_type == "Polygon":
            return [merged]
        elif merged.geom_type == "MultiPolygon":
            return list(merged.geoms)

    return []


def polygon_to_loop(gmsh_model, poly: Polygon, lc=0.0):
    """
    Adds a shapely polygon to gmsh as a curve loop.
    Returns curveLoopTag.
    """
    pts = list(poly.exterior.coords)
    pts = pts[:-1]
    p_tags = []
    for x, y in pts:
        p_tags.append(gmsh_model.occ.addPoint(float(x), float(y), lc))
    # create lines
    l_tags = []
    n = len(p_tags)
    for i in range(n):
        a = p_tags[i]
        b = p_tags[(i + 1) % n]
        l_tags.append(gmsh_model.occ.addLine(a, b))
    loop = gmsh_model.occ.addCurveLoop(l_tags)
    return loop


def mesh_2d(
    field: np.ndarray,
    algo: Literal["frontal-delaunay", "delaunay"] = "frontal-delaunay",
    element_order: Literal[1, 2] = 2,
    h: float = 0.03,
    name_model: str = "RVE",
    name_phase_a: str = "PHASE-A",
    name_phase_b: str = "PHASE-B",
    mesh_level: Literal[1, 2] = 2,
    physical_spacing: float = 1.0,
    show: bool = True,
    **kwargs,
):

    if algo not in ["frontal-delaunay", "delaunay"]:
        raise ValueError("Algorithm must be either frontal-delaunay or delaunay!")

    with timer("Find Contours"):
        polygons = periodic_contours(field)

    # Shift polygons into origin tile
    ny, nx = field.shape
    polys = []
    for poly in polygons:
        p = translate(poly, xoff=-nx, yoff=-ny)
        p = p.simplify(0.5, preserve_topology=True)
        polys.append(p)

    rve = GenericRVE(
        size=[float(nx), float(ny), 0.0],
        origin=[0.0, 0.0, 0.0],
        periodicityFlags=[1, 1, 0],
    )
    gmsh.model.add(name_model)

    h = np.sqrt(nx**2 + ny**2) * h
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 2.0 * h)
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", str(h))
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    gmsh.option.setNumber("Mesh.ElementOrder", element_order)
    algo = {"delaunay": 5, "frontal-delaunay": 6}[algo]
    gmsh.option.setNumber("Mesh.Algorithm3D", algo)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.ColorCarousel", 1)
    gmsh.option.setNumber("Mesh.Format", 39)  # = ABAQUS .inp
    gmsh.option.setNumber("General.Verbosity", 0)
    occ = gmsh.model.occ

    with timer("Setup Geometry"):

        # Inclusion surfaces (now also in origin coordinates)
        surfs = []
        for poly in polys:
            if poly.is_empty or (not poly.is_valid) or poly.area <= 0:
                continue
            loop = polygon_to_loop(gmsh.model, poly)
            surf = occ.addPlaneSurface([loop])
            surfs.append(surf)
        occ.synchronize()

        # RVE at origin
        rect = occ.addRectangle(0, 0, 0, nx, ny)

        # Mutual fragmentation: every surface is fragmented against every other
        objs = [(2, rect)] + [(2, s) for s in surfs]
        outDimTags, outMap = occ.fragment(objs, [])
        occ.synchronize()

        # Clip inclusions to the rve
        rect = occ.addRectangle(0, 0, 0, nx, ny)
        volumes = gmsh.model.occ.getEntities(dim=2)
        out, _ = gmsh.model.occ.intersect(
            volumes,
            [(2, rect)],
            removeObject=True,
            removeTool=True
        )
        gmsh.model.occ.synchronize()

        matrix_surfs = sorted({tag for (dim, tag) in outMap[0] if dim == 2})
        inclusion_surfs = sorted({
            tag
            for m in outMap[1:]
            for (dim, tag) in m
            if dim == 2
        })

        # --- Physical groups ---
        gmsh.model.addPhysicalGroup(2, inclusion_surfs, tag=1)
        gmsh.model.setPhysicalName(2, 1, "inclusions")
        gmsh.model.addPhysicalGroup(2, matrix_surfs, tag=2)
        gmsh.model.setPhysicalName(2, 2, "matrix")

        rve.setupPeriodicity()

    with timer("MESHING"):
        gmsh.model.mesh.generate(mesh_level)

    with timer("STORE RVE"):
        gmsh.write(f"{name_model}.inp")

    if show:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    exit()


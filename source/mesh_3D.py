import subprocess
from pathlib import Path
from typing import List, Literal

import gmsh
import numpy as np
import pyvista as pv
from gmshModel.Model.GenericRVE import GenericRVE
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes

from source.timer import timer
from source.abaqus import postprocess_inp

ROOT = Path(__file__).resolve().parents[1]
FREECAD_WORKER = str(ROOT / "source" / "freecad_utils.py")
FREECAD_PYTHON = r"C:\Program Files\FreeCAD 1.0\bin\python.exe"


def _get_volume_boundary_faces(vol_tags: list[int]) -> list[int]:
    faces = set()
    for v in vol_tags:
        for dim, tag in gmsh.model.getBoundary([(3, v)], oriented=False, recursive=False):
            if dim == 2:
                faces.add(tag)
    return sorted(faces)


def _match_interface_faces_by_com(
    a_faces: list[int],
    b_faces: list[int],
    tol: float = 1e-6,
) -> tuple[list[int], list[int]]:
    """
    Match interface faces between A and B by Center-of-Mass (COM) within a tolerance.
    Assumes the interface geometry is perfect and coincident.
    """

    def _round_key(xyz, tol: float) -> tuple[int, int, int]:
        """stable “binning” for matching coincident faces"""
        return tuple(int(round(c / tol)) for c in xyz)

    b_map: dict[tuple[int, int, int], list[int]] = {}
    for f in b_faces:
        com = gmsh.model.occ.getCenterOfMass(2, f)
        b_map.setdefault(_round_key(com, tol), []).append(f)

    a_iface, b_iface = [], []
    for f in a_faces:
        com = gmsh.model.occ.getCenterOfMass(2, f)
        key = _round_key(com, tol)
        candidates = b_map.get(key, [])
        if candidates:
            # if multiple candidates exist, just take one; geometry should make this 1-1
            g = candidates.pop()
            a_iface.append(f)
            b_iface.append(g)

    return a_iface, b_iface


def _import_and_get_new_volume_tags(path: str) -> list[int]:
    """import shapes from a step file and extract tags"""
    before = set(gmsh.model.getEntities(3))
    gmsh.model.occ.importShapes(path)
    gmsh.model.occ.synchronize()
    after = set(gmsh.model.getEntities(3))
    new = list(after - before)
    return [tag for _, tag in new]


def stl_to_step(stl_path: str, a_path: str, b_path: str, nx: int, ny: int, nz: int):
    subprocess.run(
        [FREECAD_PYTHON, FREECAD_WORKER, stl_path, a_path, b_path, str(nx), str(ny), str(nz)],
        text=True,
        capture_output=True
    )


def _clean_pv(poly: pv.PolyData, tol: float = 1e-6) -> pv.PolyData:
    poly = poly.triangulate()
    poly = poly.clean(tolerance=tol)
    poly = poly.compute_normals(auto_orient_normals=True, consistent_normals=True)
    return poly


def _split_parts(poly: pv.PolyData):
    out = poly.connectivity(extraction_mode="all")

    # Newer PyVista: MultiBlock
    if isinstance(out, pv.MultiBlock):
        return [p for p in out if p is not None and p.n_cells > 0]

    # Older PyVista: PolyData with RegionId
    rid_name = "RegionId" if "RegionId" in out.array_names else out.array_names[-1]
    regions = np.unique(out[rid_name])

    parts = []
    for r in regions:
        part = out.threshold([r, r], scalars=rid_name).extract_geometry()
        if part.n_cells:
            parts.append(part)
    return parts


def periodic_iso_surfaces(field, min_area: float = 50.0, step_size: int = 2) -> List[pv.PolyData]:

    # --- tile field ---
    tiled = np.tile(field, (3, 3, 3)).astype(np.float32)
    nx, ny, nz = field.shape
    x, y, z = np.indices(tiled.shape)

    # --- zero out outer region ---
    tiled[
        (x < nx * 0.85) | (x > nx * 2.15) |
        (y < ny * 0.85) | (y > ny * 2.15) |
        (z < nz * 0.85) | (z > nz * 2.15)
    ] = 0.0

    # --- signed distance field ---
    mask = tiled > 0.5
    phi = distance_transform_edt(mask) - distance_transform_edt(~mask)

    # --- marching cubes ---
    verts_zyx, faces, _, _ = marching_cubes(phi, level=0.0, step_size=step_size, allow_degenerate=False)
    verts_xyz = verts_zyx[:, [2, 1, 0]]
    faces_pv = np.c_[np.full(len(faces), 3), faces].astype(np.int64).ravel()
    surf = pv.PolyData(verts_xyz, faces_pv)

    # --- split into connected components ---
    return [p for p in _split_parts(surf) if float(p.area) >= min_area]


def mesh_3d(
    field: np.ndarray,
    algo: Literal["frontal", "delaunay", "hxt"] = "frontal",
    element_order: Literal[1, 2] = 1,
    char_len_factor: float = 1.0,
    name_model: str = "RVE",
    name_phase_a: str = "PHASE-A",
    name_phase_b: str = "PHASE-B",
    show: bool = False,
    physical_spacing: float = 1.0,
):

    stl_path = str(ROOT / "temp" / f"{name_model}.stl")
    inp_path = str(ROOT / "temp" / f"{name_model}.inp")
    out_path = inp_path.replace(".inp", "_post.inp")
    a_path = str(ROOT / "temp" / f"{name_model}_{name_phase_a}.step")
    b_path = str(ROOT / "temp" / f"{name_model}_{name_phase_b}.step")

    with timer("POSTPROCESS INP"):
        postprocess_inp(inp_path, out_path, name_model, name_phase_a, name_phase_b)
    exit()

    nx, ny, nz = field.shape
    rve = GenericRVE(
        size=[float(nx), float(ny), float(nz)],
        origin=[0.0, 0.0, 0.0],
        periodicityFlags=[1, 1, 1],
    )
    gmsh.model.add(name_model)

    algo = {"delaunay": 1, "frontal": 4, "hxt": 10}[algo]
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", element_order)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", char_len_factor)
    gmsh.option.setNumber("Mesh.Algorithm3D", algo)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.ColorCarousel", 1)
    gmsh.option.setNumber("Mesh.Format", 39)  # = ABAQUS .inp
    gmsh.option.setNumber("General.Verbosity", 0)

    with timer("MARSHING CUBES"):
        parts = periodic_iso_surfaces(field, step_size=nx//32)
        shift = np.array([nx, ny, nz], dtype=float)
        parts = [_clean_pv(p.translate(-shift, inplace=False)) for p in parts]
        merged = pv.merge(parts)
        merged.save(stl_path)

    with timer("FREECAD STEP"):
        stl_to_step(stl_path, a_path, b_path, nx, ny, nz)

    with timer("GMSH LOAD A"):
        a_tags = _import_and_get_new_volume_tags(a_path)

    with timer("GMSH LOAD B"):
        b_tags = _import_and_get_new_volume_tags(b_path)

    with timer("GMSH SETUP"):
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, a_tags, name=name_phase_a)
        gmsh.model.addPhysicalGroup(3, b_tags, name=name_phase_b)
        # a_faces = _get_volume_boundary_faces(a_tags)
        # b_faces = _get_volume_boundary_faces(b_tags)
        # a_iface, b_iface = _match_interface_faces_by_com(a_faces, b_faces, tol=1e-6)
        # gmsh.model.addPhysicalGroup(2, a_iface, name=f"{name_phase_a}-IF")
        # gmsh.model.addPhysicalGroup(2, b_iface, name=f"{name_phase_b}-IF")
        rve.setupPeriodicity()

    with timer("MESHING VOLUMES"):
        gmsh.model.mesh.generate(3)

    with timer("STORE RVE"):
        gmsh.write(inp_path)

    with timer("POSTPROCESS INP"):
        postprocess_inp(inp_path, out_path, name_model, name_phase_a, name_phase_b)

    if show:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    exit()

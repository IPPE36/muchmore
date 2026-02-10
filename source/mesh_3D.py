import subprocess
from pathlib import Path
from typing import List, Literal

import gmsh
import numpy as np
import pyvista as pv
from gmshModel.Model.GenericRVE import GenericRVE
from gmshModel.Model.RandomInclusionRVE import RandomInclusionRVE
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes

from source.timer import timer
from source.abaqus2 import postprocess_inp

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

    def _round(xyz, tol: float) -> tuple[int, int, int]:
        """stable “binning” for matching coincident faces"""
        return tuple(int(round(c / tol)) for c in xyz)

    b_map: dict[tuple[int, int, int], list[int]] = {}
    for f in b_faces:
        com = gmsh.model.occ.getCenterOfMass(2, f)
        b_map.setdefault(_round(com, tol), []).append(f)

    a_iface, b_iface = [], []
    for f in a_faces:
        com = gmsh.model.occ.getCenterOfMass(2, f)
        key = _round(com, tol)
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
    element_order: Literal[1, 2] = 2,
    h: float = 0.05,
    name_model: str = "RVE",
    name_phase_a: str = "PHASE-A",
    name_phase_b: str = "PHASE-B",
    mesh_level: Literal[1, 2, 3] = 3,
    physical_spacing: float = 1.0,
    load_case: Literal["Tensile-X", "Tensile-Y", "Tensile-Z", "Shear-XY", "Shear-XZ", "Shear-YZ"] = "Tensile-X",
    strain: float = 0.03,
    show: bool = False,
    **kwargs,
):

    if algo not in ["frontal", "delaunay", "hxt"]:
        raise ValueError("Algorithm must be either frontal, delaunay or hxt!")

    stl_path = str(ROOT / "temp" / f"{name_model}.stl")
    inp_path = str(ROOT / "temp" / f"{name_model}.inp")
    out_path = inp_path.replace(".inp", "_post.inp")
    a_path = str(ROOT / "temp" / f"{name_model}_{name_phase_a}.step")
    b_path = str(ROOT / "temp" / f"{name_model}_{name_phase_b}.step")

    with timer("POSTPROCESS INP"):
        postprocess_inp(inp_path, out_path, name_model, name_phase_a, name_phase_b, load_case, strain, physical_spacing)
    exit()

    nx, ny, nz = field.shape
    rve = GenericRVE(
        size=[float(nx), float(ny), float(nz)],
        origin=[0.0, 0.0, 0.0],
        periodicityFlags=[1, 1, 1],
    )
    gmsh.model.add(name_model)

    h = np.sqrt(nx**2 + ny**2 + nz**2) * h
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.75 * h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1.25 * h)
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", str(h))
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    gmsh.option.setNumber("Mesh.ElementOrder", element_order)
    algo = {"delaunay": 1, "frontal": 4, "hxt": 10}[algo]
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
        a_faces = _get_volume_boundary_faces(a_tags)
        b_faces = _get_volume_boundary_faces(b_tags)
        a_iface, b_iface = _match_interface_faces_by_com(a_faces, b_faces, tol=1e-6)
        gmsh.model.addPhysicalGroup(2, a_iface, name=f"{name_phase_a}-IF")
        gmsh.model.addPhysicalGroup(2, b_iface, name=f"{name_phase_b}-IF")
        rve.setupPeriodicity()

    with timer("MESHING"):
        gmsh.model.mesh.generate(mesh_level)

    with timer("STORE RVE"):
        gmsh.write(inp_path)

    if show:
        gmsh.fltk.run()

    with timer("POSTPROCESS INP"):
        postprocess_inp(inp_path, out_path, name_model, name_phase_a, name_phase_b, load_case, strain, physical_spacing)

    gmsh.finalize()
    return


def mesh_3d_random_inclusion(
        size: int = 128,
        inclusion_sets: List[int] = [24, 12],  # place 12 inclusions with radius 24
        algo: Literal["frontal", "delaunay", "hxt"] = "frontal",
        element_order: Literal[1, 2] = 2,
        name_model: str = "RVE",
        max_attempts: int = 10000,
        min_rel_dist_bnd: float = 0.1,
        min_rel_dist_inc: float = 0.1,
        name_phase_a: str = "PHASE-A",
        name_phase_b: str = "PHASE-B",
        physical_spacing: float = 1.0,
        load_case: Literal["Tensile-X", "Tensile-Y", "Tensile-Z", "Shear-XY", "Shear-XZ", "Shear-YZ"] = "Tensile-X",
        strain: float = 0.03,
        show: bool = True,
        **kwargs,
):
    init_params = {
        "inclusionSets": inclusion_sets,
        "inclusionType": "Sphere",
        "size": [size] * 3,
        "origin": [0, 0, 0],
        "periodicityFlags": [1, 1, 1],
        "domainGroup": name_phase_a,
        "inclusionGroup": name_phase_b,
        "gmshConfigChanges": {
            "General.Terminal": 0,
            "Mesh.CharacteristicLengthExtendFromBoundary": 0,
            "Mesh.ElementOrder": element_order,
            "Mesh.Algorithm3D": {"delaunay": 1, "frontal": 4, "hxt": 10}[algo],
            "Mesh.OptimizeNetgen": 1,
            "Mesh.Optimize": 1,
            "Mesh.ColorCarousel": 1,
            "Mesh.Format": 39,  # = ABAQUS .inp
            "General.Verbosity": 0,
        }
    }
    model_params = {
        "placementOptions": {
            "maxAttempts": max_attempts,
            "min_rel_dist_bnd": min_rel_dist_bnd,
            "minRelDistInc": min_rel_dist_inc
        }
    }
    mesh_params = {
        "threads": None,
        "refinementOptions": {
            "maxMeshSize": "auto",
            "inclusionRefinement": True,
            "interInclusionRefinement": True,
            "elementsPerCircumference": 18,
            "elementsBetweenInclusions": 3,
            "inclusionRefinementWidth": 3,
            "transitionElements": "auto",
            "aspectRatio": 1.5
        }
    }
    rve = RandomInclusionRVE(**init_params)

    with timer("Setup Geometry"):
        rve.createGmshModel(**model_params)

    with timer("MESHING"):
        rve.createMesh(**mesh_params)

    with timer("STORE RVE"):
        inp_path = str(ROOT / "temp" / f"{name_model}.inp")
        rve.saveMesh(inp_path)

    if show:
        gmsh.fltk.run()

    with timer("POSTPROCESS INP"):
        out_path = inp_path.replace(".inp", "_post.inp")
        postprocess_inp(inp_path, out_path, name_model, name_phase_a, name_phase_b, load_case, strain, physical_spacing)

    return


if __name__ == "__main__":
    exit()

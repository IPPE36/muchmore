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

ROOT = Path(__file__).resolve().parents[1]
FREECAD_WORKER = str(ROOT / "source" / "freecad_utils.py")
FREECAD_PYTHON = r"C:\Program Files\FreeCAD 1.0\bin\python.exe"


def stl_to_step(stl_path: str, step_path: str, nx: int, ny: int, nz: int, verbose: bool = False):
    p = subprocess.run(
        [FREECAD_PYTHON, FREECAD_WORKER, stl_path, step_path, str(nx), str(ny), str(nz)],
        text=True,
        capture_output=True
    )
    if verbose:
        print("returncode:", p.returncode)
        print("STDOUT:\n", p.stdout)
        print("STDERR:\n", p.stderr)
    return step_path


def clean_pv(poly: pv.PolyData, tol: float = 1e-6) -> pv.PolyData:
    poly = poly.triangulate()
    poly = poly.clean(tolerance=tol)
    poly = poly.compute_normals(auto_orient_normals=True, consistent_normals=True)
    return poly


def split_parts(poly: pv.PolyData):
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


def periodic_iso_surfaces(field, min_area: float = 50.0) -> List[pv.PolyData]:

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
    verts_zyx, faces, _, _ = marching_cubes(phi, level=0.0, step_size=2, allow_degenerate=False)
    verts_xyz = verts_zyx[:, [2, 1, 0]]

    faces_pv = np.c_[np.full(len(faces), 3), faces].astype(np.int64).ravel()
    surf = pv.PolyData(verts_xyz, faces_pv)

    # --- split into connected components ---
    return [p for p in split_parts(surf) if float(p.area) >= min_area]


def mesh_3d(
    field: np.ndarray,
    algo: Literal["frontal", "delaunay", "hxt"] = "frontal",
    element_order: Literal[1, 2] = 1,
    lc: float = 0.3,
    model_name: str = "new_rve",
    show: bool = True,
):
    stl_path = str(ROOT / "temp" / f"{model_name}.stl")
    step_path = str(ROOT / "temp" / f"{model_name}.step")

    nx, ny, nz = field.shape
    rve = GenericRVE(
        size=[float(nx), float(ny), float(nz)],
        origin=[0.0, 0.0, 0.0],
        periodicityFlags=[1, 1, 1],
    )
    gmsh.model.add(model_name)
    occ = gmsh.model.occ

    algo_dict = {
        "delaunay": 1,  # default
        "frontal": 4,  # most robust/slower
        "hxt": 10,  # hex dominated
    }

    algo = algo_dict[algo]
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", element_order)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", lc)
    gmsh.option.setNumber("Mesh.Algorithm3D", algo)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.ColorCarousel", 1)
    gmsh.option.setNumber("Mesh.Format", 39)  # = ABAQUS .inp
    gmsh.option.setNumber("General.Verbosity", 0)

    with timer("MARSHING CUBES"):
        parts = periodic_iso_surfaces(field)
        shift = np.array([nx, ny, nz], dtype=float)
        parts = [clean_pv(p.translate(-shift, inplace=False)) for p in parts]
        merged = pv.merge(parts)
        merged.save(stl_path)

    with timer("FREECAD STEP"):
        stl_to_step(stl_path, step_path, nx, ny, nz)

    with timer("GMSH SETUP"):
        occ.importShapes(step_path)
        occ.synchronize()
        occ.removeAllDuplicates()
        occ.synchronize()
        rve.setupPeriodicity()

    with timer("MESHING VOLUMES"):
        gmsh.model.mesh.generate(3)

    with timer("STORE RVE"):
        gmsh.write(f"{model_name}.inp")

    if show:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    exit()

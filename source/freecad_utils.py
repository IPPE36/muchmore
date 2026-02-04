import sys
import traceback

import FreeCAD as App
import Mesh
import Part


def stl_to_step(
    stl_path: str,
    a_path: str,
    b_path: str,
    nx: float,
    ny: float,
    nz: float,
    tol: float = 1e-12,
) -> None:

    doc = App.ActiveDocument
    if doc is None:
        doc = App.newDocument("STL_to_STEP")

    # ----------------------------
    # Import STL as mesh
    # ----------------------------
    mesh = Mesh.Mesh(stl_path)
    mesh_obj = doc.addObject("Mesh::Feature", "Mesh")
    mesh_obj.Mesh = mesh

    # ----------------------------
    # Mesh -> Shape
    # ----------------------------
    shape = Part.Shape()
    shape.makeShapeFromMesh(mesh.Topology, tol)

    shape_obj = doc.addObject("Part::Feature", "Shape")
    shape_obj.Shape = shape

    # ----------------------------
    # Shape -> Solid
    # ----------------------------
    solid = Part.makeSolid(shape)
    solid_obj = doc.addObject("Part::Feature", "Solid")
    solid_obj.Shape = solid
    doc.recompute()

    # ----------------------------
    # Create RVE cutting box
    # Origin = (0,0,0)
    # ----------------------------
    box = Part.makeBox(nx, ny, nz)
    box_obj = doc.addObject("Part::Feature", "RVE_Box")
    box_obj.Shape = box
    doc.recompute()

    # ----------------------------
    # Phase A: solid ∩ box
    # ----------------------------
    phaseA_shape = solid_obj.Shape.common(box_obj.Shape)
    phaseA_obj = doc.addObject("Part::Feature", "Phase_A")
    phaseA_obj.Shape = phaseA_shape
    doc.recompute()

    # ----------------------------
    # Phase B: box \ Phase A
    # (complement of A inside the RVE)
    # ----------------------------
    phaseB_shape = box_obj.Shape.cut(phaseA_shape)
    phaseB_obj = doc.addObject("Part::Feature", "Phase_B")
    phaseB_obj.Shape = phaseB_shape
    doc.recompute()

    # ----------------------------
    # Export both to STEP
    # ----------------------------
    Part.export([phaseA_obj], a_path)
    Part.export([phaseB_obj], b_path)

    return None


def main(argv: list[str]) -> int:
    stl_path = argv[1]
    a_path = argv[2]
    b_path = argv[3]
    nx = float(argv[4])
    ny = float(argv[5])
    nz = float(argv[6])
    stl_to_step(stl_path, a_path, b_path, nx, ny, nz)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

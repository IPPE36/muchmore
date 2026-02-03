import sys
import traceback

import FreeCAD as App
import Mesh
import Part


def stl_to_step(
    stl_path: str,
    step_path: str,
    nx: float,
    ny: float,
    nz: float,
    tol: float = 1e-12,
) -> str:

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
    # Boolean Cut: solid ∩ box
    # (keep only the part inside RVE)
    # ----------------------------
    cut_shape = solid_obj.Shape.common(box_obj.Shape)
    # cut_shape = cut_shape.removeSplitter()

    cut_obj = doc.addObject("Part::Feature", "RVE_Cut")
    cut_obj.Shape = cut_shape
    doc.recompute()

    # ----------------------------
    # Export to STEP
    # ----------------------------
    Part.export([cut_obj], step_path)

    return step_path


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: freecad_worker.py input.stl output.step [tol]", file=sys.stderr)
        return 2

    stl_path = argv[1]
    step_path = argv[2]
    nx = float(argv[3])
    ny = float(argv[4])
    nz = float(argv[5])

    stl_to_step(stl_path, step_path, nx, ny, nz)
    print(f"OK: {step_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

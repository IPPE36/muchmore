
import gmsh
import numpy as np
from skimage import measure
from source.plotting import plot_contours, plot_surface, plot_2d_mesh
from matplotlib.path import Path
from scipy.spatial import Delaunay


def surface_mesh(field: np.ndarray):
    dim = field.ndim
    if dim == 2:
        # contours (lines)
        contours = measure.find_contours(field, level=0.5)
        plot_contours(field, contours, "contours.png")

        # mesh inclusions (phase=1)
        v1, f1, r1, p1 = mesh_all_contours_gmsh(
            contours, field,
            lc=2.0, periodic_x=True, periodic_y=True,
            target_phase=1, compact=True
        )
        plot_2d_mesh(field, v1, f1, filepath="mesh_phase1.png")

        # mesh matrix (phase=0)
        v0, f0, r0, p0 = mesh_all_contours_gmsh(
            contours, field,
            lc=2.0, periodic_x=True, periodic_y=True,
            target_phase=0, compact=True
        )
        plot_2d_mesh(field, v0, f0, filepath="mesh_phase0.png")

    elif dim == 3:
        # isosurfaces (triangles)
        verts, faces, normals, values = measure.marching_cubes(field, level=0.5)
        plot_surface(verts, faces, filepath="surface.png")
    else:
        raise NotImplementedError
    return


def face_phase_from_field(field: np.ndarray, verts: np.ndarray, faces: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """
    Determine phase per triangle by sampling 'field' at triangle centroid.
    Returns 0/1 labels (int).
    """
    assert field.ndim == 2
    c = verts[faces].mean(axis=1)  # (nF, 2) in (x,y)

    # centroid -> nearest pixel index (row=y, col=x)
    x = np.clip(np.rint(c[:, 0]).astype(int), 0, field.shape[1] - 1)
    y = np.clip(np.rint(c[:, 1]).astype(int), 0, field.shape[0] - 1)

    return (field[y, x] >= threshold).astype(np.int32)


def compact_mesh(verts: np.ndarray, faces: np.ndarray):
    """
    Remove unused vertices and reindex faces.
    """
    used = np.unique(faces.ravel())
    new_index = -np.ones(verts.shape[0], dtype=int)
    new_index[used] = np.arange(used.shape[0], dtype=int)

    verts2 = verts[used]
    faces2 = new_index[faces]
    return verts2, faces2


def mesh_all_contours_gmsh(
    contours,
    field,
    *,
    lc=2.0,
    simplify_step=1,
    smooth=False,
    periodic_x=False,
    periodic_y=False,
    msh_path=None,
    target_phase: int | None = None,
    threshold: float = 0.5,
    compact: bool = True,
):
    """
    Mesh a bounding rectangle [0,W]x[0,H] and all contour regions (as separate subdomains).
    Uses boolean fragment to obtain separate surfaces.

    Returns
    -------
    verts : (N,2) float
    faces : (M,3) int (0-based)
    face_region : (M,) int surface-tag id (Gmsh surface tag) for each face
    """

    H, W = field.shape
    if len(contours) == 0:
        raise ValueError("No contours provided.")

    def to_xy(c):
        c = c[::max(1, simplify_step)]
        xy = np.stack([c[:, 1], c[:, 0]], axis=1).astype(float)  # (x,y)=(col,row)
        if np.linalg.norm(xy[0] - xy[-1]) > 1e-9:
            xy = np.vstack([xy, xy[0]])
        if smooth and xy.shape[0] > 5:
            xys = xy.copy()
            xys[1:-1] = (xy[:-2] + xy[1:-1] + xy[2:]) / 3.0
            xys[-1] = xys[0]
            xy = xys
        return xy

    loops_xy = [to_xy(c) for c in contours]

    # --- build nesting: parent[i] = index of smallest loop that contains i, or None
    paths = [Path(l) for l in loops_xy]
    areas = []
    for l in loops_xy:
        x, y = l[:-1, 0], l[:-1, 1]
        areas.append(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    abs_area = np.abs(np.array(areas))

    # pick a representative point for each loop (first vertex)
    reps = [l[0] for l in loops_xy]

    parent = [None] * len(loops_xy)
    for i in range(len(loops_xy)):
        containers = []
        for j in range(len(loops_xy)):
            if i == j:
                continue
            if paths[j].contains_point(reps[i]):
                containers.append(j)
        if containers:
            # choose smallest-area container
            parent[i] = min(containers, key=lambda j: abs_area[j])

    children = [[] for _ in loops_xy]
    for i, p in enumerate(parent):
        if p is not None:
            children[p].append(i)

    # parity depth: depth%2==0 means "filled", depth%2==1 means "hole", etc.
    depth = [0] * len(loops_xy)
    for i in range(len(loops_xy)):
        d = 0
        p = parent[i]
        while p is not None:
            d += 1
            p = parent[p]
        depth[i] = d

    gmsh.initialize()
    gmsh.model.add("all_contours")

    # rectangle domain
    rect = gmsh.model.occ.addRectangle(0, 0, 0, float(W), float(H))

    # helper: add closed spline loop as a surface
    def add_loop_surface(xy):
        pt_tags = [gmsh.model.occ.addPoint(float(x), float(y), 0.0) for x, y in xy[:-1]]
        # close the spline by repeating first point in the point list
        curve = gmsh.model.occ.addSpline(pt_tags + [pt_tags[0]])
        loop = gmsh.model.occ.addCurveLoop([curve])
        surf = gmsh.model.occ.addPlaneSurface([loop])
        return surf

    # Create surfaces for "filled" loops at even depth (0,2,4,...)
    # These represent inclusion regions; odd depth loops are holes inside those inclusions.
    inclusion_surfs = [add_loop_surface(loops_xy[i]) for i in range(len(loops_xy)) if depth[i] % 2 == 0]

    gmsh.model.occ.synchronize()

    # Fragment rectangle with all inclusion surfaces -> gives separate subdomains
    # Output is a list of (dim, tag) entities created
    objs = [(2, rect)]
    tools = [(2, s) for s in inclusion_surfs]
    out, _ = gmsh.model.occ.fragment(objs, tools, removeObject=False, removeTool=False)
    gmsh.model.occ.synchronize()

    # Periodic boundaries on the (possibly fragmented) outer boundary:
    if periodic_x or periodic_y:
        # identify boundary curves on outer rectangle by bounding box
        # we use the rectangle extents [0,W]x[0,H]
        eps = 1e-7
        # collect ALL boundary curves of the rectangle entity (dim=2)
        # after fragment, the "rectangle surface" might be split; easiest: find curves on the global outer boundary.
        curves = gmsh.model.getEntities(1)
        left, right, bottom, top = [], [], [], []
        for dim, tag in curves:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            on_left   = abs(xmin - 0.0) < eps and abs(xmax - 0.0) < eps
            on_right  = abs(xmin - float(W)) < eps and abs(xmax - float(W)) < eps
            on_bottom = abs(ymin - 0.0) < eps and abs(ymax - 0.0) < eps
            on_top    = abs(ymin - float(H)) < eps and abs(ymax - float(H)) < eps
            if on_left: left.append(tag)
            if on_right: right.append(tag)
            if on_bottom: bottom.append(tag)
            if on_top: top.append(tag)

        if periodic_x and left and right:
            gmsh.model.mesh.setPeriodic(
                1, right, left,
                [1, 0, 0, float(W),
                 0, 1, 0, 0.0,
                 0, 0, 1, 0.0]
            )
        if periodic_y and bottom and top:
            gmsh.model.mesh.setPeriodic(
                1, top, bottom,
                [1, 0, 0, 0.0,
                 0, 1, 0, float(H),
                 0, 0, 1, 0.0]
            )

    # Mesh options
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    gmsh.model.mesh.generate(2)

    if msh_path:
        gmsh.write(msh_path)

    # Extract nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(node_coords, dtype=float).reshape(-1, 3)
    verts = node_coords[:, :2]
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    # Extract triangles per surface, keep a region id for each face
    faces_list = []
    region_list = []

    # get all surfaces (dim=2) and extract triangles on each
    for dim, s_tag in gmsh.model.getEntities(2):
        etypes, _, enodes = gmsh.model.mesh.getElements(dim=2, tag=s_tag)
        for etype, conn in zip(etypes, enodes):
            if int(etype) == 2:  # 3-node triangles
                conn = np.array(conn, dtype=int).reshape(-1, 3)
                tri = np.vectorize(tag_to_idx.get)(conn)
                faces_list.append(tri)
                region_list.append(np.full(tri.shape[0], int(s_tag), dtype=int))

    if not faces_list:
        gmsh.finalize()
        raise RuntimeError("No triangular elements extracted.")

    faces = np.vstack(faces_list).astype(int)
    face_region = np.concatenate(region_list)

    # phase per face from the scalar field
    face_phase = face_phase_from_field(field, verts, faces, threshold=threshold)

    # Filter to a target phase if requested
    if target_phase is not None:
        if target_phase not in (0, 1):
            gmsh.finalize()
            raise ValueError("target_phase must be 0, 1, or None")

        keep = (face_phase == int(target_phase))
        faces = faces[keep]
        face_region = face_region[keep]
        face_phase = face_phase[keep]

        if compact:
            verts, faces = compact_mesh(verts, faces)

    gmsh.finalize()
    return verts, faces, face_region, face_phase


if __name__ == "__main__":
    exit()
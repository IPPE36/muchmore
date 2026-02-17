from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict
from typing import List, Tuple, Literal, Set

from source.elements import face_to_nodes_mapping, nodes_to_face_mapping

_element_hdr = re.compile(r"^\s*\*ELEMENT\b", re.IGNORECASE)


def _read_elset(lines: List[str], elset_name: str) -> Set[int]:
    """Read an Abaqus *ELSET from an .inp file. Returns a Set of element IDs in the ELSET"""

    elset_name = elset_name.upper()
    elements: Set[int] = set()

    in_elset = False
    is_generate = False

    for line in lines:

        # Skip empty or comment lines
        if not line or line.startswith("**"):
            continue

        # Start of a new keyword
        if line.startswith("*"):
            in_elset = False
            is_generate = False

            if line.upper().startswith("*ELSET"):
                header = line.upper().replace(" ", "")
                if f"ELSET={elset_name}" in header:
                    in_elset = True
                    is_generate = "GENERATE" in header
            continue

        # Inside the ELSET definition
        if in_elset:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if is_generate:
                if len(parts) != 3:
                    raise ValueError(
                        f"Malformed GENERATE line in ELSET {elset_name}: '{line}'"
                    )
                start, end, step = map(int, parts)
                elements.update(range(start, end + 1, step))
            else:
                for p in parts:
                    elements.add(int(p))

    return elements


def _classify_boundary_nodes(side_nodes: dict) -> dict:
    """Returns a dict with:
      - corners: dict[name] -> sorted list of node ids (usually length 1 each)
      - edges:   dict[name] -> sorted list of node ids (excluding corners)
      - face_interior: dict[face] -> sorted list of node ids (excluding edges+corners)
    """

    # make sets for fast ops
    S = {k: set(v) for k, v in side_nodes.items()}

    # --- 8 corners (triple intersections) ---
    corner_defs = {
        "C_XMIN_YMIN_ZMIN": ("XMIN", "YMIN", "ZMIN"),  # 1 C_XMIN_YMIN_ZMIN
        "C_XMIN_YMIN_ZMAX": ("XMIN", "YMIN", "ZMAX"),  # 2 C_XMIN_YMIN_ZMAX
        "C_XMIN_YMAX_ZMAX": ("XMIN", "YMAX", "ZMAX"),  # 3 C_XMIN_YMAX_ZMAX
        "C_XMIN_YMAX_ZMIN": ("XMIN", "YMAX", "ZMIN"),  # 4 C_XMIN_YMAX_ZMIN
        "C_XMAX_YMIN_ZMIN": ("XMAX", "YMIN", "ZMIN"),  # 5 C_XMAX_YMIN_ZMIN
        "C_XMAX_YMIN_ZMAX": ("XMAX", "YMIN", "ZMAX"),  # 6 C_XMAX_YMIN_ZMAX
        "C_XMAX_YMAX_ZMAX": ("XMAX", "YMAX", "ZMAX"),  # 7 C_XMAX_YMAX_ZMAX
        "C_XMAX_YMAX_ZMIN": ("XMAX", "YMAX", "ZMIN"),  # 8 C_XMAX_YMAX_ZMIN
    }
    corners = {nm: sorted(S[a] & S[b] & S[c]) for nm, (a, b, c) in corner_defs.items()}
    all_corner_nodes = set().union(*[set(v) for v in corners.values()]) if corners else set()

    # --- 12 edges (pairwise intersections minus corners) ---
    edge_defs = {
        # edges parallel to Z (XY corners)
        "E_XMIN_YMIN": ("XMIN", "YMIN"),  # zpar2 E_XMIN_YMIN
        "E_XMIN_YMAX": ("XMIN", "YMAX"),  # zpar1 E_XMIN_YMAX
        "E_XMAX_YMIN": ("XMAX", "YMIN"),  # zpar3 E_XMAX_YMIN
        "E_XMAX_YMAX": ("XMAX", "YMAX"),  # zpar4 E_XMAX_YMAX
        # edges parallel to Y (XZ corners)
        "E_XMIN_ZMIN": ("XMIN", "ZMIN"),  # ypar2 E_XMIN_ZMIN
        "E_XMIN_ZMAX": ("XMIN", "ZMAX"),  # ypar3 E_XMIN_ZMAX
        "E_XMAX_ZMIN": ("XMAX", "ZMIN"),  # ypar1 E_XMAX_ZMIN
        "E_XMAX_ZMAX": ("XMAX", "ZMAX"),  # ypar4 E_XMAX_ZMAX
        # edges parallel to X (YZ corners)
        "E_YMIN_ZMIN": ("YMIN", "ZMIN"),  # xpar1 E_YMIN_ZMIN
        "E_YMIN_ZMAX": ("YMIN", "ZMAX"),  # xpar2 E_YMIN_ZMAX
        "E_YMAX_ZMIN": ("YMAX", "ZMIN"),  # xpar4 E_YMAX_ZMIN
        "E_YMAX_ZMAX": ("YMAX", "ZMAX"),  # xpar3 E_YMAX_ZMAX
    }
    edges = {}
    for nm, (a, b) in edge_defs.items():
        e = (S[a] & S[b]) - all_corner_nodes
        edges[nm] = sorted(e)

    face_to_edges = {
        "XMIN": ["E_XMIN_YMIN", "E_XMIN_YMAX", "E_XMIN_ZMIN", "E_XMIN_ZMAX"],
        "XMAX": ["E_XMAX_YMIN", "E_XMAX_YMAX", "E_XMAX_ZMIN", "E_XMAX_ZMAX"],
        "YMIN": ["E_XMIN_YMIN", "E_XMAX_YMIN", "E_YMIN_ZMIN", "E_YMIN_ZMAX"],
        "YMAX": ["E_XMIN_YMAX", "E_XMAX_YMAX", "E_YMAX_ZMIN", "E_YMAX_ZMAX"],
        "ZMIN": ["E_XMIN_ZMIN", "E_XMAX_ZMIN", "E_YMIN_ZMIN", "E_YMAX_ZMIN"],
        "ZMAX": ["E_XMIN_ZMAX", "E_XMAX_ZMAX", "E_YMIN_ZMAX", "E_YMAX_ZMAX"],
    }

    # --- face interior = face nodes minus all edges on that face minus corners on that face ---
    face_interior = {}
    for face in ["XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"]:
        face_edges_nodes = set().union(*[set(edges[e]) for e in face_to_edges[face]])
        face_corner_nodes = S[face] & all_corner_nodes
        interior = S[face] - face_edges_nodes - face_corner_nodes
        face_interior[f"F_{face}"] = sorted(interior)

    return {"corners": corners, "edges": edges, "faces": face_interior}


def _bbox(nodes: dict) -> tuple:
    """Returns RVE boundary box dimensions"""
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    zs = [p[2] for p in nodes.values()]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _parse_nodes(lines) -> dict:
    """Reads an .inp file node table and returns a dict of {index:[x,y,z]}"""
    lines = lines.copy()
    nodes = {}  # nid -> (x,y,z)
    i = 0
    while i < len(lines):
        if lines[i].lstrip().upper().startswith("*NODE"):
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("*"):
                s = lines[i].strip()
                if s and not s.startswith("**"):
                    parts = [p.strip() for p in s.split(",")]
                    nid = int(parts[0])
                    x, y, z = map(float, parts[1:4])
                    nodes[nid] = (x, y, z)
                i += 1
            break
        i += 1
    if not nodes:
        raise RuntimeError("No *NODE block found.")
    return nodes


def _parse_elements(lines):
    """Reads an .inp file element table and returns a dict of {index:[nids]}, {index:C3D4 etc.}"""
    elem_conn = {}
    elem_type = {}
    i = 0
    while i < len(lines):
        ln = lines[i].lstrip()
        if ln.upper().startswith("*ELEMENT"):
            m = re.search(r"TYPE\s*=\s*([^,\s]+)", ln, flags=re.IGNORECASE)
            etype = (m.group(1).strip().upper() if m else "UNKNOWN")
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("*"):
                s = lines[i].strip()
                if s and not s.startswith("**"):
                    parts = [p.strip() for p in s.split(",") if p.strip()]
                    eid = int(parts[0])
                    conn = [int(p) for p in parts[1:]]
                    elem_conn[eid] = conn
                    elem_type[eid] = etype
                i += 1
            continue
        i += 1
    return elem_conn, elem_type


def _pair_coords(nodes, side_plus, side_minus, plane: str, tol=1e-6) -> Tuple[List[int], List[int]]:
    """
    - pair XMIN<->XMAX by matching (y,z)
    - pair YMIN<->YMAX by matching (x,z)
    - pair ZMIN<->ZMAX by matching (x,y)
    Returns: plus_nodes, minus_nodes
    """
    if plane == "X":
        key = lambda nid: (round(nodes[nid][1] / tol), round(nodes[nid][2] / tol))
    elif plane == "Y":
        key = lambda nid: (round(nodes[nid][0] / tol), round(nodes[nid][2] / tol))
    elif plane == "Z":
        key = lambda nid: (round(nodes[nid][0] / tol), round(nodes[nid][1] / tol))
    else:
        raise ValueError("plane must be X, Y, or Z")

    minus_map = {}
    for nid in side_minus:
        minus_map[key(nid)] = nid

    plus_nodes: List[int] = []
    minus_nodes: List[int] = []
    missing = 0

    for nidp in side_plus:
        k = key(nidp)
        nidm = minus_map.get(k)
        if nidm is None:
            missing += 1
            continue
        plus_nodes.append(nidp)
        minus_nodes.append(nidm)

    return plus_nodes, minus_nodes


def _format_nset(name, node_ids, per_line=16):
    """Adds a node set on the assembly level"""
    out = [f"*Nset, nset={name}, instance=PART-1\n"]
    for i in range(0, len(node_ids), per_line):
        chunk = node_ids[i:i+per_line]
        out.append(", ".join(str(n) for n in chunk) + "\n")
    return out


def _elem_centroid_nodes(nids, nodes):
    """Returns x,y,z of the centroid of elements"""
    xs = ys = zs = 0.0
    n = len(nids)
    for nid in nids:
        x, y, z = nodes[nid]
        xs += x; ys += y; zs += z
    return (xs / n, ys / n, zs / n)


def _elem_centroid_2d(eid, elem_conn, nodes):
    conn = elem_conn[eid]
    return _elem_centroid_nodes(conn, nodes)


def _round_key(x, y, z, tol):
    return (round(x / tol), round(y / tol), round(z / tol))


def _pair_2D_3D_surfaces(
    solid_elems,
    elem_conn,
    elem_types,
    nodes,
    surface_centroids,
    tol=1e-6,
):
    """Match *3D element faces* to a set/list of target surface centroids.
    Returns list of (solid_element_id, face_label like "S1")"""

    # hash all target surface centroids
    target = defaultdict(list)
    for c in surface_centroids:
        target[_round_key(c[0], c[1], c[2], tol)].append(c)

    matched = []
    used_targets = set()  # (key, index) pairs

    for eid in solid_elems:
        etype = elem_types[eid]
        faces = nodes_to_face_mapping(etype)

        conn = elem_conn[eid]

        for local_ids, flabel in faces:
            fnodes = [conn[i] for i in local_ids]
            fc = _elem_centroid_nodes(fnodes, nodes)

            k = _round_key(fc[0], fc[1], fc[2], tol)
            cands = target.get(k)
            if not cands:
                continue

            best_j = None
            best_d2 = None
            for j, tc in enumerate(cands):
                if (k, j) in used_targets:
                    continue
                dx = fc[0] - tc[0]
                dy = fc[1] - tc[1]
                dz = fc[2] - tc[2]
                d2 = dx*dx + dy*dy + dz*dz
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_j = j

            if best_j is not None and best_d2 <= tol * tol:
                matched.append((eid, flabel))
                used_targets.add((k, best_j))

    return matched


def _clean_body(
    nodes: Dict[int, Tuple[float, float, float]],
    elem_conn: Dict[int, List[int]],
    elem_types: Dict[int, str],
    elem_a_solid: Set[int],
    elem_b_solid: Set[int],
    surf_a: List[Tuple[int, str]],
    surf_b: List[Tuple[int, str]],
    *,
    surf_name_a: str = "SA",
    surf_name_b: str = "SB",
    elset_name_a: str = "PHASE-A",
    elset_name_b: str = "PHASE-B",
    physical_spacing: float = 1e-6,
) -> List[str]:
    """Cleans up the node table and the element table. Removes every non-3D elements. Resets indices.
    Adds the interface boundaries as surfaces referencing the 3D element surfaces.
    Scales node coordinates by physical spacing ps: x_new=x*ps, y_new=y*ps, z_new=z*ps."""

    keep_elems = set(elem_a_solid) | set(elem_b_solid)

    # --- used nodes from kept elements ---
    used_nodes = set()
    for eid in keep_elems:
        used_nodes.update(elem_conn[eid])

    # --- build node map (stable order) ---
    n_old_sorted = sorted(used_nodes)
    nid_map = {nid_old: i + 1 for i, nid_old in enumerate(n_old_sorted)}

    # --- build element map (stable order) ---
    e_old_sorted = sorted(keep_elems)
    eid_map = {eid_old: i + 1 for i, eid_old in enumerate(e_old_sorted)}

    # --- remap elements ---
    elem_conn_new: Dict[int, List[int]] = {}
    elem_types_new: Dict[int, str] = {}
    for eid_old in e_old_sorted:
        eid_new = eid_map[eid_old]
        elem_conn_new[eid_new] = [nid_map[n] for n in elem_conn[eid_old]]
        elem_types_new[eid_new] = elem_types[eid_old]

    # --- remap phase elsets ---
    elem_a_solid_new = sorted(eid_map[e] for e in elem_a_solid)
    elem_b_solid_new = sorted(eid_map[e] for e in elem_b_solid)

    # --- remap surfaces (element id changes, face label stays) ---
    surf_a_new = [(eid_map[eid], s) for (eid, s) in surf_a if eid in eid_map]
    surf_b_new = [(eid_map[eid], s) for (eid, s) in surf_b if eid in eid_map]

    # --- rebuild cleaned_body: *Node, *Element (grouped), *Elset, *Surface ---
    out: List[str] = []
    out.append("** --- cleaned 3D-only mesh (reindexed) ---\n")
    out.append(f"** --- physical spacing ps = {physical_spacing} mm/unit (coords scaled by ps) ---\n")

    # Nodes (scaled)
    out.append("*Node\n")
    for nid_old in n_old_sorted:
        nid_new = nid_map[nid_old]
        x, y, z = nodes[nid_old]
        x_new = x * physical_spacing
        y_new = y * physical_spacing
        z_new = z * physical_spacing
        out.append(f"{nid_new}, {x_new}, {y_new}, {z_new}\n")

    # Elements grouped by type
    by_type: "OrderedDict[str, List[int]]" = OrderedDict()
    for eid_new, et in elem_types_new.items():
        etu = (et or "UNKNOWN").upper()
        by_type.setdefault(etu, []).append(eid_new)

    for et, eids in by_type.items():
        out.append(f"*Element, type={et}\n")
        for eid_new in sorted(eids):
            conn = elem_conn_new[eid_new]
            out.append(f"{eid_new}, " + ", ".join(map(str, conn)) + "\n")

    # Elsets (solid only)
    def _write_elset(name: str, eids: List[int]) -> None:
        out.append(f"*Elset, elset={name}\n")
        chunk: List[str] = []
        for eid in eids:
            chunk.append(str(eid))
            if len(chunk) == 16:
                out.append(", ".join(chunk) + "\n")
                chunk = []
        if chunk:
            out.append(", ".join(chunk) + "\n")

    _write_elset(elset_name_a, elem_a_solid_new)
    _write_elset(elset_name_b, elem_b_solid_new)

    # Surfaces (element faces)
    def _write_surface(surf_name: str, surf: List[Tuple[int, str]]) -> None:
        out.append(f"*Surface, type=ELEMENT, name={surf_name}\n")
        for eid, face in surf:
            out.append(f"{eid}, {face}\n")

    out.append("** --- interface surfaces on reindexed solids ---\n")
    _write_surface(surf_name_a, surf_a_new)
    _write_surface(surf_name_b, surf_b_new)

    out.append("** --- end cleaned mesh ---\n")
    return out


def surface_to_nodes(
    lines: list[str],
    surface_name: str,
    elem_conn: dict[int, list[int]],
    elem_type: dict[int, str],
) -> set[int]:
    """Read an Abaqus *Surface, type=ELEMENT block and return the set of node IDs."""

    surface_name = surface_name.upper()
    nodes = set()

    in_surface = False

    for line in lines:
        if not line.strip() or line.startswith("**"):
            continue

        if line.startswith("*"):
            in_surface = False
            if line.upper().startswith("*SURFACE") and f"NAME={surface_name}" in line.upper():
                in_surface = True
            continue

        if in_surface:
            eid_str, face = [p.strip() for p in line.split(",")]
            eid = int(eid_str)

            conn = elem_conn[eid]
            fmap = face_to_nodes_mapping(elem_type[eid])

            local_ids = fmap[face.upper()]
            for i in local_ids:
                nodes.add(conn[i])

    return nodes


def postprocess_inp(
    inp_path: str | Path,
    out_path: str | Path,
    name_model: str = "RVE",
    name_phase_a: str = "PHASE-A",
    name_phase_b: str = "PHASE-B",
    load_case: Literal["Tensile-X", "Tensile-Y", "Tensile-Z"] = "Tensile-X",
    strain: float = 0.01,
    physical_spacing: float = 1.0,
    tie_constraint: bool = True,
    tol: float = 1e-6,
    mat_a: str = None,
    mat_b: str = None,
) -> Path:
    """Convert an orphan Abaqus .inp (no *Part/*Assembly) into Part/Assembly. Create periodic boundary conditions.
    Create loading scenarios..., Scale xyz to physical dimensions of the RVE..."""

    name_instance = f"PART-1"
    name_iface_a = f"{name_phase_a}-IF"
    name_iface_b = f"{name_phase_b}-IF"

    # ---- read ----
    text = Path(inp_path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines(keepends=True)

    # ---- header ----
    start_idx = 2
    preamble = lines[:start_idx]
    body = lines[start_idx:]

    # ---- body ----
    out: List[str] = []
    out.extend(preamble)
    out.append(f"*Part, name={name_model}\n")

    # --- element ids ---
    elem_conn, elem_type = _parse_elements(body)
    elem_a = _read_elset(body, name_phase_a)
    elem_b = _read_elset(body, name_phase_b)
    elem_a_iface = _read_elset(body, name_iface_a)
    elem_b_iface = _read_elset(body, name_iface_b)
    elem_a_solid = elem_a - elem_a_iface
    elem_b_solid = elem_b - elem_b_iface

    # --- node coordinates ---
    nodes = _parse_nodes(body)

    # --- define surface on 3d node set ---
    surf_centroids_a = [_elem_centroid_2d(e, elem_conn, nodes) for e in elem_a_iface]
    surf_centroids_b = [_elem_centroid_2d(e, elem_conn, nodes) for e in elem_b_iface]

    surf_a = _pair_2D_3D_surfaces(
        solid_elems=elem_a_solid,
        elem_conn=elem_conn,
        nodes=nodes,
        surface_centroids=surf_centroids_a,
        elem_types=elem_type,
        tol=tol,
    )
    surf_b = _pair_2D_3D_surfaces(
        solid_elems=elem_b_solid,
        elem_conn=elem_conn,
        nodes=nodes,
        surface_centroids=surf_centroids_b,
        elem_types=elem_type,
        tol=tol,
    )

    cleaned_body = _clean_body(
        nodes, elem_conn, elem_type, elem_a_solid, elem_b_solid, surf_a, surf_b,
        surf_name_a=name_iface_a,
        surf_name_b=name_iface_b,
        elset_name_a=name_phase_a,
        elset_name_b=name_phase_b,
        physical_spacing=physical_spacing,
    )
    out.extend(cleaned_body)

    # --- node ids ---
    nodes = _parse_nodes(cleaned_body)
    elem_conn, elem_type = _parse_elements(cleaned_body)
    elem_a = _read_elset(cleaned_body, name_phase_a)
    elem_b = _read_elset(cleaned_body, name_phase_b)
    nids_a = {nid for e in elem_a for nid in elem_conn[e]}
    nids_b = {nid for e in elem_b for nid in elem_conn[e]}
    nodes_a = {key: val for key, val in nodes.items() if key in nids_a}
    nodes_b = {key: val for key, val in nodes.items() if key in nids_b}

    nids_a_iface = surface_to_nodes(
        lines=cleaned_body,
        surface_name=name_iface_a,
        elem_conn=elem_conn,
        elem_type=elem_type,
    )
    nids_b_iface = surface_to_nodes(
        lines=cleaned_body,
        surface_name=name_iface_b,
        elem_conn=elem_conn,
        elem_type=elem_type,
    )
    nids_iface = nids_a_iface.union(nids_b_iface)

    # --- section assignments on PART level ---
    out.append("** --- section assignments ---\n")
    out.append(f"*Solid Section, elset={name_phase_a}, material={name_phase_a}\n")
    out.append(f"*Solid Section, elset={name_phase_b}, material={name_phase_b}\n")
    out.append("** --- end section assignments ---\n")

    out.append("*End Part\n\n")

    # --- ASSEMBLY block ---
    out.append("*Assembly, name=Assembly\n")
    out.append(f"*Instance, name={name_instance}, part={name_model}\n")
    out.append("*End Instance\n")

    # --- tie surfaces
    # stiffer matrix = master / assumes PP-PS nomenclature for phases A-B
    if tie_constraint:
        out.append("*Tie, name=TIE_AB, adjust=NO\n")
        out.append(f"{name_phase_b}-IF, {name_phase_a}-IF\n")

    # --- global bbox ---
    bb_min, bb_max = _bbox(nodes)
    xmin, ymin, zmin = bb_min
    xmax, ymax, zmax = bb_max
    Lx = bb_max[0] - bb_min[0]
    Ly = bb_max[1] - bb_min[1]
    Lz = bb_max[2] - bb_min[2]

    used_nsets = set()

    def add_to_used(nid: int):
        if nid not in used_nsets:
            out.append(f"*Nset, nset=NS-{nid}, instance=PART-1\n")
            out.append(f"{nid},\n")
            used_nsets.add(nid)

    side_names = ["XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"]
    side_nodes_all = {k: [] for k in side_names}
    corner_nodes = {"PMIN": [], "PMAX": []}
    for n, (x, y, z) in nodes.items():
        if n in nids_iface:
            continue
        if abs(x - xmin) < tol:
            side_nodes_all["XMIN"].append(n)
        if abs(x - xmax) < tol:
            side_nodes_all["XMAX"].append(n)
        if abs(y - ymin) < tol:
            side_nodes_all["YMIN"].append(n)
        if abs(y - ymax) < tol:
            side_nodes_all["YMAX"].append(n)
        if abs(z - zmin) < tol:
            side_nodes_all["ZMIN"].append(n)
        if abs(z - zmax) < tol:
            side_nodes_all["ZMAX"].append(n)
        if abs(x - xmax) < tol and abs(y - ymax) < tol and abs(z - zmax) < tol:
            corner_nodes["PMAX"].append(n)
        if abs(x - xmin) < tol and abs(y - ymin) < tol and abs(z - zmin) < tol:
            corner_nodes["PMIN"].append(n)

    for key in side_nodes_all:
        side_nodes_all[key] = sorted(side_nodes_all[key])

    # remove PMAX corner from *_MAX faces and PMIN corner from *_MIN faces
    # but only for the faces that are NOT the loaded axis
    loaded_axis = {"Tensile-X": "X", "Tensile-Y": "Y", "Tensile-Z": "Z"}[load_case]
    for ax in ["X", "Y", "Z"]:
        if ax == loaded_axis:
            continue
        side_nodes_all[f"{ax}MAX"] = [n for n in side_nodes_all[f"{ax}MAX"] if n not in corner_nodes["PMAX"]]
        side_nodes_all[f"{ax}MIN"] = [n for n in side_nodes_all[f"{ax}MIN"] if n not in corner_nodes["PMIN"]]

    out.extend(_format_nset("PMIN", corner_nodes["PMIN"]))
    out.extend(_format_nset("PMAX", corner_nodes["PMAX"]))
    out.extend(_format_nset("XMIN", side_nodes_all["XMIN"]))
    out.extend(_format_nset("XMAX", side_nodes_all["XMAX"]))
    out.extend(_format_nset("YMIN", side_nodes_all["YMIN"]))
    out.extend(_format_nset("YMAX", side_nodes_all["YMAX"]))
    out.extend(_format_nset("ZMIN", side_nodes_all["ZMIN"]))
    out.extend(_format_nset("ZMAX", side_nodes_all["ZMAX"]))

    for phase_, nodes_, nids_if_ in zip([name_phase_b, name_phase_a], [nodes_b, nodes_a], [nids_b_iface, nids_a_iface]):
        side_nodes = {k: [] for k in side_names}

        for n, (x, y, z) in nodes_.items():
            if n in nids_if_:
                continue
            if abs(x - xmin) < tol:
                side_nodes["XMIN"].append(n)
            if abs(x - xmax) < tol:
                side_nodes["XMAX"].append(n)
            if abs(y - ymin) < tol:
                side_nodes["YMIN"].append(n)
            if abs(y - ymax) < tol:
                side_nodes["YMAX"].append(n)
            if abs(z - zmin) < tol:
                side_nodes["ZMIN"].append(n)
            if abs(z - zmax) < tol:
                side_nodes["ZMAX"].append(n)

        for key in side_names:
            side_nodes[key] = sorted(side_nodes[key])

        bnd = _classify_boundary_nodes(side_nodes)

        P_XMIN, M_XMAX = _pair_coords(nodes_, bnd["faces"]["F_XMIN"], bnd["faces"]["F_XMAX"], "X")
        P_YMAX, M_YMIN = _pair_coords(nodes_, bnd["faces"]["F_YMAX"], bnd["faces"]["F_YMIN"], "Y")
        P_ZMAX, M_ZMIN = _pair_coords(nodes_, bnd["faces"]["F_ZMAX"], bnd["faces"]["F_ZMIN"], "Z")
        faces = [
            ('13', P_YMAX, M_YMIN),
            ('23', P_XMIN, M_XMAX),
            ('12', P_ZMAX, M_ZMIN),
        ]
        for face in faces:
            dofs, plus_nodes, minus_nodes = face
            for dof in dofs:
                for n_plus, n_minus in zip(plus_nodes, minus_nodes):
                    add_to_used(n_plus)
                    add_to_used(n_minus)
                    out.append(f"*Equation\n2\n")
                    out.append(f"NS-{n_plus}, {dof},  1.0\n")
                    out.append(f"NS-{n_minus}, {dof}, -1.0\n")

        P_XMAX_ZMAX1, M_XMIN_ZMAX1 = _pair_coords(nodes_, bnd["edges"]["E_XMAX_ZMAX"], bnd["edges"]["E_XMIN_ZMAX"], "X")
        P_XMAX_ZMIN2, M_XMIN_ZMIN2 = _pair_coords(nodes_, bnd["edges"]["E_XMAX_ZMIN"], bnd["edges"]["E_XMIN_ZMIN"], "X")
        P_XMIN_ZMAX3, M_XMIN_ZMIN3 = _pair_coords(nodes_, bnd["edges"]["E_XMIN_ZMAX"], bnd["edges"]["E_XMIN_ZMIN"], "Z")
        P_YMAX_ZMAX4, M_YMAX_ZMIN4 = _pair_coords(nodes_, bnd["edges"]["E_YMAX_ZMAX"], bnd["edges"]["E_YMAX_ZMIN"], "Z")
        P_YMIN_ZMAX5, M_YMIN_ZMIN5 = _pair_coords(nodes_, bnd["edges"]["E_YMIN_ZMAX"], bnd["edges"]["E_YMIN_ZMIN"], "Z")
        P_YMAX_ZMIN6, M_YMIN_ZMIN6 = _pair_coords(nodes_, bnd["edges"]["E_YMAX_ZMIN"], bnd["edges"]["E_YMIN_ZMIN"], "Y")
        P_XMAX_YMAX7, M_XMIN_YMAX7 = _pair_coords(nodes_, bnd["edges"]["E_XMAX_YMAX"], bnd["edges"]["E_XMIN_YMAX"], "X")
        P_XMAX_YMIN8, M_XMIN_YMIN8 = _pair_coords(nodes_, bnd["edges"]["E_XMAX_YMIN"], bnd["edges"]["E_XMIN_YMIN"], "X")
        P_XMIN_YMAX9, M_XMIN_YMIN9 = _pair_coords(nodes_, bnd["edges"]["E_XMIN_YMAX"], bnd["edges"]["E_XMIN_YMIN"], "Y")

        edges = [
            ('2', P_XMAX_ZMAX1, M_XMIN_ZMAX1),
            ('2', P_XMAX_ZMIN2, M_XMIN_ZMIN2),
            ('2', P_XMIN_ZMAX3, M_XMIN_ZMIN3),
            ('1', P_YMAX_ZMAX4, M_YMAX_ZMIN4),
            ('1', P_YMIN_ZMAX5, M_YMIN_ZMIN5),
            ('1', P_YMAX_ZMIN6, M_YMIN_ZMIN6),
            ('3', P_XMAX_YMAX7, M_XMIN_YMAX7),
            ('3', P_XMAX_YMIN8, M_XMIN_YMIN8),
            ('3', P_XMIN_YMAX9, M_XMIN_YMIN9),
        ]
        for edge in edges:
            dofs, plus_nodes, minus_nodes = edge
            for dof in dofs:
                for n_plus, n_minus in zip(plus_nodes, minus_nodes):
                    add_to_used(n_plus)
                    add_to_used(n_minus)
                    out.append(f"*Equation\n2\n")
                    out.append(f"NS-{n_plus}, {dof},  1.0\n")
                    out.append(f"NS-{n_minus}, {dof}, -1.0\n")

    if load_case == "Tensile-X":
        couple_dofs = [2, 3]  # tie Y & Z to PMIN/PMAX
    elif load_case == "Tensile-Y":
        couple_dofs = [1, 3]  # tie X & Z to PMIN/PMAX
    elif load_case == "Tensile-Z":
        couple_dofs = [1, 2]  # tie X & Y to PMIN/PMAX
    else:
        couple_dofs = []

    def couple_face_to_corner(face_key, corner_set, dof):
        for nid_ in side_nodes_all[face_key]:
            add_to_used(nid_)
            out.append("*Equation\n2\n")
            out.append(f"NS-{nid_}, {dof},  1.0\n")
            out.append(f"{corner_set}, {dof}, -1.0\n")

    if 1 in couple_dofs:
        couple_face_to_corner("XMAX", "PMAX", 1)
        couple_face_to_corner("XMIN", "PMIN", 1)

    if 2 in couple_dofs:
        couple_face_to_corner("YMAX", "PMAX", 2)
        couple_face_to_corner("YMIN", "PMIN", 2)

    if 3 in couple_dofs:
        couple_face_to_corner("ZMAX", "PMAX", 3)
        couple_face_to_corner("ZMIN", "PMIN", 3)

    out.append("*End Assembly\n\n")

    # --- materials ---
    out.append("** --- materials ---\n")
    if mat_a is None or mat_b is None:
        out.append(f"*Material, name={name_phase_a}\n")
        out.append("*Density\n")
        out.append("9.5e-10\n")
        out.append("*Elastic\n")
        out.append("1500., 0.39\n")
        out.append(f"*Material, name={name_phase_b}\n")
        out.append("*Density\n")
        out.append("9.5e-10\n")
        out.append("*Elastic\n")
        out.append("1000., 0.43\n")
        out.append("** --- end materials ---\n")
    else:
        out.append(mat_a)
        out.append(mat_b)

    # --- boundary conditions ---
    if load_case == "Tensile-X":
        bc = [("XMIN", 1, 0.0), ("XMAX", 1, strain * Lx)]
    elif load_case == "Tensile-Y":
        bc = [("YMIN", 2, 0.0), ("YMAX", 2, strain * Ly)]
    elif load_case == "Tensile-Z":
        bc = [("ZMIN", 3, 0.0), ("ZMAX", 3, strain * Lz)]
    else:
        raise ValueError(f"Unsupported load_case: {load_case}")

    # --- stepping ---
    out.append("** --- step & output ---\n")
    out.append(f"*STEP, name={load_case}, nlgeom=YES, inc=10000\n")
    out.append("*Static\n")
    out.append("0.1, 1., 1e-08, 0.1\n")
    out.append("*Boundary\n")
    out.append("PMIN, 1, 3, 0.0\n")
    for rp, dof, val in bc:
        out.append(f"{rp}, {dof}, {dof}, {val}\n")

    # --- FIELD OUTPUT ---
    out.append("*Restart, write, frequency=0\n")
    out.append("*Output, field, time interval=0.1, time marks=NO\n")
    out.append("*Node Output\n")
    out.append("U\n")
    out.append("*Element Output, directions=YES\n")
    out.append("S, LE, PE, PEEQ, STATUS, EVOL\n")
    out.append("*Contact Output\n")
    out.append("CSTRESS, CDISP, CSTATUS\n")

    # --- HISTORY OUTPUT ---
    out.append("*Output, history, frequency=1\n")
    out.append("*Energy Output\n")
    out.append("ALLIE, ALLSE, ALLWK, ALLPD, ALLDMD, ALLCD, ETOTAL\n")
    out.append("** --- end step & output ---\n")

    Path(out_path).write_text("".join(out), encoding="utf-8")

    return out_path


if __name__ == "__main__":
    exit()

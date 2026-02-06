from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict
from typing import List, Tuple, Literal, Set

_element_hdr = re.compile(r"^\s*\*ELEMENT\b", re.IGNORECASE)


def _pair_overlapping(nodes, nodes_a, nodes_b, tol=1e-6):
    """
    Find overlapping nodes between phase A and phase B by coordinates.

    Returns
    -------
    tuple[list[int], list[int]]
        Two lists:
        - node IDs from phase A
        - corresponding node IDs from phase B
    """

    def _round(x, y, z):
        return (round(x / tol), round(y / tol), round(z / tol))

    # build coordinate map for phase B
    coord_map_b = {}
    for nid in nodes_b:
        x, y, z = nodes[nid]
        k = _round(x, y, z)
        if k not in coord_map_b:
            coord_map_b[k] = nid

    nodesA = []
    nodesB = []

    # match phase A nodes against phase B
    for nid_a in nodes_a:
        x, y, z = nodes[nid_a]
        k = _round(x, y, z)
        nid_b = coord_map_b.get(k)
        if nid_b is not None:
            nodesA.append(nid_a)
            nodesB.append(nid_b)

    return nodesA, nodesB


def _read_elset(lines: List[str], elset_name: str) -> Set[int]:
    """
    Read an Abaqus *ELSET from an .inp file.
    Supports:
      *ELSET, ELSET=name
      *ELSET, ELSET=name, GENERATE
    Args:
        inp_path:
            Path to the .inp file
        elset_name:
            Name of the element set to read (case-insensitive)
    Returns:
        Set of element IDs in the ELSET
    Raises:
        ValueError if ELSET is not found or malformed
    """

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


def _is_3d_continuum(etype: str) -> bool:
    t = (etype or "").strip().upper()
    return t.startswith("C3D")  # covers C3D4, C3D10, C3D8R, C3D20R, ...


def _classify_rve_boundary_nodes(side_nodes: dict) -> dict:
    """
    side_nodes: dict with keys {"XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"} mapping to node-id lists.
    Returns a dict with:
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


def _clean_3d_continuum(body_lines: List[str],
                        elem_conn: Dict[int, List[int]],
                        elem_type: Dict[int, str]) -> List[str]:
    """
    Replaces the *ELEMENT section in `body_lines` with freshly generated element
    connectivity from elem_conn/elem_type (after any remapping/merging).
    Keeps everything before the first *ELEMENT and everything after the element
    section unchanged.
    """
    # Find first *ELEMENT line
    first_elem = None
    for i, line in enumerate(body_lines):
        if _element_hdr.match(line):
            first_elem = i
            break
    if first_elem is None:
        return body_lines[:]

    # Find end of element section (first keyword after element blocks)
    post_start = None
    i = first_elem
    while i < len(body_lines):
        if _element_hdr.match(body_lines[i]):
            i += 1
            while i < len(body_lines) and not body_lines[i].lstrip().startswith("*"):
                i += 1
            continue
        if body_lines[i].lstrip().startswith("*"):
            post_start = i
            break
        i += 1
    if post_start is None:
        post_start = len(body_lines)

    preamble = body_lines[:first_elem]
    postamble = body_lines[post_start:]

    # Preserve first-seen TYPE order from the original file
    type_order: List[str] = []
    seen = set()
    i = first_elem
    while i < post_start:
        ln = body_lines[i].lstrip()
        if ln.upper().startswith("*ELEMENT"):
            m = re.search(r"TYPE\s*=\s*([^,\s]+)", ln, flags=re.IGNORECASE)
            et = (m.group(1).strip().upper() if m else "UNKNOWN")
            if et not in seen:
                seen.add(et)
                type_order.append(et)
        i += 1

    # Bucket elements by type (only C3D*)
    by_type: "OrderedDict[str, List[int]]" = OrderedDict()
    for et in type_order:
        if _is_3d_continuum(et):
            by_type[et] = []

    # Some types might exist in parsed elems but not in headers (rare); append them
    for eid, et in elem_type.items():
        et = (et or "UNKNOWN").upper()
        if _is_3d_continuum(et) and et not in by_type:
            by_type[et] = []

    for eid, et in elem_type.items():
        et = (et or "UNKNOWN").upper()
        if _is_3d_continuum(et):
            by_type.setdefault(et, []).append(eid)

    out: List[str] = []
    out.extend(preamble)

    for et, eids in by_type.items():
        if not eids:
            continue
        out.append(f"*ELEMENT, TYPE={et}\n")
        for eid in eids:
            conn = elem_conn[eid]
            out.append(str(eid) + ", " + ", ".join(str(n) for n in conn) + "\n")

    out.extend(postamble)
    return out


def _bbox(nodes: dict) -> tuple:
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    zs = [p[2] for p in nodes.values()]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _ref_points(rps: list) -> list[str]:
    out: List[str] = []
    out.append("** --- reference points (for loads/BCs) ---\n")
    out.append("*Node\n")
    for nid, x, y, z, _ in rps:
        out.append(f"{nid}, {x}, {y}, {z}\n")
    for nid, _, _, _, name in rps:
        out.append(f"*Nset, nset={name}\n")
        out.append(f"{nid},\n")
    out.append("** --- end reference points ---\n")
    return out


def _parse_nodes(lines) -> dict:
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


def _pair_coords(nodes, side_minus, side_plus, plane: str, tol=1e-6) -> Tuple[List[int], List[int]]:
    """
    plane in {"X","Y","Z"}:
      - pair XMIN<->XMAX by matching (y,z)
      - pair YMIN<->YMAX by matching (x,z)
      - pair ZMIN<->ZMAX by matching (x,y)
    Returns:
        plus_nodes, minus_nodes
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
    out = [f"*Nset, nset={name}, instance=PART-1\n"]
    for i in range(0, len(node_ids), per_line):
        chunk = node_ids[i:i+per_line]
        out.append(", ".join(str(n) for n in chunk) + "\n")
    return out


def _centroid_of_nodes(nids, nodes):
    xs = ys = zs = 0.0
    n = len(nids)
    for nid in nids:
        x, y, z = nodes[nid]
        xs += x; ys += y; zs += z
    return (xs / n, ys / n, zs / n)


def _round_key(x, y, z, tol):
    return (round(x / tol), round(y / tol), round(z / tol))


def _face_maps(elem_type):
    """
    Returns list of (local_node_indices_for_face, abaqus_face_label)
    """
    if elem_type == "C3D4":  # tet4
        return [
            ((0, 1, 2), "S1"),
            ((0, 3, 1), "S2"),
            ((1, 3, 2), "S3"),
            ((2, 3, 0), "S4"),
        ]
    elif elem_type == "C3D8":  # hex8  (must match your node ordering!)
        return [
            ((0, 1, 2, 3), "S1"),
            ((4, 5, 6, 7), "S2"),
            ((0, 4, 5, 1), "S3"),
            ((1, 5, 6, 2), "S4"),
            ((2, 6, 7, 3), "S5"),
            ((3, 7, 4, 0), "S6"),
        ]
    else:
        raise ValueError(f"Unsupported elem_type={elem_type}. Add a face map.")


def _solid_faces_from_surface_centroids(
    solid_elems,
    elem_conn,
    elem_types,
    nodes,
    surface_centroids,
    tol=1e-6,
):
    """
    Match *3D element faces* to a set/list of target surface centroids.

    Parameters
    ----------
    solid_elems : iterable[int]
        3D element IDs (one phase)
    elem_conn : dict[int, tuple[int,...]]
        element_id -> node connectivity
    elem_types : dict[int, str]
        element_id -> Abaqus element type (e.g. "C3D4", "C3D8")
    nodes : dict[int, tuple[float,float,float]]
        node_id -> coordinates
    surface_centroids : iterable[tuple[float,float,float]]
        centroids from your 2D surface mesh
    tol : float
        tolerance for centroid matching

    Returns
    -------
    list[tuple[int,str]]
        list of (solid_element_id, face_label like "S1")
    """

    # hash all target surface centroids
    target = defaultdict(list)
    for c in surface_centroids:
        target[_round_key(c[0], c[1], c[2], tol)].append(c)

    matched = []
    used_targets = set()  # (key, index) pairs

    for eid in solid_elems:
        etype = elem_types[eid]
        faces = _face_maps(etype)

        conn = elem_conn[eid]

        for local_ids, flabel in faces:
            fnodes = [conn[i] for i in local_ids]
            fc = _centroid_of_nodes(fnodes, nodes)

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


def _elem_centroid_2d(eid, elem_conn, nodes):
    conn = elem_conn[eid]
    return _centroid_of_nodes(conn, nodes)


def _reindex_and_rebuild_3d_body(
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
) -> List[str]:
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

    # Nodes
    out.append("*Node\n")
    for nid_old in n_old_sorted:
        nid_new = nid_map[nid_old]
        x, y, z = nodes[nid_old]
        out.append(f"{nid_new}, {x}, {y}, {z}\n")

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


def postprocess_inp(
    inp_path: str | Path,
    out_path: str | Path,
    name_model: str = "RVE",
    name_phase_a: str = "PHASE-A",
    name_phase_b: str = "PHASE-B",
    load_case: Literal["Tensile-X", "Tensile-Y", "Tensile-Z", "Shear-XY", "Shear-XZ", "Shear-YZ"] = "Tensile-X",
    strain: float = 0.03,
) -> Path:
    """Convert an orphan Abaqus .inp (no *Part/*Assembly) into Part/Assembly."""

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

    surf_a = _solid_faces_from_surface_centroids(
        solid_elems=elem_a_solid,
        elem_conn=elem_conn,
        nodes=nodes,
        surface_centroids=surf_centroids_a,
        elem_types=elem_type,
        tol=1e-6,
    )

    surf_b = _solid_faces_from_surface_centroids(
        solid_elems=elem_b_solid,
        elem_conn=elem_conn,
        nodes=nodes,
        surface_centroids=surf_centroids_b,
        elem_types=elem_type,
        tol=1e-6,
    )

    cleaned_body = _reindex_and_rebuild_3d_body(
        nodes, elem_conn, elem_type, elem_a_solid, elem_b_solid, surf_a, surf_b,
        surf_name_a=name_iface_a,
        surf_name_b=name_iface_b,
        elset_name_a=name_phase_a,
        elset_name_b=name_phase_b,
    )
    out.extend(cleaned_body)

    # --- node ids ---
    nodes = _parse_nodes(cleaned_body)
    nids_a = {nid for e in elem_a for nid in elem_conn[e]}
    nids_b = {nid for e in elem_b for nid in elem_conn[e]}
    nodes_a = {key: val for key, val in nodes.items() if key in nids_a}
    nodes_b = {key: val for key, val in nodes.items() if key in nids_b}

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

    # --- global bbox ---
    bb_min, bb_max = _bbox(nodes)
    xmin, ymin, zmin = bb_min
    xmax, ymax, zmax = bb_max

    # --- reference points (part level) ---
    cx = 0.5 * (bb_min[0] + bb_max[0])
    cy = 0.5 * (bb_min[1] + bb_max[1])
    cz = 0.5 * (bb_min[2] + bb_max[2])
    span = max(
        bb_max[0] - bb_min[0],
        bb_max[1] - bb_min[1],
        bb_max[2] - bb_min[2],
        1.0
    )
    off = 0.20 * span  # 20% outward offset
    max_nid = max(nodes.keys())
    n = max_nid + 1
    rps = [
        (n := n + 1, bb_max[0] + off, cy, cz, "RPX"),
        (n := n + 1, cx, bb_max[1] + off, cz, "RPY"),
        (n := n + 1, cx, cy, bb_max[2] + off, "RPZ"),
    ]
    out.extend(_ref_points(rps))

    used_nsets = []
    for phase_, nodes_ in zip([name_phase_b, name_phase_a], [nodes_b, nodes_a]):
        side_names = ["XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"]
        side_nodes = {k: [] for k in side_names}

        tol = 1e-6
        for n, (x, y, z) in nodes_.items():
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

        boundary = _classify_rve_boundary_nodes(side_nodes)
        for name, n in boundary["corners"].items():
            if len(n):
                out.extend(_format_nset(f"{name}", n))
        for name, n in boundary["edges"].items():
            if len(n):
                out.extend(_format_nset(f"{name}-{phase_}", n))
        for name, n in boundary["faces"].items():
            if len(n):
                out.extend(_format_nset(f"{name}-{phase_}", n))

        # --- pbc face equations ---
        P_XMAX, M_XMIN = _pair_coords(nodes_, boundary["faces"]["F_XMIN"], boundary["faces"]["F_XMAX"], "X")
        P_YMAX, M_YMIN = _pair_coords(nodes_, boundary["faces"]["F_YMIN"], boundary["faces"]["F_YMAX"], "Y")
        P_ZMAX, M_ZMIN = _pair_coords(nodes_, boundary["faces"]["F_ZMIN"], boundary["faces"]["F_ZMAX"], "Z")
        faces = [
            (('XMAX', 'XMIN', 'RPX'), (P_XMAX, M_XMIN)),
            (('YMAX', 'YMIN', 'RPY'), (P_YMAX, M_YMIN)),
            (('ZMAX', 'ZMIN', 'RPZ'), (P_ZMAX, M_ZMIN)),
        ]
        for face in faces:
            face_plus, face_minus, rp = face[0]
            plus_nodes, minus_nodes = face[1]
            for dof in range(1, 4):
                for n_plus, n_minus in zip(plus_nodes, minus_nodes):
                    if n_plus not in used_nsets:
                        out.append(f"*Nset, nset=NS-{n_plus}, instance=PART-1\n")
                        out.append(f"{n_plus},\n")
                        used_nsets.append(n_plus)
                    if n_minus not in used_nsets:
                        out.append(f"*Nset, nset=NS-{n_minus}, instance=PART-1\n")
                        out.append(f"{n_minus},\n")
                        used_nsets.append(n_minus)
                    out.append(f"*Equation\n3\n")
                    out.append(f"NS-{n_plus}, {dof},  1.0\n")
                    out.append(f"NS-{n_minus}, {dof}, -1.0\n")
                    out.append(f"{rp}, {dof}, -1.0\n")

        # --- your pairings (unchanged) ---
        P_YMIN_ZMAX1, M_YMIN_ZMIN1 = _pair_coords(nodes_, boundary["edges"]["E_YMIN_ZMAX"], boundary["edges"]["E_YMIN_ZMIN"], "Z")
        P_YMAX_ZMAX2, M_YMIN_ZMAX2 = _pair_coords(nodes_, boundary["edges"]["E_YMAX_ZMAX"], boundary["edges"]["E_YMIN_ZMAX"], "Y")
        P_YMAX_ZMIN3, M_YMAX_ZMAX3 = _pair_coords(nodes_, boundary["edges"]["E_YMAX_ZMIN"],
                                                  boundary["edges"]["E_YMAX_ZMAX"], "Z")

        P_XMAX_ZMIN5, M_XMIN_ZMIN5 = _pair_coords(nodes_, boundary["edges"]["E_XMAX_ZMIN"],
                                                  boundary["edges"]["E_XMIN_ZMIN"], "X")
        P_XMIN_ZMAX6, M_XMIN_ZMIN6 = _pair_coords(nodes_, boundary["edges"]["E_XMIN_ZMAX"],
                                                  boundary["edges"]["E_XMIN_ZMIN"], "Z")
        P_XMAX_ZMAX7, M_XMIN_ZMAX7 = _pair_coords(nodes_, boundary["edges"]["E_XMAX_ZMAX"],
                                                  boundary["edges"]["E_XMIN_ZMAX"], "X")

        P_XMIN_YMAX9, M_XMIN_YMIN9 = _pair_coords(nodes_, boundary["edges"]["E_XMIN_YMAX"],
                                                  boundary["edges"]["E_XMIN_YMIN"], "Y")
        P_XMAX_YMIN10, M_XMIN_YMIN10 = _pair_coords(nodes_, boundary["edges"]["E_XMAX_YMIN"],
                                                    boundary["edges"]["E_XMIN_YMIN"], "X")
        P_XMAX_YMAX11, M_XMAX_YMIN11 = _pair_coords(nodes_, boundary["edges"]["E_XMAX_YMAX"],
                                                    boundary["edges"]["E_XMAX_YMIN"], "Y")

        edges = [
            (('E_YMIN_ZMAX', 'E_YMIN_ZMIN', 'RPZ'), (P_YMIN_ZMAX1, M_YMIN_ZMIN1)),
            (('E_YMAX_ZMAX', 'E_YMIN_ZMAX', 'RPY'), (P_YMAX_ZMAX2, M_YMIN_ZMAX2)),
            (('E_YMAX_ZMIN', 'E_YMAX_ZMAX', 'RPZ'), (P_YMAX_ZMIN3, M_YMAX_ZMAX3)),

            (('E_XMAX_ZMIN', 'E_XMIN_ZMIN', 'RPX'), (P_XMAX_ZMIN5, M_XMIN_ZMIN5)),
            (('E_XMIN_ZMAX', 'E_XMIN_ZMIN', 'RPZ'), (P_XMIN_ZMAX6, M_XMIN_ZMIN6)),
            (('E_XMAX_ZMAX', 'E_XMIN_ZMAX', 'RPX'), (P_XMAX_ZMAX7, M_XMIN_ZMAX7)),

            (('E_XMIN_YMAX', 'E_XMIN_YMIN', 'RPY'), (P_XMIN_YMAX9, M_XMIN_YMIN9)),
            (('E_XMAX_YMIN', 'E_XMIN_YMIN', 'RPX'), (P_XMAX_YMIN10, M_XMIN_YMIN10)),
            (('E_XMAX_YMAX', 'E_XMAX_YMIN', 'RPY'), (P_XMAX_YMAX11, M_XMAX_YMIN11)),
        ]

        rp_to_dof = {"RPX": 1, "RPY": 2, "RPZ": 3}

        for edge in edges:
            edge_plus, edge_minus, rp = edge[0]
            plus_nodes, minus_nodes = edge[1]

            dof = rp_to_dof[rp]  # <--- only one DOF, no dof loop

            for n_plus, n_minus in zip(plus_nodes, minus_nodes):
                if n_plus not in used_nsets:
                    out.append(f"*Nset, nset=NS-{n_plus}, instance=PART-1\n")
                    out.append(f"{n_plus},\n")
                    used_nsets.append(n_plus)
                if n_minus not in used_nsets:
                    out.append(f"*Nset, nset=NS-{n_minus}, instance=PART-1\n")
                    out.append(f"{n_minus},\n")
                    used_nsets.append(n_minus)

                out.append("*Equation\n3\n")
                out.append(f"NS-{n_plus}, {dof},  1.0\n")
                out.append(f"NS-{n_minus}, {dof}, -1.0\n")
                out.append(f"{rp}, {dof}, -1.0\n")

    # --- pbc corner equations ---
    corners = [
        (('C_XMAX_YMAX_ZMAX', 'C_XMIN_YMIN_ZMIN', 'RPX', 'RPY', 'RPZ'), (1.0, -1.0, -1.0, -1.0, -1.0)),
        (('C_XMIN_YMIN_ZMAX', 'C_XMIN_YMAX_ZMAX', 'RPY'), (1.0, -1.0, 1.0)),
        (('C_XMIN_YMAX_ZMAX', 'C_XMIN_YMAX_ZMIN', 'RPZ'), (1.0, -1.0, -1.0)),
        (('C_XMIN_YMAX_ZMIN', 'C_XMAX_YMAX_ZMIN', 'RPX'), (1.0, -1.0, 1.0)),
        (('C_XMAX_YMAX_ZMIN', 'C_XMAX_YMIN_ZMIN', 'RPY'), (1.0, -1.0, -1.0)),
        (('C_XMAX_YMIN_ZMIN', 'C_XMAX_YMIN_ZMAX', 'RPZ'), (1.0, -1.0, 1.0)),
        (('C_XMAX_YMIN_ZMAX', 'C_XMAX_YMAX_ZMAX', 'RPY'), (1.0, -1.0, 1.0)),
    ]
    for c in corners:
        for dof in range(1, 4):
            out.append(f"*Equation\n{len(c[0])}\n")
            for set_, val in zip(c[0], c[1]):
                out.append(f"{set_}, {dof}, {val}\n")

    out.append("*End Assembly\n\n")

    # --- materials ---
    out.append("** --- materials ---\n")
    out.append(f"*Material, name={name_phase_a}\n")
    out.append("*Density\n")
    out.append("9.5e-10\n")
    out.append("*Elastic\n")
    out.append("1500., 0.36\n")
    out.append(f"*Material, name={name_phase_b}\n")
    out.append("*Density\n")
    out.append("9.5e-10\n")
    out.append("*Elastic\n")
    out.append("350., 0.46\n")
    out.append("** --- end materials ---\n")

    # --- boundary conditions ---
    out.append("*Boundary\n")
    out.append(f"C_XMIN_YMIN_ZMIN, 1, 1\n")
    out.append(f"C_XMIN_YMIN_ZMIN, 2, 2\n")
    out.append(f"C_XMIN_YMIN_ZMIN, 3, 3\n")
    out.append(f"RPZ, 1, 1\n")
    out.append(f"RPX, 2, 2\n")
    out.append(f"RPY, 3, 3\n")

    # --- stepping ---
    out.append(f"*STEP, name={load_case}, nlgeom=YES, inc=10000\n")
    out.append("*Static\n")
    out.append("0.1, 1., 1e-08, 0.1\n")

    # --- boundary conditions ---
    sequence = []
    if load_case == "Tensile-X":
        sequence = [["RPX"]]
        strain = (bb_max[0] - bb_min[0]) * strain
    elif load_case == "Tensile-Y":
        sequence = [["RPY"]]
        strain = (bb_max[1] - bb_min[1]) * strain
    elif load_case == "Tensile-Z":
        sequence = [["RPZ"]]
        strain = (bb_max[2] - bb_min[2]) * strain
    elif load_case == "Shear-XY":
        sequence = [["RPX", "RPY"]]
        strain = (bb_max[0] - bb_min[0]) * strain / 2
    elif load_case == "Shear-XZ":
        sequence = [["RPX", "RPZ"]]
        strain = (bb_max[1] - bb_min[1]) * strain / 2
    elif load_case == "Shear-YZ":
        sequence = [["RPY", "RPZ"]]
        strain = (bb_max[2] - bb_min[2]) * strain / 2

    for analysis in sequence:
        tag = "".join(b[-1] for b in analysis)
        tag = "tensile-" + tag if len(tag) == 1 else "shear-" + tag
        dof = []
        out.append(f"*Boundary\n")
        for a in analysis:
            if a == "RPX":
                dof.append(1)
            elif a == "RPY":
                dof.append(2)
            elif a == "RPZ":
                dof.append(3)
        if tag.startswith("shear"):
            dof = dof[::-1]
        for a, u in zip(analysis, dof):
            out.append(f"{a}, {u}, {u}, {strain}\n")

    out.append("**\n")
    out.append("** OUTPUT (RVE micromechanics, PBC, no RP)\n")
    out.append("**\n")

    # --- Restart ---
    out.append("*Restart, write, frequency=0\n")

    # --- FIELD OUTPUT ---
    out.append("**\n")
    out.append("** FIELD OUTPUT\n")
    out.append("**\n")
    out.append("*Output, field, frequency=20\n")
    out.append("*Node Output\n")
    out.append("U\n")
    out.append("*Element Output\n")
    out.append("S, LE, PEEQ, EVOL\n")

    # --- HISTORY OUTPUT ---
    out.append("**\n")
    out.append("** HISTORY OUTPUT (global checks)\n")
    out.append("**\n")
    out.append("*Output, history, frequency=1\n")
    out.append("*Energy Output\n")
    out.append("ALLIE, ALLSE, ALLWK\n")

    out.append("**\n")

    Path(out_path).write_text("".join(out), encoding="utf-8")

    return out_path


if __name__ == "__main__":
    exit()

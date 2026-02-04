from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Literal

_element_hdr = re.compile(r"^\s*\*ELEMENT\b", re.IGNORECASE)


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
        "C_XMIN_YMIN_ZMIN": ("XMIN", "YMIN", "ZMIN"),
        "C_XMIN_YMIN_ZMAX": ("XMIN", "YMIN", "ZMAX"),
        "C_XMIN_YMAX_ZMAX": ("XMIN", "YMAX", "ZMAX"),
        "C_XMIN_YMAX_ZMIN": ("XMIN", "YMAX", "ZMIN"),
        "C_XMAX_YMIN_ZMIN": ("XMAX", "YMIN", "ZMIN"),
        "C_XMAX_YMIN_ZMAX": ("XMAX", "YMIN", "ZMAX"),
        "C_XMAX_YMAX_ZMAX": ("XMAX", "YMAX", "ZMAX"),
        "C_XMAX_YMAX_ZMIN": ("XMAX", "YMAX", "ZMIN"),
    }
    corners = {nm: sorted(S[a] & S[b] & S[c]) for nm, (a, b, c) in corner_defs.items()}
    all_corner_nodes = set().union(*[set(v) for v in corners.values()]) if corners else set()

    # --- 12 edges (pairwise intersections minus corners) ---
    edge_defs = {
        # edges parallel to Z (XY corners)
        "E_XMIN_YMIN": ("XMIN", "YMIN"),
        "E_XMIN_YMAX": ("XMIN", "YMAX"),
        "E_XMAX_YMIN": ("XMAX", "YMIN"),
        "E_XMAX_YMAX": ("XMAX", "YMAX"),
        # edges parallel to Y (XZ corners)
        "E_XMIN_ZMIN": ("XMIN", "ZMIN"),
        "E_XMIN_ZMAX": ("XMIN", "ZMAX"),
        "E_XMAX_ZMIN": ("XMAX", "ZMIN"),
        "E_XMAX_ZMAX": ("XMAX", "ZMAX"),
        # edges parallel to X (YZ corners)
        "E_YMIN_ZMIN": ("YMIN", "ZMIN"),
        "E_YMIN_ZMAX": ("YMIN", "ZMAX"),
        "E_YMAX_ZMIN": ("YMAX", "ZMIN"),
        "E_YMAX_ZMAX": ("YMAX", "ZMAX"),
    }
    edges = {}
    for nm, (a, b) in edge_defs.items():
        e = (S[a] & S[b]) - all_corner_nodes
        edges[nm] = sorted(e)

    # convenience: for each face, which 4 edges belong to it?
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
        # corners that lie on this face are just (face set ∩ all_corner_nodes)
        face_corner_nodes = S[face] & all_corner_nodes
        interior = S[face] - face_edges_nodes - face_corner_nodes
        face_interior[face] = sorted(interior)

    return {
        "corners": corners,
        "edges": edges,
        "face_interior": face_interior,
    }


def _simplify_elements_merge_by_type(lines: List[str]) -> List[str]:
    """
    Keep preamble and postamble identical, but replace the entire *ELEMENT section with
    one *ELEMENT header per TYPE (first-seen order), and strip ELSET=... from headers.
    """
    # Find first *ELEMENT line
    first_elem = None
    for i, line in enumerate(lines):
        if _element_hdr.match(line):
            first_elem = i
            break
    if first_elem is None:
        return lines[:]  # no elements found

    # Collect all element blocks, and find where element section ends
    elem_data_by_type: "OrderedDict[str, List[str]]" = OrderedDict()
    post_start = None

    i = first_elem
    while i < len(lines):
        line = lines[i]
        if _element_hdr.match(line):
            # Parse TYPE= from header (default if missing)
            # Also strip ELSET=... by ignoring it entirely.
            hdr = line.strip()
            tokens = [t.strip() for t in hdr.split(",")]
            etype = None
            for t in tokens:
                if t.upper().startswith("TYPE="):
                    etype = t.split("=", 1)[1].strip()
                    break
            if etype is None:
                etype = "UNKNOWN"

            elem_data_by_type.setdefault(etype, [])

            # Copy following connectivity lines until next keyword line
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("*"):
                elem_data_by_type[etype].append(lines[i])
                i += 1
            continue

        # If we’ve started element parsing and hit a non-*ELEMENT keyword, element section ends
        if line.lstrip().startswith("*"):
            post_start = i
            break

        # Shouldn't happen normally (element data must be under a header), but just in case:
        i += 1

    if post_start is None:
        post_start = len(lines)

    preamble = lines[:first_elem]
    postamble = lines[post_start:]

    out: List[str] = []
    out.extend(preamble)

    for etype, data_lines in elem_data_by_type.items():
        out.append(f"*ELEMENT, TYPE={etype}\n")
        out.extend(data_lines)

    out.extend(postamble)
    return out


def _bbox(nodes: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    zs = [p[2] for p in nodes.values()]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _format_refpoints(rps: list) -> list[str]:
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


def _parse_elements(lines) -> Tuple[dict, dict]:
    """
    Returns:
      elem_conn: dict[eid] -> list[node_ids]
      elem_dim_hint: dict[eid] -> "2D" or "3D" guessed from current *ELEMENT header TYPE=
    We only need 2D elements to turn face elsets into node sets.
    """
    elem_conn = {}
    elem_is2d = {}
    i = 0
    while i < len(lines):
        ln = lines[i].lstrip()
        if ln.upper().startswith("*ELEMENT"):
            # heuristic: surface elements usually TYPE=S3, S4, STRI3, etc.
            m = re.search(r"TYPE\s*=\s*([^,\s]+)", ln, flags=re.IGNORECASE)
            etype = (m.group(1).strip().upper() if m else "")
            # quick heuristic: if it starts with 'S' it's likely a surface element
            current_is2d = etype.startswith("S") or "STRI" in etype or "SFM" in etype
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("*"):
                s = lines[i].strip()
                if s and not s.startswith("**"):
                    parts = [p.strip() for p in s.split(",") if p.strip()]
                    eid = int(parts[0])
                    conn = [int(p) for p in parts[1:]]
                    elem_conn[eid] = conn
                    elem_is2d[eid] = bool(current_is2d)
                i += 1
            continue
        i += 1
    if not elem_conn:
        raise RuntimeError("No *ELEMENT blocks found.")
    return elem_conn, elem_is2d


def _parse_elset(lines, elset_name) -> list:
    """
    Reads an *ELSET, ELSET=<name> block and returns the list of element IDs.
    Handles comma-separated lists over multiple lines.
    """
    target = elset_name.lower()
    i = 0
    ids = []
    while i < len(lines):
        ln = lines[i]
        if ln.lstrip().startswith("*") and not ln.lstrip().startswith("**"):
            if ln.lstrip().upper().startswith("*ELSET"):
                m = re.search(r"\bELSET\s*=\s*([^,\n\r]+)", ln, flags=re.IGNORECASE)
                if m and m.group(1).strip().strip('"').strip("'").lower() == target:
                    i += 1
                    while i < len(lines) and not lines[i].lstrip().startswith("*"):
                        s = lines[i].strip()
                        if s and not s.startswith("**"):
                            parts = [p.strip() for p in s.split(",") if p.strip()]
                            ids.extend(int(p) for p in parts)
                        i += 1
                    return ids
        i += 1
    return []


def _nodeset_from_face_elset(face_eids, elem_conn, elem_is2d):
    nset = set()
    for eid in face_eids:
        conn = elem_conn.get(eid)
        if not conn:
            continue
        # prefer explicit flag, but also accept tri/quad connectivity as “surface”
        if elem_is2d.get(eid, False) or len(conn) in (3, 4):
            nset.update(conn)
    return sorted(nset)


def _pair_coords(
    nodes,
    side_minus,
    side_plus,
    plane: str,
    tol=1e-6
) -> Tuple[List[int], List[int]]:
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

    if missing:
        # not fatal, but usually indicates tol too tight or non-matching node sets
        pass

    return plus_nodes, minus_nodes


def _format_nset(name, node_ids, per_line=16):
    out = [f"*Nset, nset={name}, instance=PART-1\n"]
    for i in range(0, len(node_ids), per_line):
        chunk = node_ids[i:i+per_line]
        out.append(", ".join(str(n) for n in chunk) + "\n")
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
    cleaned_body = _simplify_elements_merge_by_type(body)

    # ---- body ----
    out: List[str] = []
    out.extend(preamble)
    out.append(f"*Part, name={name_model}\n")
    out.extend(cleaned_body)

    # --- section assignments on PART level ---
    out.append("** --- section assignments ---\n")
    out.append(f"*Solid Section, elset={name_phase_a}, material={name_phase_a}\n")
    out.append(",\n")
    out.append(f"*Solid Section, elset={name_phase_b}, material={name_phase_b}\n")
    out.append(",\n")
    out.append("** --- end section assignments ---\n")

    # ---- define Abaqus surfaces for surface-based cohesive ----
    # surfaces defined at PART level -> referenced in Assembly as: <instance_name>.SA / <instance_name>.SB
    out.append("** --- interphase surfaces ---\n")
    out.append("*Surface, type=ELEMENT, name=SA\n")
    out.append(f"{name_iface_a}, S1\n")
    out.append("*Surface, type=ELEMENT, name=SB\n")
    out.append(f"{name_iface_b}, S1\n")
    out.append("** --- end interphase surfaces ---\n")

    out.append("*End Part\n\n")

    # --- ASSEMBLY block ---
    out.append("*Assembly, name=Assembly\n")
    out.append(f"*Instance, name={name_instance}, part={name_model}\n")
    out.append("*End Instance\n")

    # --- node sets ---
    side_names = ["XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"]
    nodes = _parse_nodes(body)
    elem_conn, elem_is2d = _parse_elements(body)
    face_elsets = {nm: _parse_elset(body, nm) for nm in side_names}
    out.append("** --- RVE corners/edges/faces ---\n")
    side_nodes = {}
    for nm in side_names:
        side_nodes[nm] = _nodeset_from_face_elset(face_elsets[nm], elem_conn, elem_is2d)
        out.extend(_format_nset(f"{nm}_N", side_nodes[nm]))
    boundary = _classify_rve_boundary_nodes(side_nodes)
    for name, nids in boundary["corners"].items():
        out.extend(_format_nset(name, nids))
    for name, nids in boundary["edges"].items():
        out.extend(_format_nset(name, nids))
    for face, nids in boundary["face_interior"].items():
        out.extend(_format_nset(f"{face}_INT", nids))
    out.append("** --- end RVE corners/edges/face interiors ---\n")

    # --- reference points (part level) ---
    (bb_min, bb_max) = _bbox(nodes)
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
    nid = max_nid + 1
    rps = [
        (nid := nid + 1, bb_max[0] + off, cy, cz, "RPX"),
        (nid := nid + 1, cx, bb_max[1] + off, cz, "RPY"),
        (nid := nid + 1, cx, cy, bb_max[2] + off, "RPZ"),
    ]
    rps_str = _format_refpoints(rps)
    out.extend(rps_str)

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
            for set, val in zip(c[0], c[1]):
                out.append(f"{set}, {dof}, {val}\n")

    # --- pbc edge equations ---
    YMIN_ZMAX0, YMIN_ZMIN0 = _pair_coords(nodes, boundary["edges"]["E_YMIN_ZMAX"], boundary["edges"]["E_YMIN_ZMIN"], "Z")
    YMAX_ZMAX1, YMIN_ZMAX1 = _pair_coords(nodes, boundary["edges"]["E_YMAX_ZMAX"], boundary["edges"]["E_YMIN_ZMAX"], "Y")
    YMAX_ZMIN2, YMAX_ZMAX2 = _pair_coords(nodes, boundary["edges"]["E_YMAX_ZMIN"], boundary["edges"]["E_YMAX_ZMAX"], "Z")
    XMAX_ZMIN3, XMIN_ZMIN3 = _pair_coords(nodes, boundary["edges"]["E_XMAX_ZMIN"], boundary["edges"]["E_XMIN_ZMIN"], "X")
    XMIN_ZMAX4, XMIN_ZMIN4 = _pair_coords(nodes, boundary["edges"]["E_XMIN_ZMAX"], boundary["edges"]["E_XMIN_ZMIN"], "Z")
    XMAX_ZMAX5, XMIN_ZMAX5 = _pair_coords(nodes, boundary["edges"]["E_XMAX_ZMAX"], boundary["edges"]["E_XMIN_ZMAX"], "X")
    XMIN_YMAX6, XMIN_YMIN6 = _pair_coords(nodes, boundary["edges"]["E_XMIN_YMAX"], boundary["edges"]["E_XMIN_YMIN"], "Y")
    XMAX_YMIN7, XMIN_YMIN7 = _pair_coords(nodes, boundary["edges"]["E_XMAX_YMIN"], boundary["edges"]["E_XMIN_YMIN"], "X")
    XMAX_YMAX8, XMAX_YMIN8 = _pair_coords(nodes, boundary["edges"]["E_XMAX_YMAX"], boundary["edges"]["E_XMAX_YMIN"], "Y")
    edges = [
        (('E_YMIN_ZMAX', 'E_YMIN_ZMIN', 'RPZ'), (YMIN_ZMAX0, YMIN_ZMIN0)),
        (('E_YMAX_ZMAX', 'E_YMIN_ZMAX', 'RPY'), (YMAX_ZMAX1, YMIN_ZMAX1)),
        (('E_YMAX_ZMIN', 'E_YMAX_ZMAX', 'RPZ'), (YMAX_ZMIN2, YMAX_ZMAX2)),
        (('E_XMAX_ZMIN', 'E_XMIN_ZMIN', 'RPX'), (XMAX_ZMIN3, XMIN_ZMIN3)),
        (('E_XMIN_ZMAX', 'E_XMIN_ZMIN', 'RPZ'), (XMIN_ZMAX4, XMIN_ZMIN4)),
        (('E_XMAX_ZMAX', 'E_XMIN_ZMAX', 'RPX'), (XMAX_ZMAX5, XMIN_ZMAX5)),
        (('E_XMIN_YMAX', 'E_XMIN_YMIN', 'RPY'), (XMIN_YMAX6, XMIN_YMIN6)),
        (('E_XMAX_YMIN', 'E_XMIN_YMIN', 'RPX'), (XMAX_YMIN7, XMIN_YMIN7)),
        (('E_XMAX_YMAX', 'E_XMAX_YMIN', 'RPY'), (XMAX_YMAX8, XMAX_YMIN8)),
    ]
    used_nsets = []
    for edge in edges:
        edge_plus, edge_minus, rp = edge[0]
        plus_nodes, minus_nodes = edge[1]
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
                out.append("*Equation\n3\n")
                out.append(f"NS-{n_plus}, {dof},  1.0\n")
                out.append(f"NS-{n_minus}, {dof}, -1.0\n")
                out.append(f"{rp}, {dof}, -1.0\n")

    # --- pbc face equations ---
    XMAX, XMIN = _pair_coords(nodes, boundary["face_interior"]["XMIN"], boundary["face_interior"]["XMAX"], "X")
    YMAX, YMIN = _pair_coords(nodes, boundary["face_interior"]["YMIN"], boundary["face_interior"]["YMAX"], "Y")
    ZMAX, ZMIN = _pair_coords(nodes, boundary["face_interior"]["ZMIN"], boundary["face_interior"]["ZMAX"], "Z")
    faces = [
        (('XMAX', 'XMIN', 'RPX'), (XMAX, XMIN)),
        (('YMAX', 'YMIN', 'RPY'), (YMAX, YMIN)),
        (('ZMAX', 'ZMIN', 'RPZ'), (ZMAX, ZMIN)),
    ]
    face_jump_dof = {"XMAX": 1, "YMAX": 2, "ZMAX": 3}
    for face in faces:
        face_plus, face_minus, rp = face[0]
        plus_nodes, minus_nodes = face[1]
        jump_dof = face_jump_dof[face_plus]
        for dof in (1, 2, 3):
            for n_plus, n_minus in zip(plus_nodes, minus_nodes):
                if n_plus not in used_nsets:
                    out.append(f"*Nset, nset=NS-{n_plus}, instance=PART-1\n")
                    out.append(f"{n_plus},\n")
                    used_nsets.append(n_plus)
                if n_minus not in used_nsets:
                    out.append(f"*Nset, nset=NS-{n_minus}, instance=PART-1\n")
                    out.append(f"{n_minus},\n")
                    used_nsets.append(n_minus)
                if dof == jump_dof:
                    out.append("*Equation\n3\n")
                    out.append(f"NS-{n_plus}, {dof},  1.0\n")
                    out.append(f"NS-{n_minus}, {dof}, -1.0\n")
                    out.append(f"{rp}, {dof}, -1.0\n")
                else:
                    out.append("*Equation\n2\n")
                    out.append(f"NS-{n_plus}, {dof},  1.0\n")
                    out.append(f"NS-{n_minus}, {dof}, -1.0\n")

    out.append("*End Assembly\n\n")

    # --- materials ---
    out.append("** --- materials ---\n")
    out.append(f"*Material, name={name_phase_a}\n")
    out.append("*Density\n")
    out.append("9.5e-10,\n")
    out.append("*Elastic\n")
    out.append("1500, 0.36\n")
    out.append(f"*Material, name={name_phase_b}\n")
    out.append("*Density\n")
    out.append("9.5e-10,\n")
    out.append("*Elastic\n")
    out.append("350, 0.46\n")
    out.append("** --- end materials ---\n\n")

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
    elif load_case == "Tensile-Y":
        sequence = [["RPY"]]
    elif load_case == "Tensile-Z":
        sequence = [["RPZ"]]
    elif load_case == "Shear-XY":
        sequence = [["RPX", "RPY"]]
    elif load_case == "Shear-XZ":
        sequence = [["RPX", "RPZ"]]
    elif load_case == "Shear-YZ":
        sequence = [["RPY", "RPZ"]]

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
            out.append(f"{a}, {u}, {u}, {strain / len(analysis)}\n")

    out.append("**\n")
    out.append("** OUTPUT\n")
    out.append("**\n")
    out.append("* Restart, write, frequency = 0\n")
    out.append("**\n")
    out.append("** FIELD OUTPUT: F-Output-1\n")
    out.append("**\n")
    out.append("*Output, field\n")
    out.append("*Node Output\n")
    out.append("RF, U\n")
    out.append("*Element Output, direction=YES\n")
    out.append("EVOL, LE, PE, PEEQ, PEMAG, S\n")
    out.append("**\n")
    out.append("** HISTORY OUTPUT: H-Output-1\n")
    out.append("**\n")
    out.append("*Output, history, variable=PRESELECT\n")
    out.append("**\n")

    Path(out_path).write_text("".join(out), encoding="utf-8")

    return out_path


if __name__ == "__main__":
    exit()

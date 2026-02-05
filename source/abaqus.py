from __future__ import annotations

import re
from collections import OrderedDict
from collections import defaultdict
from pathlib import Path
from typing import Dict, Callable, DefaultDict
from typing import List, Tuple, Literal

_element_hdr = re.compile(r"^\s*\*ELEMENT\b", re.IGNORECASE)


def _merge_duplicate_nodes(
    nodes: Dict[int, Tuple[float, float, float]],
    elem_conn: Dict[int, List[int]],
    tol: float = 1e-9,
    *,
    verbose: bool = True,
    sample: int = 10,
) -> tuple[Dict[int, Tuple[float, float, float]], Dict[int, List[int]]]:
    """
    Merge coincident nodes (same coordinates within tol).

    - Keeps the first-seen node id for each coordinate key
    - Remaps all other node ids to that kept id
    - Updates element connectivity accordingly

    Returns:
      (new_nodes, new_elem_conn)
    """
    # Map quantized coordinate -> kept node id
    key_to_keep: Dict[tuple[int, int, int], int] = {}
    # Remap old node id -> kept node id
    remap: Dict[int, int] = {}

    merged_into = defaultdict(list)  # kept_id -> [merged_ids...]

    for nid, (x, y, z) in nodes.items():
        key = (round(x / tol), round(y / tol), round(z / tol))
        keep = key_to_keep.get(key)
        if keep is None:
            key_to_keep[key] = nid
            remap[nid] = nid
        else:
            remap[nid] = keep
            merged_into[keep].append(nid)

    n_merged = sum(len(v) for v in merged_into.values())
    if verbose and n_merged:
        # show a small sample of merges for sanity
        examples = []
        for keep, mids in list(merged_into.items())[:sample]:
            examples.append((keep, mids[:sample], nodes[keep]))
        print(
            f"[merge_duplicate_nodes] merged {n_merged} duplicate nodes "
            f"into {len(merged_into)} kept nodes (tol={tol}). "
            f"Examples (keep_id, merged_ids, coords): {examples}"
        )
    elif verbose:
        print(f"[merge_duplicate_nodes] no duplicate nodes found (tol={tol}).")

    # Build new node dict: only keep representative ids
    kept_ids = set(remap.values())
    new_nodes = {nid: nodes[nid] for nid in kept_ids}

    # Remap connectivity
    new_elem_conn: Dict[int, List[int]] = {}
    for eid, conn in elem_conn.items():
        new_elem_conn[eid] = [remap[n] for n in conn]

    return new_nodes, new_elem_conn


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

    return {"corners": corners, "edges": edges, "face_interior": face_interior}


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
        if not _is_3d_continuum(etype):
            continue
        out.append(f"*ELEMENT, TYPE={etype}\n")
        out.extend(data_lines)

    out.extend(postamble)
    return out


def _rewrite_3d_elements_from_conn(body_lines: List[str],
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



def _bbox_subset(nodes: dict, nids: set[int]):
    xs = [nodes[n][0] for n in nids]
    ys = [nodes[n][1] for n in nids]
    zs = [nodes[n][2] for n in nids]
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
    if not elem_conn:
        raise RuntimeError("No *ELEMENT blocks found.")
    return elem_conn, elem_type


def _nodes_used_by_3d_elems(elem_conn, elem_type):
    n = set()
    for eid, conn in elem_conn.items():
        if _is_3d_continuum(elem_type.get(eid, "")):
            n.update(conn)
    return n


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


def _pair_coords(
    nodes: Dict[int, Tuple[float, float, float]],
    side_minus: List[int],
    side_plus: List[int],
    plane: str,
    tol: float = 1e-3,
    *,
    name: str = "",
    strict: bool = True,
    sample: int = 12,
) -> Tuple[List[int], List[int]]:
    """
    Robust pairing of nodes between two periodic sides.

    plane in {"X","Y","Z"} defines the *jump direction* (which side differs):
      - plane="X": pairing XMIN<->XMAX by matching (y,z)
      - plane="Y": pairing YMIN<->YMAX by matching (x,z)
      - plane="Z": pairing ZMIN<->ZMAX by matching (x,y)

    Args:
      nodes: nid -> (x,y,z)
      side_minus: node ids on the "minus" side (e.g. XMIN)
      side_plus:  node ids on the "plus"  side (e.g. XMAX)
      tol: quantization tolerance for matching
      name: optional label for error messages (e.g. "E_XMIN_ZMIN <-> E_XMAX_ZMIN")
      strict: if True, raise on duplicates or missing matches; if False, returns only matched pairs
      sample: how many nodes/keys to print in diagnostics

    Returns:
      plus_nodes, minus_nodes  (same length)
    """
    plane = plane.upper()
    if plane == "X":
        key: Callable[[int], Tuple[int, int]] = lambda nid: (
            round(nodes[nid][1] / tol),  # y
            round(nodes[nid][2] / tol),  # z
        )
    elif plane == "Y":
        key = lambda nid: (
            round(nodes[nid][0] / tol),  # x
            round(nodes[nid][2] / tol),  # z
        )
    elif plane == "Z":
        key = lambda nid: (
            round(nodes[nid][0] / tol),  # x
            round(nodes[nid][1] / tol),  # y
        )
    else:
        raise ValueError("plane must be 'X', 'Y', or 'Z'")

    label = f"[{name}] " if name else ""

    # Build multimap for minus side (so we can detect duplicates)
    minus_map: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    for nid in side_minus:
        minus_map[key(nid)].append(nid)

    # Detect duplicate keys on minus side
    dup = [(k, v) for k, v in minus_map.items() if len(v) > 1]
    if dup and strict:
        k, v = dup[0]
        coords = [(nid, nodes[nid]) for nid in v[:sample]]
        raise ValueError(
            f"{label}Duplicate minus-side keys for plane='{plane}' (tol={tol}). "
            f"Example key {k} maps to {len(v)} nodes. Sample nid+coords: {coords}. "
            f"This means your matching coordinates are not unique (often wrong plane, "
            f"too-large tol, or non-conforming boundary mesh)."
        )

    # Pair plus nodes to minus nodes
    plus_nodes: List[int] = []
    minus_nodes: List[int] = []
    missing_plus: List[int] = []

    for nidp in side_plus:
        k = key(nidp)
        candidates = minus_map.get(k)
        if not candidates:
            missing_plus.append(nidp)
            continue

        # If strict=False and duplicates exist, just pick one deterministically.
        # If strict=True, duplicates would have raised already.
        nidm = candidates[0]
        plus_nodes.append(nidp)
        minus_nodes.append(nidm)

    if missing_plus and strict:
        coords = [(nid, nodes[nid]) for nid in missing_plus[:sample]]
        raise ValueError(
            f"{label}Missing matches for plane='{plane}' (tol={tol}): "
            f"{len(missing_plus)}/{len(side_plus)} plus-side nodes had no partner. "
            f"Sample nid+coords: {coords}"
        )

    # Final length sanity
    if strict and (len(plus_nodes) != len(side_plus) or len(plus_nodes) != len(minus_nodes)):
        raise ValueError(
            f"{label}Pairing length mismatch for plane='{plane}': "
            f"paired={len(plus_nodes)} plus={len(side_plus)} minus_paired={len(minus_nodes)}"
        )

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

    nodes = _parse_nodes(body)
    elem_conn, elem_type = _parse_elements(body)
    # nodes, elem_conn = _merge_duplicate_nodes(nodes, elem_conn)
    cleaned_body = _rewrite_3d_elements_from_conn(body, elem_conn, elem_type)
    out.extend(cleaned_body)

    # --- section assignments on PART level ---
    out.append("** --- section assignments ---\n")
    out.append(f"*Solid Section, elset={name_phase_a}, material={name_phase_a}\n")
    out.append(f"*Solid Section, elset={name_phase_b}, material={name_phase_b}\n")
    out.append("** --- end section assignments ---\n")

    # ---- define Abaqus surfaces for surface-based cohesive ----
    # surfaces defined at PART level -> referenced in Assembly as: <instance_name>.SA / <instance_name>.SB
    # name_iface_a = f"{name_phase_a}-IF"
    # name_iface_b = f"{name_phase_b}-IF"
    # out.append("** --- interphase surfaces ---\n")
    # out.append("*Surface, type=ELEMENT, name=SA\n")
    # out.append(f"{name_iface_a}, S1\n")
    # out.append("*Surface, type=ELEMENT, name=SB\n")
    # out.append(f"{name_iface_b}, S1\n")
    # out.append("** --- end interphase surfaces ---\n")

    out.append("*End Part\n\n")

    # --- ASSEMBLY block ---
    out.append("*Assembly, name=Assembly\n")
    out.append(f"*Instance, name={name_instance}, part={name_model}\n")
    out.append("*End Instance\n")

    # --- node sets ---
    nodes_3d = _nodes_used_by_3d_elems(elem_conn, elem_type)

    # --- global bbox ---
    (bb_min, bb_max) = _bbox_subset(nodes, nodes_3d)
    xmin, ymin, zmin = bb_min
    xmax, ymax, zmax = bb_max
    tol = 1e-6

    out.append("** --- RVE corners/edges/faces ---\n")
    side_names = ["XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"]
    side_nodes = {k: [] for k in side_names}

    for nid in nodes_3d:
        x, y, z = nodes[nid]
        if abs(x - xmin) < tol:
            side_nodes["XMIN"].append(nid)
        if abs(x - xmax) < tol:
            side_nodes["XMAX"].append(nid)
        if abs(y - ymin) < tol:
            side_nodes["YMIN"].append(nid)
        if abs(y - ymax) < tol:
            side_nodes["YMAX"].append(nid)
        if abs(z - zmin) < tol:
            side_nodes["ZMIN"].append(nid)
        if abs(z - zmax) < tol:
            side_nodes["ZMAX"].append(nid)

    for nm in side_names:
        side_nodes[nm] = sorted(side_nodes[nm])
        out.extend(_format_nset(f"{nm}_N", side_nodes[nm]))

    boundary = _classify_rve_boundary_nodes(side_nodes)
    for name, nids in boundary["corners"].items():
        out.extend(_format_nset(name, nids))
    for name, nids in boundary["edges"].items():
        out.extend(_format_nset(name, nids))
    for name, nids in boundary["face_interior"].items():
        out.extend(_format_nset(name, nids))
    out.append("** --- end RVE corners/edges/face interiors ---\n")

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
    nid = max_nid + 1
    rps = [
        (nid := nid + 1, bb_max[0] + off, cy, cz, "RPX"),
        (nid := nid + 1, cx, bb_max[1] + off, cz, "RPY"),
        (nid := nid + 1, cx, cy, bb_max[2] + off, "RPZ"),
    ]
    rps_str = _ref_points(rps)
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

    # --- pbc face equations ---
    XMAX, XMIN = _pair_coords(nodes, boundary["face_interior"]["F_XMIN"], boundary["face_interior"]["F_XMAX"], "X")
    YMAX, YMIN = _pair_coords(nodes, boundary["face_interior"]["F_YMIN"], boundary["face_interior"]["F_YMAX"], "Y")
    ZMAX, ZMIN = _pair_coords(nodes, boundary["face_interior"]["F_ZMIN"], boundary["face_interior"]["F_ZMAX"], "Z")
    faces = [
        (('XMAX', 'XMIN', 'RPX'), (XMAX, XMIN)),
        (('YMAX', 'YMIN', 'RPY'), (YMAX, YMIN)),
        (('ZMAX', 'ZMIN', 'RPZ'), (ZMAX, ZMIN)),
    ]
    used_nsets = []
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

    XY1Y0, XY0Y0 = _pair_coords(nodes, boundary["edges"]["E_YMIN_ZMAX"], boundary["edges"]["E_YMIN_ZMIN"], "Z")
    XY1Y1, XY1Y0 = _pair_coords(nodes, boundary["edges"]["E_YMAX_ZMAX"], boundary["edges"]["E_YMIN_ZMAX"], "Y")
    XY0Y1, XY1Y1 = _pair_coords(nodes, boundary["edges"]["E_YMAX_ZMIN"], boundary["edges"]["E_YMAX_ZMAX"], "Z")
    XY0X1, XY0X0 = _pair_coords(nodes, boundary["edges"]["E_XMAX_ZMIN"], boundary["edges"]["E_XMIN_ZMIN"], "X")
    XY1X0, XY0X0 = _pair_coords(nodes, boundary["edges"]["E_XMIN_ZMAX"], boundary["edges"]["E_XMIN_ZMIN"], "Z")
    XY1X1, XY1X0 = _pair_coords(nodes, boundary["edges"]["E_XMAX_ZMAX"], boundary["edges"]["E_XMIN_ZMAX"], "X")
    XZ1X0, XZ0X0 = _pair_coords(nodes, boundary["edges"]["E_XMIN_YMAX"], boundary["edges"]["E_XMIN_YMIN"], "Y")
    XZ0X1, XZ0X0 = _pair_coords(nodes, boundary["edges"]["E_XMAX_YMIN"], boundary["edges"]["E_XMIN_YMIN"], "X")
    XZ1X1, XZ0X1 = _pair_coords(nodes, boundary["edges"]["E_XMAX_YMAX"], boundary["edges"]["E_XMAX_YMIN"], "Y")
    edges = [
        (('E_YMIN_ZMAX', 'E_YMIN_ZMIN', 'RPZ'), (XY1Y0, XY0Y0)),
        (('E_YMAX_ZMAX', 'E_YMIN_ZMAX', 'RPY'), (XY1Y1, XY1Y0)),
        (('E_YMAX_ZMIN', 'E_YMAX_ZMAX', 'RPZ'), (XY0Y1, XY1Y1)),
        (('E_XMAX_ZMIN', 'E_XMIN_ZMIN', 'RPX'), (XY0X1, XY0X0)),
        (('E_XMIN_ZMAX', 'E_XMIN_ZMIN', 'RPZ'), (XY1X0, XY0X0)),
        (('E_XMAX_ZMAX', 'E_XMIN_ZMAX', 'RPX'), (XY1X1, XY1X0)),
        (('E_XMIN_YMAX', 'E_XMIN_YMIN', 'RPY'), (XZ1X0, XZ0X0)),
        (('E_XMAX_YMIN', 'E_XMIN_YMIN', 'RPX'), (XZ0X1, XZ0X0)),
        (('E_XMAX_YMAX', 'E_XMAX_YMIN', 'RPY'), (XZ1X1, XZ0X1)),
    ]
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
                if edge_plus == "E_YMAX_ZMIN":
                    out.append(f"*Equation\n3\n")
                    out.append(f"NS-{n_plus}, {dof},  1.0\n")
                    out.append(f"NS-{n_minus}, {dof}, -1.0\n")
                    out.append(f"{rp}, {dof}, 1.0\n")
                else:
                    out.append(f"*Equation\n3\n")
                    out.append(f"NS-{n_plus}, {dof},  1.0\n")
                    out.append(f"NS-{n_minus}, {dof}, -1.0\n")
                    out.append(f"{rp}, {dof}, -1.0\n")

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

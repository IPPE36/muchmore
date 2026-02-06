def nodes_to_face_mapping(elem_type: str):

    elem_type = elem_type.upper()

    # ---------- Linear ----------
    if elem_type == "C3D4":  # tet4
        return [
            ((0, 1, 2), "S1"),
            ((0, 3, 1), "S2"),
            ((1, 3, 2), "S3"),
            ((2, 3, 0), "S4"),
        ]

    if elem_type == "C3D8":  # hex8
        return [
            ((0, 1, 2, 3), "S1"),
            ((4, 5, 6, 7), "S2"),
            ((0, 4, 5, 1), "S3"),
            ((1, 5, 6, 2), "S4"),
            ((2, 6, 7, 3), "S5"),
            ((3, 7, 4, 0), "S6"),
        ]

    if elem_type == "C3D6":  # wedge6 (triangular prism)
        return [
            ((0, 1, 2), "S1"),      # bottom tri
            ((3, 4, 5), "S2"),      # top tri
            ((0, 1, 4, 3), "S3"),   # quad face
            ((1, 2, 5, 4), "S4"),   # quad face
            ((2, 0, 3, 5), "S5"),   # quad face
        ]

    if elem_type == "C3D5":  # pyramid5
        return [
            ((0, 1, 2, 3), "S1"),   # base quad
            ((0, 1, 4), "S2"),      # side tri
            ((1, 2, 4), "S3"),
            ((2, 3, 4), "S4"),
            ((3, 0, 4), "S5"),
        ]

    # ---------- Quadratic ----------
    if elem_type == "C3D10":  # tet10
        # Edge nodes (standard):
        # (0-1)=4, (1-2)=5, (2-0)=6, (0-3)=7, (1-3)=8, (2-3)=9
        return [
            ((0, 1, 2, 4, 5, 6), "S1"),
            ((0, 3, 1, 7, 8, 4), "S2"),
            ((1, 3, 2, 8, 9, 5), "S3"),
            ((2, 3, 0, 9, 7, 6), "S4"),
        ]

    if elem_type in ("C3D20", "C3D20R"):  # hex20
        # Corner nodes: 0..7 (same as C3D8)
        # Mid-edge nodes: 8..19
        # Standard Abaqus edges:
        #  0-1=8,  1-2=9,  2-3=10, 3-0=11,
        #  4-5=12, 5-6=13, 6-7=14, 7-4=15,
        #  0-4=16, 1-5=17, 2-6=18, 3-7=19
        return [
            ((0, 1, 2, 3, 8, 9, 10, 11), "S1"),      # bottom
            ((4, 5, 6, 7, 12, 13, 14, 15), "S2"),    # top
            ((0, 4, 5, 1, 16, 12, 17, 8), "S3"),
            ((1, 5, 6, 2, 17, 13, 18, 9), "S4"),
            ((2, 6, 7, 3, 18, 14, 19, 10), "S5"),
            ((3, 7, 4, 0, 19, 15, 16, 11), "S6"),
        ]

    if elem_type == "C3D15":  # wedge15
        # Corner nodes: 0..5 (same as C3D6)
        # Mid-edge nodes: 6..14
        # Standard Abaqus edges:
        # 0-1=6, 1-2=7, 2-0=8,
        # 3-4=9, 4-5=10,5-3=11,
        # 0-3=12,1-4=13,2-5=14
        return [
            ((0, 1, 2, 6, 7, 8), "S1"),                 # bottom tri6
            ((3, 4, 5, 9, 10, 11), "S2"),               # top tri6
            ((0, 1, 4, 3, 6, 13, 9, 12), "S3"),         # quad8
            ((1, 2, 5, 4, 7, 14, 10, 13), "S4"),        # quad8
            ((2, 0, 3, 5, 8, 12, 11, 14), "S5"),        # quad8
        ]

    if elem_type == "C3D13":  # pyramid13
        # Corner nodes: 0..4 (base 0-1-2-3, apex 4)
        # Mid-edge nodes: 5..12
        # Standard Abaqus edges:
        # 0-1=5, 1-2=6, 2-3=7, 3-0=8,
        # 0-4=9, 1-4=10,2-4=11,3-4=12
        return [
            ((0, 1, 2, 3, 5, 6, 7, 8), "S1"),           # base quad8
            ((0, 1, 4, 5, 10, 9), "S2"),                # tri6
            ((1, 2, 4, 6, 11, 10), "S3"),
            ((2, 3, 4, 7, 12, 11), "S4"),
            ((3, 0, 4, 8, 9, 12), "S5"),
        ]

    raise ValueError(f"Unsupported element type: {elem_type}")


def face_to_nodes_mapping(elem_type: str):

    elem_type = elem_type.upper()

    if elem_type == "C3D4":
        return {
            "S1": (0, 1, 2),
            "S2": (0, 3, 1),
            "S3": (1, 3, 2),
            "S4": (2, 3, 0),
        }

    if elem_type == "C3D8":
        return {
            "S1": (0, 1, 2, 3),
            "S2": (4, 5, 6, 7),
            "S3": (0, 4, 5, 1),
            "S4": (1, 5, 6, 2),
            "S5": (2, 6, 7, 3),
            "S6": (3, 7, 4, 0),
        }

    if elem_type == "C3D6":
        return {
            "S1": (0, 1, 2),
            "S2": (3, 4, 5),
            "S3": (0, 1, 4, 3),
            "S4": (1, 2, 5, 4),
            "S5": (2, 0, 3, 5),
        }

    if elem_type == "C3D5":
        return {
            "S1": (0, 1, 2, 3),
            "S2": (0, 1, 4),
            "S3": (1, 2, 4),
            "S4": (2, 3, 4),
            "S5": (3, 0, 4),
        }

    if elem_type == "C3D10":
        return {
            "S1": (0, 1, 2, 4, 5, 6),
            "S2": (0, 3, 1, 7, 8, 4),
            "S3": (1, 3, 2, 8, 9, 5),
            "S4": (2, 3, 0, 9, 7, 6),
        }

    if elem_type in ("C3D20", "C3D20R"):
        return {
            "S1": (0, 1, 2, 3, 8, 9, 10, 11),
            "S2": (4, 5, 6, 7, 12, 13, 14, 15),
            "S3": (0, 4, 5, 1, 16, 12, 17, 8),
            "S4": (1, 5, 6, 2, 17, 13, 18, 9),
            "S5": (2, 6, 7, 3, 18, 14, 19, 10),
            "S6": (3, 7, 4, 0, 19, 15, 16, 11),
        }

    if elem_type == "C3D15":
        return {
            "S1": (0, 1, 2, 6, 7, 8),
            "S2": (3, 4, 5, 9, 10, 11),
            "S3": (0, 1, 4, 3, 6, 13, 9, 12),
            "S4": (1, 2, 5, 4, 7, 14, 10, 13),
            "S5": (2, 0, 3, 5, 8, 12, 11, 14),
        }

    if elem_type == "C3D13":
        return {
            "S1": (0, 1, 2, 3, 5, 6, 7, 8),
            "S2": (0, 1, 4, 5, 10, 9),
            "S3": (1, 2, 4, 6, 11, 10),
            "S4": (2, 3, 4, 7, 12, 11),
            "S5": (3, 0, 4, 8, 9, 12),
        }

    raise ValueError(f"Unsupported element type: {elem_type}")

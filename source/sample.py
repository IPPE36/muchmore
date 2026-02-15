from __future__ import annotations

import math
from typing import Dict

import numpy as np

from source.poisson import weight_to_volume_fraction


def _shifts_for_periodic_tiling(dim: int, L: float) -> np.ndarray:
    """All {-L,0,+L}^dim shifts."""
    vals = [-L, 0.0, L]
    grids = np.meshgrid(*([vals] * dim), indexing="ij")
    shifts = np.stack([g.ravel() for g in grids], axis=1)
    return shifts  # (3^dim, dim)


def _render_ellipse_periodic(
    field: np.ndarray,
    center_xy: np.ndarray,
    a: float,
    b: float,
    theta: float,
    L: float,
    dx: float,
) -> None:
    """
    Rasterize a rotated ellipse with semi-axes a,b into 'field' with periodic tiling.
    field: (H,W)
    center_xy: in physical coords [0,L)
    a,b: semi-axes in physical units
    theta: radians
    L: domain length
    dx: pixel size (physical)
    """
    H, W = field.shape
    # Precompute rotation
    ct = math.cos(theta)
    st = math.sin(theta)

    # Conservative bounding radius (for bbox); ellipse fits in circle of radius a
    r = max(a, b)

    # periodic copies (9 shifts)
    shifts = _shifts_for_periodic_tiling(2, L)
    for s in shifts:
        c = center_xy + s  # possibly outside [0,L)

        # bbox in pixel coordinates
        x0 = int(math.floor((c[0] - r) / dx))
        x1 = int(math.ceil((c[0] + r) / dx))
        y0 = int(math.floor((c[1] - r) / dx))
        y1 = int(math.ceil((c[1] + r) / dx))

        # Skip if bbox doesn't overlap the primary raster window [0,W)x[0,H)
        if x1 < 0 or x0 >= W or y1 < 0 or y0 >= H:
            continue

        # Clip to array bounds
        x0c, x1c = max(0, x0), min(W - 1, x1)
        y0c, y1c = max(0, y0), min(H - 1, y1)

        xs = (np.arange(x0c, x1c + 1) + 0.5) * dx
        ys = (np.arange(y0c, y1c + 1) + 0.5) * dx
        X, Y = np.meshgrid(xs, ys, indexing="xy")

        # shift to center
        x = X - c[0]
        y = Y - c[1]

        # rotate into ellipse frame
        xp = ct * x + st * y
        yp = -st * x + ct * y

        inside = (xp / a) ** 2 + (yp / b) ** 2 <= 1.0
        field[y0c : y1c + 1, x0c : x1c + 1][inside] = 1


def _render_spheroid_periodic_axis_aligned(
    field: np.ndarray,
    center_xyz: np.ndarray,
    a: float,
    b: float,
    c: float,
    L: float,
    dx: float,
) -> None:
    """
    Rasterize an axis-aligned ellipsoid into 'field' with periodic tiling.
    field: (D,H,W)
    center_xyz: in physical coords [0,L)^3
    a,b,c: semi-axes in physical units (x,y,z)
    """
    D, H, W = field.shape
    r = max(a, b, c)

    shifts = _shifts_for_periodic_tiling(3, L)
    for s in shifts:
        cen = center_xyz + s

        x0 = int(math.floor((cen[0] - r) / dx))
        x1 = int(math.ceil((cen[0] + r) / dx))
        y0 = int(math.floor((cen[1] - r) / dx))
        y1 = int(math.ceil((cen[1] + r) / dx))
        z0 = int(math.floor((cen[2] - r) / dx))
        z1 = int(math.ceil((cen[2] + r) / dx))

        if x1 < 0 or x0 >= W or y1 < 0 or y0 >= H or z1 < 0 or z0 >= D:
            continue

        x0c, x1c = max(0, x0), min(W - 1, x1)
        y0c, y1c = max(0, y0), min(H - 1, y1)
        z0c, z1c = max(0, z0), min(D - 1, z1)

        xs = (np.arange(x0c, x1c + 1) + 0.5) * dx
        ys = (np.arange(y0c, y1c + 1) + 0.5) * dx
        zs = (np.arange(z0c, z1c + 1) + 0.5) * dx

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")  # shapes: (Ny, Nx, Nz)
        # reorder to (Nz, Ny, Nx) later
        x = X - cen[0]
        y = Y - cen[1]
        z = Z - cen[2]

        inside = (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 <= 1.0
        # inside currently (Ny, Nx, Nz) -> transpose to (Nz, Ny, Nx)
        inside = np.transpose(inside, (2, 0, 1))

        field[z0c : z1c + 1, y0c : y1c + 1, x0c : x1c + 1][inside] = 1


def sample_periodic_ellipsoids(
    *,
    pts: np.ndarray,
    out_dim: int,
    out_size: float,
    out_shape: int,
    ref_areas: np.ndarray,
    ref_aspect_ratio: np.ndarray,
    wf_pp: float,
    wf_ps: float,
    rho_pp: float,
    rho_ps: float,
    inclusion_phase: str = "PP",
    spheroid_c_over_a: float = 1.2,
    rng_seed: int = 0,
    random_orientation_2d: bool = True,
    n_iter_vf: int = 6,
    vf_tol: float = 5e-3,
    avoid_overlaps: bool = True,
    n_max_trials: int = 50,
    drop_if_no_fit: bool = True,
) -> Dict[str, object]:
    ...
    rng = np.random.default_rng(rng_seed)
    N = pts.shape[0]

    # target volume fraction from composition (same as before)
    phi_pp, phi_ps = weight_to_volume_fraction(wf_pp, wf_ps, rho_pp, rho_ps)
    phase = inclusion_phase.strip().upper()
    if phase not in ("PP", "PS"):
        raise ValueError('inclusion_phase must be "PP" or "PS".')
    vf_target = float(phi_pp if phase == "PP" else phi_ps)

    sampled_areas = rng.choice(ref_areas, size=N, replace=True)
    sampled_ar = rng.choice(ref_aspect_ratio, size=N, replace=True)

    dx = out_size / float(out_shape)
    scale = 1.0

    L = float(out_size)

    def _min_image_delta(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Minimum-image vector from q -> p on a periodic box of side L."""
        d = p - q
        d -= L * np.round(d / L)
        return d

    def _fits_no_overlap(
        center: np.ndarray,
        r: float,
        accepted_centers: list[np.ndarray],
        accepted_radii: list[float],
    ) -> bool:
        """Hard-core test using bounding circles/spheres with minimum-image convention."""
        for cj, rj in zip(accepted_centers, accepted_radii):
            d = _min_image_delta(center, cj)
            if float(np.dot(d, d)) < (r + rj) ** 2:
                return False
        return True

    # We will store the final per-particle parameters *after* resampling for non-overlap
    placed_params: list[dict] = []  # each: {"A0":..., "ar":..., "theta":..., "r_bound":...}

    def _prepare_particles_for_scale(scale_factor: float) -> None:
        """
        For the given scale_factor, (re)build per-particle parameters, optionally enforcing non-overlap
        via rejection/resampling. Deterministic given rng_seed + call order.
        """
        placed_params.clear()
        accepted_centers: list[np.ndarray] = []
        accepted_radii: list[float] = []

        for i in range(N):
            center = pts[i]

            # Try to find a non-overlapping particle descriptor
            chosen = None
            for trial in range(max(1, int(n_max_trials))):
                A0 = float(sampled_areas[i])
                ar = float(sampled_ar[i])

                if out_dim == 2:
                    # scaled area
                    A = A0 * (scale_factor ** 2)
                    b = math.sqrt(A / (math.pi * ar))
                    a = ar * b
                    theta = rng.uniform(0.0, math.pi) if random_orientation_2d else 0.0
                    r_bound = max(a, b)
                else:
                    # use area -> r_eq, then spheroid a=b=r_eq, c=k*r_eq
                    r_eq = math.sqrt(A0 / math.pi) * scale_factor
                    a = r_eq
                    b = r_eq
                    c = float(spheroid_c_over_a) * r_eq
                    theta = 0.0
                    r_bound = max(a, b, c)

                if not avoid_overlaps:
                    chosen = {"A0": A0, "ar": ar, "theta": theta, "r_bound": r_bound}
                    break

                if _fits_no_overlap(center, r_bound, accepted_centers, accepted_radii):
                    chosen = {"A0": A0, "ar": ar, "theta": theta, "r_bound": r_bound}
                    break

                # Otherwise: resample particle descriptors (area/ar) for the next trial
                sampled_areas[i] = float(rng.choice(ref_areas))
                sampled_ar[i] = float(rng.choice(ref_aspect_ratio))

            if chosen is None:
                if drop_if_no_fit:
                    # Skip this particle entirely at this scale
                    continue
                else:
                    # Place anyway (last computed values, even if overlapping)
                    chosen = {"A0": float(sampled_areas[i]), "ar": float(sampled_ar[i]), "theta": 0.0, "r_bound": r_bound}

            placed_params.append(chosen)
            accepted_centers.append(center)
            accepted_radii.append(float(chosen["r_bound"]))

    def render_with_scale(scale_factor: float) -> np.ndarray:
        # (Re)build particle descriptors for this scale, enforcing non-overlap if requested
        _prepare_particles_for_scale(scale_factor)

        if out_dim == 2:
            field = np.zeros((out_shape, out_shape), dtype=np.uint8)
            for i, p in enumerate(placed_params):
                A = p["A0"] * (scale_factor ** 2)
                ar = p["ar"]
                b = math.sqrt(A / (math.pi * ar))
                a = ar * b
                theta = float(p["theta"])
                _render_ellipse_periodic(field, pts[i], a, b, theta, out_size, dx)
            return field
        else:
            field = np.zeros((out_shape, out_shape, out_shape), dtype=np.uint8)
            k = float(spheroid_c_over_a)
            for i, p in enumerate(placed_params):
                r_eq = math.sqrt(p["A0"] / math.pi) * scale_factor
                a = r_eq
                b = r_eq
                c = k * r_eq
                _render_spheroid_periodic_axis_aligned(field, pts[i], a, b, c, out_size, dx)
            return field

    # Iteratively adjust scale to match target vf (note: with hard-core, vf might saturate)
    field = render_with_scale(scale)
    vf = float(field.mean())

    for _ in range(n_iter_vf):
        if abs(vf - vf_target) <= vf_tol:
            break
        if vf <= 1e-12:
            scale *= 2.0
        else:
            scale *= (vf_target / vf) ** (1.0 / out_dim)
        field = render_with_scale(scale)
        vf = float(field.mean())

    return {
        "field": field,
        "vf_target": vf_target,
        "vf_achieved": vf,
        "scale_factor": scale,
        "sampled_areas": sampled_areas,
        "sampled_aspect_ratio": sampled_ar,
        "notes": [
            "Non-overlap enforced (optionally) via rejection sampling with bounding circles/spheres and minimum-image periodic distance.",
            f"n_max_trials={n_max_trials}, drop_if_no_fit={drop_if_no_fit}.",
            "Because hard-core constraints can saturate packing, vf_target may become unreachable for large particles / dense seeds.",
        ],
    }


if __name__ == "__main__":
    exit()

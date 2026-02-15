from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Union, Iterable
import numpy as np

from source.features import particle_features
import math


@dataclass(frozen=True)
class PoissonRVEStats:
    # 2D stats
    analyzed_area: float                 # [length^2]
    n_particles_2d: int                  # count in 2D field
    area_fraction_AA: float              # A_A (≈ V_V)
    number_density_NA: float             # N_A = n / A [1/length^2]

    # Inferred 3D particle geometry (prolate spheroid: a=a=b, c)
    aspect_ratio_k: float                # k = c/a
    mean_a: float                        # [length]
    mean_c: float                        # [length]
    mean_volume_particle: float          # <V_p> [length^3]

    # 3D Poisson parameters
    volume_fraction_VV: float            # V_V ≈ A_A
    number_density_NV: float             # N_V [1/length^3]  (Poisson intensity λ)
    intensity_lambda: float              # λ = N_V

    # Convenience
    expected_in_cube: Optional[float] = None  # E[N] for a cube of side L if requested


def estimate_poisson_from_2d_ellipses(
    *,
    analyzed_area: float,
    particle_areas: Optional[Iterable[float]] = None,
    major_axes: Iterable[float],
    minor_axes: Iterable[float],
    aspect_ratio_k: float = 1.2,
    enforce_aspect_ratio: bool = True,
    cube_side: Optional[float] = None,
) -> PoissonRVEStats:
    """
    Estimate 3D Poisson intensity (N_V = λ) for ellipsoidal particles from 2D ellipse measurements.

    Inputs
    ------
    analyzed_area:
        Total analyzed 2D area (same length-units^2 as your axes).
    particle_areas:
        Optional: list of particle areas in 2D. If not provided, they are computed as π*(major/2)*(minor/2).
        Providing measured areas is preferred if your segmentation area is more accurate than ellipse fit.
    major_axes, minor_axes:
        Lists/iterables of fitted ellipse major and minor axis lengths (same units as analyzed_area^0.5).
    aspect_ratio_k:
        Mean 3D aspect ratio k = c/a for a prolate spheroid (c is long semi-axis).
    enforce_aspect_ratio:
        If True: infer a from the 2D equivalent-area radius and set c = k*a.
        If False: use 2D semi-axes directly as (a=minor/2, c=major/2) and report the resulting mean k.
    cube_side:
        If provided: compute expected number of particles in cube of side L: E[N]=N_V*L^3.

    Returns
    -------
    PoissonRVEStats with A_A, N_A, inferred <V_p>, and N_V (= λ).

    Notes / Assumptions
    -------------------
    - Uses Delesse: V_V ≈ A_A (requires random section through isotropic medium).
    - The step converting 2D ellipses → 3D volumes is approximate and ignores sectioning bias.
      It’s often OK for a first Poisson RVE, but can be improved via forward-simulation fitting.
    """
    major_axes = list(major_axes)
    minor_axes = list(minor_axes)
    if len(major_axes) != len(minor_axes):
        raise ValueError("major_axes and minor_axes must have the same length.")
    if analyzed_area <= 0:
        raise ValueError("analyzed_area must be > 0.")
    n = len(major_axes)
    if n == 0:
        raise ValueError("No particles provided (empty major_axes/minor_axes).")
    if aspect_ratio_k <= 0:
        raise ValueError("aspect_ratio_k must be > 0.")

    # 2D particle areas
    if particle_areas is None:
        areas = []
        for maj, minr in zip(major_axes, minor_axes):
            if maj <= 0 or minr <= 0:
                raise ValueError("All major/minor axes must be > 0.")
            a2d = 0.5 * minr
            b2d = 0.5 * maj
            areas.append(math.pi * a2d * b2d)
    else:
        areas = list(particle_areas)
        if len(areas) != n:
            raise ValueError("particle_areas must match length of major_axes/minor_axes.")
        if any(a <= 0 for a in areas):
            raise ValueError("All particle_areas must be > 0.")

    area_inclusions = sum(areas)
    AA = area_inclusions / analyzed_area
    NA = n / analyzed_area

    # Infer 3D semi-axes and volumes
    volumes = []
    a_list = []
    c_list = []

    if enforce_aspect_ratio:
        # Use equivalent-area radius from 2D profile: r_eq = sqrt(A/π)
        # Then set spheroid a = r_eq, c = k*a.
        for A in areas:
            a = math.sqrt(A / math.pi)
            c = aspect_ratio_k * a
            V = (4.0 / 3.0) * math.pi * (a**2) * c
            a_list.append(a)
            c_list.append(c)
            volumes.append(V)
        k_used = aspect_ratio_k
    else:
        # Directly map 2D semi-axes to spheroid semi-axes (a=minor/2, c=major/2).
        # This implicitly assumes your 2D ellipse is a “central section” of the 3D ellipsoid.
        # It will generally NOT match the true 3D aspect ratio if the plane is random.
        k_vals = []
        for maj, minr in zip(major_axes, minor_axes):
            a = 0.5 * minr
            c = 0.5 * maj
            if a <= 0 or c <= 0:
                raise ValueError("Invalid axis lengths encountered.")
            V = (4.0 / 3.0) * math.pi * (a**2) * c
            a_list.append(a)
            c_list.append(c)
            volumes.append(V)
            k_vals.append(c / a)
        k_used = sum(k_vals) / len(k_vals)

    mean_a = sum(a_list) / n
    mean_c = sum(c_list) / n
    mean_Vp = sum(volumes) / n

    # Delesse: V_V ≈ A_A
    VV = AA

    # N_V from volume fraction and mean particle volume
    # V_V = N_V * <V_p>  => N_V = V_V / <V_p>
    NV = VV / mean_Vp
    lam = NV

    expected = None
    if cube_side is not None:
        if cube_side <= 0:
            raise ValueError("cube_side must be > 0.")
        expected = NV * (cube_side ** 3)

    return PoissonRVEStats(
        analyzed_area=analyzed_area,
        n_particles_2d=n,
        area_fraction_AA=AA,
        number_density_NA=NA,
        aspect_ratio_k=k_used,
        mean_a=mean_a,
        mean_c=mean_c,
        mean_volume_particle=mean_Vp,
        volume_fraction_VV=VV,
        number_density_NV=NV,
        intensity_lambda=lam,
        expected_in_cube=expected,
    )


def weight_to_volume_fraction(
    w_pp: float,
    w_ps: float,
    rho_pp: float,
    rho_ps: float,
) -> Tuple[float, float]:
    """Return (phi_pp, phi_ps) from weight fractions and densities."""
    if any(v <= 0 for v in (w_pp, w_ps, rho_pp, rho_ps)):
        raise ValueError("Weight fractions and densities must be > 0.")
    s = w_pp + w_ps
    w_pp /= s
    w_ps /= s

    v_pp = w_pp / rho_pp
    v_ps = w_ps / rho_ps
    denom = v_pp + v_ps
    return v_pp / denom, v_ps / denom


def estimate_n_particles_3d(
    ref_image: np.ndarray,
    *,
    physical_spacing: float,
    box_size: Union[float, Tuple[float, float, float]],
    inclusion_value: int = 1,
    min_size: Optional[int] = None,
    aspect_ratio_k: Optional[float] = None,
    # NEW: composition-based override of VV
    composition_wt: Optional[Tuple[float, float]] = None,   # (w_PP, w_PS)
    densities: Tuple[float, float] = (0.90, 1.05),          # (rho_PP, rho_PS) in g/cm^3 or consistent units
    inclusion_phase: str = "PS",                            # "PS" or "PP"
    use_composition_VV: bool = True,
) -> Dict[str, Any]:
    """
    Estimate expected number of particles in a 3D box using a 2D segmented reference image.

    If composition_wt is provided and use_composition_VV=True, uses the composition-derived
    volume fraction V_V instead of the image area fraction A_A.

    Notes:
    - Delesse: V_V ≈ A_A is used only when composition override is not used.
    - Particles modeled as prolate spheroids with c/a = k.
    - 2D equivalent radius treated as spheroid semi-minor axis a (ignores sectioning bias).
    - Spatial distribution assumed Poisson with intensity λ = N_V.
    """
    if ref_image.ndim != 2:
        raise ValueError("ref_image must be a 2D segmented array.")
    if physical_spacing <= 0:
        raise ValueError("physical_spacing must be > 0.")

    # --- VV: either from image (A_A) or from wt% + densities ---
    VV_image = float(np.mean(ref_image == inclusion_value))

    VV_used = VV_image
    VV_source = "image (Delesse: VV≈AA)"

    if composition_wt is not None and use_composition_VV:
        w_pp, w_ps = composition_wt
        rho_pp, rho_ps = densities
        phi_pp, phi_ps = weight_to_volume_fraction(w_pp, w_ps, rho_pp, rho_ps)

        phase = inclusion_phase.strip().upper()
        if phase not in ("PP", "PS"):
            raise ValueError('inclusion_phase must be "PP" or "PS".')

        VV_used = float(phi_ps if phase == "PS" else phi_pp)
        VV_source = "composition (wt%→vol% using densities)"

    # --- Extract 2D particle stats (pixel units) and scale to physical units ---
    stats_px = particle_features(ref_image, min_size=min_size)
    stats = stats_px.apply_physical_scaling(physical_spacing, ndim=2)

    # Decide aspect ratio k
    k = float(aspect_ratio_k) if aspect_ratio_k is not None else float(stats.mean_aspect_ratio)
    if k <= 0:
        raise ValueError("aspect_ratio_k (or inferred mean aspect ratio) must be > 0.")

    # Equivalent radius r_eq is already computed in particle_features for 2D: r = sqrt(area/pi)
    r = np.asarray(stats.radius, dtype=float)
    if r.size == 0:
        raise ValueError("No particles detected after filtering; cannot estimate 3D intensity.")

    # Prolate spheroid: a=a=b=r_eq, c=k*r_eq
    Vp = (4.0 / 3.0) * np.pi * (r ** 3) * k
    mean_Vp = float(np.mean(Vp))

    # Poisson intensity: V_V = N_V * <V_p>  =>  N_V = V_V / <V_p>
    lambda_3d = VV_used / mean_Vp

    # Compute requested 3D volume
    if isinstance(box_size, (int, float)):
        L = float(box_size)
        if L <= 0:
            raise ValueError("box_size must be > 0.")
        volume_3d = L ** 3
        box_dims = (L, L, L)
    else:
        if len(box_size) != 3:
            raise ValueError("box_size tuple must be (Lx, Ly, Lz).")
        Lx, Ly, Lz = map(float, box_size)
        if min(Lx, Ly, Lz) <= 0:
            raise ValueError("All box dimensions must be > 0.")
        volume_3d = Lx * Ly * Lz
        box_dims = (Lx, Ly, Lz)

    expected_N = float(lambda_3d * volume_3d)

    return {
        "box_dims": box_dims,
        "box_volume": volume_3d,
        "VV_image": VV_image,
        "VV_used": VV_used,
        "VV_source": VV_source,
        "aspect_ratio_k": k,
        "mean_particle_volume": mean_Vp,
        "lambda_3d": float(lambda_3d),
        "expected_N": expected_N,
        "stats_2d_physical": stats.to_dict(),
        "assumptions": [
            "If composition override enabled: VV from wt%→vol% using densities; otherwise Delesse VV≈AA.",
            "Particles approximated as prolate spheroids with c/a = k.",
            "2D equivalent radius treated as spheroid semi-minor axis a (ignores sectioning bias).",
            "3D spatial distribution assumed Poisson with intensity λ = N_V.",
        ],
    }


def stats_to_dict(stats: PoissonRVEStats) -> Dict[str, Any]:
    """Convenience helper to serialize results (e.g., to JSON)."""
    return asdict(stats)


if __name__ == "__main__":
    exit()
from typing import Tuple

import gstools as gs
import numpy as np
from PIL import Image
from joblib import Memory
from numpy.fft import fftfreq, rfftfreq, irfftn
from scipy.ndimage import label, gaussian_filter
from skimage import filters

from source.annealing import sample_points_with_nnd_distribution
from source.features import particle_features, ParticleStats
from source.mesh_2D import mesh_2d
from source.mesh_3D import mesh_3d
from source.poisson import estimate_n_particles_3d
from source.sample import sample_periodic_ellipsoids
from source.timer import timed

memory = Memory(".cache", verbose=0)


@timed("Sample field SRF")
@memory.cache
def sample_field_srf(
        dim: int = 2,
        vf_inclusion: float = 0.5,
        shape: int = 128,
        size: float = 30.0,
        len_scale: float = 1.0,
        anis: float = 1.0,
        mode_no: int = 36,
        random_state: int = 1
) -> np.ndarray:

    kernel = gs.covmodel.Exponential(
        dim=dim,
        len_scale=len_scale,
        anis=anis
    )

    if dim == 2:
        Lx, Ly = size, size
        nx, ny = shape, shape
        x = np.linspace(0.0, Lx, nx, endpoint=False)
        y = np.linspace(0.0, Ly, ny, endpoint=False)
        period = (Lx, Ly)
        pos = (x, y)
    else:
        Lx, Ly, Lz = size, size, size
        nx, ny, nz = shape, shape, shape
        x = np.linspace(0.0, Lx, nx, endpoint=False)
        y = np.linspace(0.0, Ly, ny, endpoint=False)
        z = np.linspace(0.0, Lz, nz, endpoint=False)
        period = (Lx, Ly, Lz)
        pos = (x, y, z)

    srf = gs.SRF(
        kernel,
        generator="Fourier",
        period=period,
        mode_no=mode_no,
        seed=random_state,
    )

    field = srf(pos, mesh_type="structured", post_process=False)

    thresh = np.quantile(field, 1 - vf_inclusion)
    field = np.where(field <= thresh, 0, 1)
    return field.astype(np.uint8)


@timed("Sample field GRF")
@memory.cache
def sample_field_grf(
    dim: int = 2,
    vf_inclusion: float = 0.5,
    shape: int = 128,
    size: float = 30.0,
    len_scale: float = 1.0,
    anis: float = 1.0,
    random_state: int = 1,
    postprocess: bool = False,
    return_raw: bool = False,
) -> np.ndarray:

    kernel = gs.covmodel.Gaussian(
        dim=dim,
        len_scale=len_scale,
        anis=anis
    )

    if dim == 2:
        Lx, Ly = size, size
        nx, ny = shape, shape
        ax, ay = float(anis), 1.0
        dx, dy = Lx / nx, Ly / ny

        # wavenumbers
        kx = (2.0 * np.pi) * fftfreq(nx, d=dx)
        ky = (2.0 * np.pi) * fftfreq(ny, d=dy)

        # apply anisotropy as scaling in k-space: kx/ax, ky/ay
        kx2 = (kx[:, None] / ax) ** 2
        ky2 = (ky[None, :] / ay) ** 2
        k = np.sqrt(kx2 + ky2, dtype=np.float32)

        Sk = kernel.spectral_density(np.abs(k)).astype(np.float32, copy=False)

        # Convert continuous PSD to discrete FFT amplitudes
        dkx = 2.0 * np.pi / Lx
        dky = 2.0 * np.pi / Ly
        Pk = Sk * (dkx * dky)

        # ---- complex Gaussian Fourier coefficients ----
        rng = np.random.default_rng(random_state)
        a = rng.standard_normal((nx, ny), dtype=np.float32)
        b = rng.standard_normal((nx, ny), dtype=np.float32)

        amp = np.sqrt(np.maximum(Pk, 0.0) / 2.0, dtype=np.float32)
        F = amp * (a + 1j * b)

        # enforce real coefficients on k_z = 0 and Nyquist planes
        F[..., 0] = (amp[..., 0] * rng.standard_normal(nx, dtype=np.float32)).astype(np.complex64)
        if ny % 2 == 0:
            F[..., -1] = (amp[..., -1] * rng.standard_normal(nx, dtype=np.float32)).astype(np.complex64)

        # ---- inverse FFT ----
        field = irfftn(F, s=(nx, ny)).astype(np.float32, copy=False)

    else:
        Lx, Ly, Lz = size, size, size
        nx, ny, nz = shape, shape, shape
        ax, ay, az = float(anis), 1.0, 1.0
        dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

        # wave numbers
        kx = (2.0 * np.pi) * fftfreq(nx, d=dx)
        ky = (2.0 * np.pi) * fftfreq(ny, d=dy)
        kz = (2.0 * np.pi) * rfftfreq(nz, d=dz)

        # apply anisotropy as scaling in k-space: kx/ax, ky/ay, kz/az
        kx2 = (kx[:, None, None] / ax) ** 2
        ky2 = (ky[None, :, None] / ay) ** 2
        kz2 = (kz[None, None, :] / az) ** 2
        k = (kx2 + ky2 + kz2).astype(np.float32, copy=False)

        Sk = kernel.spectral_density(np.abs(k)).astype(np.float32, copy=False)

        # Convert continuous PSD to discrete FFT amplitudes
        dkx = 2.0 * np.pi / Lx
        dky = 2.0 * np.pi / Ly
        dkz = 2.0 * np.pi / Lz
        Pk = Sk * (dkx * dky * dkz)

        # ---- complex Gaussian Fourier coefficients ----
        rng = np.random.default_rng(random_state)
        a = rng.standard_normal((nx, ny, nz // 2 + 1), dtype=np.float32)
        b = rng.standard_normal((nx, ny, nz // 2 + 1), dtype=np.float32)

        amp = np.sqrt(np.maximum(Pk, 0.0) / 2.0, dtype=np.float32)
        F = amp * (a + 1j * b)

        # enforce real coefficients on k_z = 0 and Nyquist planes
        F[..., 0] = (amp[..., 0] * rng.standard_normal((nx, ny), dtype=np.float32)).astype(np.complex64)
        if nz % 2 == 0:
            F[..., -1] = (amp[..., -1] * rng.standard_normal((nx, ny), dtype=np.float32)).astype(np.complex64)

        # ---- inverse FFT ----
        field = irfftn(F, s=(nx, ny, nz)).astype(np.float32, copy=False)

    import matplotlib.pyplot as plt
    plt.imshow(field)
    plt.savefig("t.png")
    plt.close()

    if postprocess:
        k = size//25
        # field = grey_opening(field, size=(k, k), mode="wrap")
        # field = grey_dilation(field, size=(k, k), mode="wrap")
        # field = np.exp(field)
        field = np.exp(field)
        field = gaussian_filter(field, sigma=k, mode="wrap")

    if return_raw:
        return field.astype(float)

    thresh = np.quantile(field, 1 - vf_inclusion)
    field = np.where(field <= thresh, 0, 1)

    return field.astype(np.uint8)


def label_field(field: np.ndarray, value: int = 0) -> np.ndarray:
    field_labelled, n = label(field == value)
    return np.array(field_labelled).astype(np.uint8)


def load_field(filepath: str) -> np.ndarray:
    return np.load(filepath).astype(np.uint8)


def dump_field(field: np.ndarray, filepath: str) -> None:
    np.save(filepath, field.astype(np.uint8))


def mesh_field(field: np.ndarray, **kwargs):
    if field.ndim == 2:
        mesh_2d(field, **kwargs)
    elif field.ndim == 3:
        mesh_3d(field, **kwargs)
    else:
        raise ValueError("mesh_field requires a 2D or 3D array!")


def segment_field(field: np.ndarray) -> np.ndarray:
    threshold = filters.threshold_otsu(field)
    return field > threshold


@timed("Feature Extraction")
def stats_field(field: np.ndarray, physical_spacing: float = 1.0) -> Tuple[ParticleStats, ParticleStats]:
    stats = particle_features(field)
    stats_mm = stats.apply_physical_scaling(physical_spacing=physical_spacing, ndim=field.ndim)
    return stats, stats_mm


def invert_binary_field(field: np.ndarray) -> np.ndarray:
    return np.where(field == 0, 1, 0)


def load_field_from_png(path: str, threshold: int = None, invert: bool = False,
                        crop_border: bool = False) -> np.ndarray:
    """Load image and convert to grayscale"""
    img = Image.open(path).convert("L")
    arr = np.asarray(img)

    if crop_border:
        # Detect rows/cols that are not completely white (255)
        rows = np.any(arr < 255, axis=1)
        cols = np.any(arr < 255, axis=0)
        if rows.any() and cols.any():  # avoid empty slice
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            arr = arr[rmin:rmax + 1, cmin:cmax + 1]

    if threshold is None:
        threshold = arr.mean()

    binary = (arr > threshold).astype(np.uint8)

    if invert:
        binary = 1 - binary

    return binary


def generate_inclusion_voxels(
    *,
    ndim: int,
    n_particles: int,
    box_size_um,   # (Lx, Ly) or (Lx, Ly, Lz) in microns
    box_size_vox,  # (Nx, Ny) or (Nx, Ny, Nz) in voxels
    mean_particle_area_um2: float,
    mean_particle_aspect_ratio: float,
    # NEW:
    target_vf: float = None,
    size_factors=None,              # sequence of length n_particles, relative sizes
    size_factors_are_relative: bool = True,  # if False, interpreted as absolute areas/volumes in um^2/um^3
    max_attempts: int = 20000,
    minreldistbound: float = 0.1,
    minreldistincl: float = 0.05,
    periodic: bool = True,
    seed: int = None,
    dtype=np.uint8,
):
    """
    Generate a binary ndarray with randomly placed *ellipsoidal* inclusions.
    Output: array with 1=inclusion phase, 0=matrix.

    NEW:
      - Optional per-particle sizing via `size_factors`
      - Optional enforcement of a target area/volume fraction via `target_vf`

    Parameters
    ----------
    mean_particle_area_um2:
        Backward-compatible default size:
          - ndim=2: mean AREA [µm²]
          - ndim=3: mean VOLUME [µm³] (name kept)

    target_vf:
        If provided, scales particle areas/volumes so the requested inclusion
        fraction is achieved (subject to RSA packing feasibility).
          - ndim=2: target area fraction
          - ndim=3: target volume fraction

    size_factors:
        If provided:
          - If size_factors_are_relative=True (default): treated as relative weights.
            Areas/volumes are assigned proportional to these weights.
          - If False: treated as absolute per-particle areas (2D) or volumes (3D),
            in µm² / µm³ respectively.

        If size_factors is None:
          - all particles identical (based on mean_particle_area_um2), unless target_vf
            is provided, in which case all particles are equal-sized but scaled to match target_vf.

    Returns
    -------
    mask : ndarray of shape box_size_vox (ndim), values {0,1}
    info : dict with placed centers/axes (in microns) and voxel size
    """
    if ndim not in (2, 3):
        raise ValueError("ndim must be 2 or 3.")
    if len(box_size_um) != ndim or len(box_size_vox) != ndim:
        raise ValueError("box_size_um and box_size_vox must have length ndim.")
    if n_particles < 0:
        raise ValueError("n_particles must be >= 0.")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0.")
    if mean_particle_aspect_ratio <= 0:
        raise ValueError("mean_particle_aspect_ratio must be > 0.")
    if target_vf is not None and not (0.0 <= target_vf <= 1.0):
        raise ValueError("target_vf must be in [0, 1].")

    rng = np.random.default_rng(seed)

    box_size_um = np.asarray(box_size_um, dtype=float)
    box_size_vox = np.asarray(box_size_vox, dtype=int)

    voxel_um = box_size_um / box_size_vox  # µm/voxel per axis
    AR = float(mean_particle_aspect_ratio)

    # --- decide per-particle areas/volumes in physical units ---
    box_measure = float(np.prod(box_size_um))  # area in 2D, volume in 3D

    if n_particles == 0:
        mask = np.zeros(tuple(box_size_vox.tolist()), dtype=dtype)
        info = {
            "placed": 0,
            "requested": 0,
            "attempts_used": 0,
            "voxel_um": voxel_um,
            "centers_um": np.zeros((0, ndim)),
            "axes_um": np.zeros((0, ndim)),
            "periodic": periodic,
            "target_vf": target_vf,
            "requested_total_measure_um": 0.0,
            "achieved_total_measure_um": 0.0,
        }
        return mask, info

    if size_factors is None:
        # default: identical particles unless scaled by target_vf
        base = float(mean_particle_area_um2)
        measures = np.full(n_particles, base, dtype=float)

        if target_vf is not None:
            total_target = target_vf * box_measure
            measures[:] = total_target / n_particles
    else:
        sf = np.asarray(size_factors, dtype=float)
        available_factors = np.asarray(sf, dtype=float)
        sf = rng.choice(available_factors, size=n_particles, replace=True)

        if size_factors_are_relative:
            if target_vf is None:
                # If relative factors are provided but no target_vf, fall back to mean size scaling
                # so that the *mean* matches mean_particle_area_um2.
                # i.e., measures_i = sf_i * k, choose k so mean(measures)=mean_particle_area_um2
                k = float(mean_particle_area_um2) / float(np.mean(sf))
                measures = sf * k
            else:
                total_target = target_vf * box_measure
                measures = sf / float(np.sum(sf)) * total_target
        else:
            # absolute areas/volumes directly
            measures = sf.copy()
            if target_vf is not None:
                # scale absolute measures to match target_vf while preserving ratios
                total_target = target_vf * box_measure
                measures *= total_target / float(np.sum(measures))

    requested_total_measure = float(np.sum(measures))

    # --- derive per-particle axes from area/volume + aspect ratio ---
    # For prolate shapes:
    # 2D: area = pi*a*b with a/b=AR => b=sqrt(A/(pi*AR)), a=AR*b
    # 3D: volume = 4/3*pi*a*b*c with a/b=AR and b=c => b=(V/((4/3)*pi*AR))^(1/3), a=AR*b, c=b
    axes_list = []
    r_eff_list = []
    bnd_clear_list = []
    inc_clear_list = []

    if ndim == 2:
        for A in measures:
            b = np.sqrt(A / (np.pi * AR))
            a = AR * b
            ax = np.array([a, b], dtype=float)
            r_eff = float(max(a, b))
            axes_list.append(ax)
            r_eff_list.append(r_eff)
            bnd_clear_list.append(r_eff * float(minreldistbound))
            inc_clear_list.append(r_eff * float(minreldistincl))
    else:
        for V in measures:
            b = (V / ((4.0 / 3.0) * np.pi * AR)) ** (1.0 / 3.0)
            a = AR * b
            c = b
            ax = np.array([a, b, c], dtype=float)
            r_eff = float(max(a, b, c))
            axes_list.append(ax)
            r_eff_list.append(r_eff)
            bnd_clear_list.append(r_eff * float(minreldistbound))
            inc_clear_list.append(r_eff * float(minreldistincl))

    axes_list = np.asarray(axes_list, dtype=float)         # (n, ndim)
    r_eff_list = np.asarray(r_eff_list, dtype=float)       # (n,)
    bnd_clear_list = np.asarray(bnd_clear_list, dtype=float)  # (n,)
    inc_clear_list = np.asarray(inc_clear_list, dtype=float)  # (n,)

    # --- periodic distance helper (minimum image convention) ---
    def min_image_delta(d, L):
        return d - L * np.round(d / L)

    # --- placement storage ---
    centers = []     # list of center vectors in microns
    axes_placed = [] # list of semi-axes vectors in microns
    reffs = []       # effective radii for conservative distance tests (microns)
    clears = []      # per-particle inclusion clearance (microns)
    measures_placed = []

    def accept_center(c, r_i, bnd_i, clear_i):
        # boundary check (non-periodic)
        if not periodic:
            pad = r_i + bnd_i
            if np.any(c < pad) or np.any(c > (box_size_um - pad)):
                return False

        # inclusion distance check vs existing (bounding spheres + clearance)
        if centers:
            C = np.vstack(centers)              # (k, ndim)
            R = np.asarray(reffs, dtype=float)  # (k,)
            CL = np.asarray(clears, dtype=float)

            d = C - c[None, :]
            if periodic:
                d = min_image_delta(d, box_size_um[None, :])
            dist = np.linalg.norm(d, axis=1)

            # require dist >= (r_i + clear_i) + (r_j + clear_j)
            min_dist = (r_i + clear_i) + (R + CL)
            if np.any(dist < min_dist):
                return False

        return True

    # --- random sequential placement ---
    placed = 0
    total_attempts = 0

    # You can place larger particles first to improve success rate (RSA heuristic).
    order = np.argsort(-r_eff_list)  # descending by size

    while placed < n_particles and total_attempts < max_attempts:
        total_attempts += 1

        idx = order[placed]  # attempt to place next-largest remaining particle
        c = rng.random(ndim) * box_size_um

        r_i = float(r_eff_list[idx])
        bnd_i = float(bnd_clear_list[idx])
        clear_i = float(inc_clear_list[idx])

        if accept_center(c, r_i, bnd_i, clear_i):
            centers.append(c)
            axes_placed.append(axes_list[idx].copy())
            reffs.append(r_i)
            clears.append(clear_i)
            measures_placed.append(float(measures[idx]))
            placed += 1
        # else: retry

    # --- rasterize to voxels ---
    mask = np.zeros(tuple(box_size_vox.tolist()), dtype=dtype)

    def wrap_indices(idxs, n):
        return np.mod(idxs, n) if periodic else idxs

    achieved_total_measure = float(np.sum(measures_placed)) if measures_placed else 0.0
    achieved_vf = achieved_total_measure / box_measure if box_measure > 0 else 0.0

    for c_um, ax_um in zip(centers, axes_placed):
        c_v = c_um / voxel_um
        ax_v = ax_um / voxel_um

        lo = np.floor(c_v - ax_v).astype(int)
        hi = np.ceil(c_v + ax_v).astype(int)

        ranges = [np.arange(lo[d], hi[d] + 1) for d in range(ndim)]
        grids = np.meshgrid(*ranges, indexing="ij")

        rel = []
        for d in range(ndim):
            g = grids[d].astype(float)
            dv = g - c_v[d]
            if periodic:
                Nv = box_size_vox[d]
                dv = dv - Nv * np.round(dv / Nv)
            rel.append(dv / ax_v[d])

        if ndim == 2:
            inside = (rel[0] ** 2 + rel[1] ** 2) <= 1.0
            I0 = wrap_indices(grids[0], box_size_vox[0])
            I1 = wrap_indices(grids[1], box_size_vox[1])
            if periodic:
                mask[I0, I1] |= inside.astype(dtype)
            else:
                valid = (
                    (grids[0] >= 0) & (grids[0] < box_size_vox[0]) &
                    (grids[1] >= 0) & (grids[1] < box_size_vox[1])
                )
                vv = valid & inside
                mask[grids[0][vv], grids[1][vv]] = 1
        else:
            inside = (rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2) <= 1.0
            I0 = wrap_indices(grids[0], box_size_vox[0])
            I1 = wrap_indices(grids[1], box_size_vox[1])
            I2 = wrap_indices(grids[2], box_size_vox[2])
            if periodic:
                mask[I0, I1, I2] |= inside.astype(dtype)
            else:
                valid = (
                    (grids[0] >= 0) & (grids[0] < box_size_vox[0]) &
                    (grids[1] >= 0) & (grids[1] < box_size_vox[1]) &
                    (grids[2] >= 0) & (grids[2] < box_size_vox[2])
                )
                vv = valid & inside
                mask[grids[0][vv], grids[1][vv], grids[2][vv]] = 1

    info = {
        "placed": placed,
        "requested": n_particles,
        "attempts_used": total_attempts,
        "voxel_um": voxel_um,
        "centers_um": np.array(centers) if centers else np.zeros((0, ndim)),
        "axes_um": np.array(axes_placed) if axes_placed else np.zeros((0, ndim)),
        "periodic": periodic,
        "target_vf": target_vf,
        "requested_total_measure_um": requested_total_measure,  # sum of A_i (2D) or V_i (3D)
        "achieved_total_measure_um": achieved_total_measure,
        "achieved_vf": achieved_vf,
        "size_factors_used": np.asarray(size_factors) if size_factors is not None else None,
        "size_factors_are_relative": size_factors_are_relative,
    }
    return mask, info



if __name__ == "__main__":
    exit()

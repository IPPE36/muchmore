from typing import Tuple

import gstools as gs
import numpy as np
from PIL import Image
from joblib import Memory
from numpy.fft import fftfreq, rfftfreq, irfftn
from scipy.ndimage import label, gaussian_filter
from skimage import filters
from tqdm import tqdm

from source.features import particle_features, ParticleStats
from source.mesh_2D import mesh_2d
from source.mesh_3D import mesh_3d
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
def stats_field(field: np.ndarray, physical_spacing: float = 1.0, inclusion_value: int = 1) -> Tuple[ParticleStats, ParticleStats]:
    stats = particle_features(field, inclusion_value=inclusion_value)
    stats_mm = stats.apply_physical_scaling(physical_spacing=physical_spacing, ndim=field.ndim)
    return stats_mm


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


@timed("Sampling Inclusions RVE")
def generate_inclusion_voxels(
    *,
    ndim: int,
    n_particles: int,
    box_size_um,   # (Lx, Ly) or (Lx, Ly, Lz) in microns
    box_size_vox,  # (Nx, Ny) or (Nx, Ny, Nz) in voxels
    mean_particle_area_um2: float,
    mean_particle_aspect_ratio: float,
    target_vf: float = None,
    size_factors=None,                   # sequence of length n_particles, relative sizes
    size_factors_are_relative: bool = True,
    max_attempts: int = 200000,
    minreldistbound: float = 0.00,
    minreldistincl: float = 0.15,
    periodic: bool = True,
    seed: int = None,
    dtype=np.uint8,
    initial_alpha: float = 0.70,         # place shrunken, then inflate
    inflate_steps: int = 25,
    inflate_sweeps: int = 30,
    inflate_reff_mode: str = "vol",      # "vol" (more permissive) or "rms" (safer)
):
    """
    Generate a binary ndarray with randomly placed *ellipsoidal* inclusions.
    Output: array with 1=inclusion phase, 0=matrix.

    In 3D: places shrunken particles (initial_alpha) then inflates to full size
    with a periodic relaxation pass to improve packing.
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
    if ndim == 3 and not (0.0 < initial_alpha <= 1.0):
        raise ValueError("initial_alpha must be in (0,1].")

    rng = np.random.default_rng(seed)

    box_size_um = np.asarray(box_size_um, dtype=float)
    box_size_vox = np.asarray(box_size_vox, dtype=int)

    voxel_um = box_size_um / box_size_vox
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
            "achieved_vf": 0.0,
        }
        return mask, info

    # --- measures per particle (A in 2D, V in 3D) ---
    if size_factors is None:
        base = float(mean_particle_area_um2)
        measures = np.full(n_particles, base, dtype=float)
        if target_vf is not None:
            total_target = target_vf * box_measure
            measures[:] = total_target / n_particles
    else:
        sf = np.asarray(size_factors, dtype=float)
        sf = rng.choice(sf, size=n_particles, replace=True)

        if size_factors_are_relative:
            if target_vf is None:
                k = float(mean_particle_area_um2) / float(np.mean(sf))
                measures = sf * k
            else:
                total_target = target_vf * box_measure
                measures = sf / float(np.sum(sf)) * total_target
        else:
            measures = sf.copy()
            if target_vf is not None:
                total_target = target_vf * box_measure
                measures *= total_target / float(np.sum(measures))

    requested_total_measure = float(np.sum(measures))

    # --- helpers ---
    def min_image_delta(d, L):
        return d - L * np.round(d / L)

    def r_eff_hybrid(a, b, c, *, voxel_um=None, size_switch_vox=6.0, sharpness=3.0):
        # volume-equivalent radius
        r_vol = (a * b * c) ** (1.0 / 3.0)
        # rms radius
        r_rms = np.sqrt((a*a + b*b + c*c) / 3.0)

        if voxel_um is None:
            x = max(a, b, c)
            t = x / float(size_switch_vox)
        else:
            max_axis_vox = max(a / voxel_um[0], b / voxel_um[1], c / voxel_um[2])
            t = max_axis_vox / float(size_switch_vox)

        w = (t ** sharpness) / (1.0 + t ** sharpness)
        return (1.0 - w) * r_vol + w * r_rms

    def propose_center_void_biased(*, trials=64):
        if not centers:
            return rng.random(ndim) * box_size_um

        C = np.vstack(centers)
        R = np.asarray(reffs, dtype=float) + np.asarray(clears, dtype=float)

        best_c = None
        best_score = -1e30

        for _ in range(trials):
            c = rng.random(ndim) * box_size_um
            d = C - c[None, :]
            if periodic:
                d = min_image_delta(d, box_size_um[None, :])
            dist = np.linalg.norm(d, axis=1)
            score = np.min(dist - R)
            if score > best_score:
                best_score = score
                best_c = c
        return best_c

    def relax_centers_spheres_periodic(C, R, L, *, n_sweeps=25, step=1.0, jitter=1e-9):
        n = C.shape[0]
        if n < 2:
            return 0.0

        max_overlap_seen = 0.0
        for _ in range(n_sweeps):
            moved = False
            idxs = rng.permutation(n)
            for ii in idxs:
                for jj in range(ii + 1, n):
                    d = C[jj] - C[ii]
                    d = min_image_delta(d, L)
                    dist = np.linalg.norm(d) + jitter
                    min_dist = R[ii] + R[jj]
                    overlap = min_dist - dist
                    if overlap > 0:
                        dir_ = d / dist
                        shift = 0.5 * step * overlap * dir_
                        C[ii] -= shift
                        C[jj] += shift
                        C[ii] = np.mod(C[ii], L)
                        C[jj] = np.mod(C[jj], L)
                        moved = True
                        if overlap > max_overlap_seen:
                            max_overlap_seen = overlap
            if not moved:
                break
        return max_overlap_seen

    def inflate_with_relaxation(
            centers_um,
            axes_target_um,
            box_size_um,
            *,
            alpha_start=0.70,
            alpha_end=1.00,
            n_steps=25,
            reff_mode="vol",  # "vol" or "rms"
            n_sweeps=30,
            step=1.0,
            growth_clear_rel=0.0,  # NEW: relative clearance during growth
            growth_constrained_mask=None,  # NEW: bool mask, True=constrained, False=free
    ):
        """
        Inflate axes from alpha_start * target -> alpha_end * target.
        After each inflation step, relax centers to remove overlaps (sphere proxy).
        Optional clearance during growth:
          - growth_clear_rel: adds clearance = growth_clear_rel * r_eff
          - growth_constrained_mask: if given, only applies clearance where mask=True
        """
        centers = np.array(centers_um, dtype=float, copy=True)
        axes_target = np.array(axes_target_um, dtype=float, copy=False)
        L = np.asarray(box_size_um, dtype=float)

        n = centers.shape[0]
        if growth_constrained_mask is None:
            constrained = np.ones(n, dtype=bool)
        else:
            constrained = np.asarray(growth_constrained_mask, dtype=bool)
            if constrained.shape != (n,):
                raise ValueError("growth_constrained_mask must have shape (n_placed,)")

        alphas = np.linspace(alpha_start, alpha_end, n_steps)
        max_ov = 0.0

        progress = tqdm(total=n_steps)
        for a in alphas:
            axes = a * axes_target

            A = axes[:, 0]
            B = axes[:, 1]
            Cc = axes[:, 2]

            if reff_mode == "vol":
                r_eff = (A * B * Cc) ** (1.0 / 3.0)
            elif reff_mode == "rms":
                r_eff = np.sqrt((A * A + B * B + Cc * Cc) / 3.0)
            else:
                raise ValueError("reff_mode must be 'vol' or 'rms'")

            # --- NEW: clearance during growth ---
            # clearance is only applied to constrained particles
            extra = np.zeros_like(r_eff)
            if growth_clear_rel != 0.0:
                extra[constrained] = float(growth_clear_rel) * r_eff[constrained]

            # Effective exclusion radius during relaxation:
            # enforce dist >= (r_i + extra_i) + (r_j + extra_j)
            R = r_eff + extra

            ov = relax_centers_spheres_periodic(
                centers, R, L,
                n_sweeps=n_sweeps,
                step=step,
            )
            max_ov = max(max_ov, ov)
            progress.update()
        progress.close()

        axes_final = alpha_end * axes_target
        return centers, axes_final, max_ov

    # --- derive per-particle axes from measure + aspect ratio ---
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
            r_eff = float(r_eff_hybrid(a, b, c, voxel_um=voxel_um, size_switch_vox=6.0, sharpness=3.0))
            axes_list.append(ax)
            r_eff_list.append(r_eff)
            bnd_clear_list.append(r_eff * float(minreldistbound))
            inc_clear_list.append(r_eff * float(minreldistincl))

    axes_list = np.asarray(axes_list, dtype=float)           # (n, ndim)
    r_eff_list = np.asarray(r_eff_list, dtype=float)         # (n,)
    bnd_clear_list = np.asarray(bnd_clear_list, dtype=float) # (n,)
    inc_clear_list = np.asarray(inc_clear_list, dtype=float) # (n,)

    # --- placement storage ---
    centers = []          # list of centers in microns
    axes_placed = []      # current axes (shrunk during RSA, full after inflate)
    axes_target = []      # full-size target axes (3D inflate uses this)
    reffs = []            # collision proxy radii (match current size used in accept)
    clears = []           # clearance (match current size)
    measures_placed = []  # analytic measure of FULL particle (not shrunk)

    def accept_center(c, r_i, bnd_i, clear_i):
        if not periodic:
            pad = r_i + bnd_i
            if np.any(c < pad) or np.any(c > (box_size_um - pad)):
                return False

        if centers:
            C = np.vstack(centers)
            R = np.asarray(reffs, dtype=float)
            CL = np.asarray(clears, dtype=float)

            d = C - c[None, :]
            if periodic:
                d = min_image_delta(d, box_size_um[None, :])
            dist = np.linalg.norm(d, axis=1)

            min_dist = (r_i + clear_i) + (R + CL)
            if np.any(dist < min_dist):
                return False
        return True

    # --- RSA (active set) ---
    placed = 0
    total_attempts = 0

    order = np.argsort(-r_eff_list)  # big -> small
    remaining = list(order.tolist())
    cursor = 0
    fail_counts = np.zeros(n_particles, dtype=int)

    stall_window = 20000
    no_progress_attempts = 0

    max_attempts_per_particle_early = 50000
    max_attempts_per_particle_late = 15000

    progress = tqdm(total=n_particles)

    while placed < n_particles and total_attempts < max_attempts and remaining:
        total_attempts += 1
        no_progress_attempts += 1

        idx = remaining[cursor]
        max_attempts_per_particle = max_attempts_per_particle_late if (placed > 0.85 * n_particles) else max_attempts_per_particle_early

        # proposal strategy tuned for 128^3
        if placed < 0.80 * n_particles and no_progress_attempts < 5000:
            c = rng.random(ndim) * box_size_um
        elif no_progress_attempts < 20000:
            c = propose_center_void_biased(trials=64)
        else:
            c = propose_center_void_biased(trials=256)

        # IMPORTANT: scale collision proxy + clearance by current alpha (3D)
        if ndim == 3:
            alpha = initial_alpha
        else:
            alpha = 1.0

        r_i = float(alpha * r_eff_list[idx])
        bnd_i = float(alpha * bnd_clear_list[idx])
        clear_i = float(alpha * inc_clear_list[idx])

        if accept_center(c, r_i, bnd_i, clear_i):
            centers.append(c)

            ax_t = axes_list[idx].copy()
            if ndim == 3:
                axes_target.append(ax_t)             # full target
                axes_placed.append(alpha * ax_t)     # current (shrunk)
            else:
                axes_placed.append(ax_t)

            reffs.append(r_i)
            clears.append(clear_i)
            measures_placed.append(float(measures[idx]))  # full analytic measure

            placed += 1
            progress.update(1)

            no_progress_attempts = 0

            remaining.pop(cursor)
            cursor = cursor % len(remaining) if remaining else 0
        else:
            fail_counts[idx] += 1
            if fail_counts[idx] >= max_attempts_per_particle:
                remaining.pop(cursor)
                cursor = cursor % len(remaining) if remaining else 0
            else:
                cursor = (cursor + 1) % len(remaining)

        if no_progress_attempts >= stall_window:
            break

    progress.close()

    # --- post-RSA inflate + relax (3D only) ---
    max_overlap = 0.0
    if ndim == 3 and placed > 0 and initial_alpha < 1.0:
        C0 = np.array(centers, dtype=float)
        AX_target = np.array(axes_target, dtype=float)

        C_new, AX_full, max_overlap = inflate_with_relaxation(
            C0, AX_target, box_size_um,
            alpha_start=0.70,
            alpha_end=1.00,
            n_steps=25,
            reff_mode="vol",
            n_sweeps=30,
            step=1.0,
            growth_clear_rel=0.1,
        )

        centers = [c for c in C_new]
        axes_placed = [ax for ax in AX_full]  # now full-size axes

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
        "requested_total_measure_um": requested_total_measure,
        "achieved_total_measure_um": achieved_total_measure,
        "achieved_vf": achieved_vf,
        "voxel_vf": float(mask.mean()),
        "inflate_initial_alpha": float(initial_alpha) if ndim == 3 else None,
        "inflate_max_overlap_um": float(max_overlap) if ndim == 3 else None,
    }
    return mask, info



if __name__ == "__main__":
    exit()

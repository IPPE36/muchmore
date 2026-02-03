import gstools as gs
import numpy as np
from scipy.ndimage import label
from numpy.fft import fftfreq, rfftfreq, irfftn
from joblib import Memory


memory = Memory(".cache", verbose=0)


def sample_field_srf(
        dim: int = 2,
        vf: float = 0.5,
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
    thresh = np.quantile(field, vf)
    field = np.where(field <= thresh, 0, 1)
    return field.astype(np.uint8)


@memory.cache
def sample_field_grf(
        dim: int = 2,
        vf: float = 0.5,
        shape: int = 128,
        size: float = 30.0,
        len_scale: float = 1.0,
        anis: float = 1.0,
        random_state: int = 1,
) -> np.ndarray:

    kernel = gs.covmodel.Exponential(
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

        # wavenumbers
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

    thresh = np.quantile(field, vf)
    field = np.where(field <= thresh, 0, 1)
    return field.astype(np.uint8)


def label_field(field: np.ndarray, value: int = 0) -> np.ndarray:
    field_labelled, n = label(field == value)
    return np.array(field_labelled).astype(np.uint8)


def load_field(filepath: str) -> np.ndarray:
    return np.load(filepath).astype(np.uint8)


def dump_field(field: np.ndarray, filepath: str) -> None:
    np.save(filepath, field.astype(np.uint8))


if __name__ == "__main__":
    exit()

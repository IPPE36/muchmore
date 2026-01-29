import gstools as gs
import numpy as np
from scipy.ndimage import label


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


def label_field(field: np.ndarray, value: int = 0) -> np.ndarray:
    field_labelled, n = label(field == value)
    return np.array(field_labelled).astype(np.uint8)


def load_field(filepath: str) -> np.ndarray:
    return np.load(filepath).astype(np.uint8)


def dump_field(field: np.ndarray, filepath: str) -> None:
    np.save(filepath, field.astype(np.uint8))


if __name__ == "__main__":
    exit()

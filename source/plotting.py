import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from typing import List

matplotlib.use("agg")


def plot_field(field, filepath: str) -> None:
    assert field.ndim == 2
    plt.imshow(field.T)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    return None


def plot_contours(field, contours, filepath: str) -> None:
    assert field.ndim == 2
    plt.imshow(field.T, origin="lower", aspect="equal")
    for c in contours:
        plt.plot(c[:, 0], c[:, 1], 'r', linewidth=1.5)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    return None


def plot_polygons(field: np.ndarray, polys: List[Polygon], filepath: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    tiled = np.tile(field, (3, 3))
    plt.imshow(tiled, origin="lower", aspect="equal")

    ny, nx = field.shape
    x0, x1 = nx, 2 * nx
    y0, y1 = ny, 2 * ny
    plt.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], color="red", lw=2.5, ls=":")

    # Plot polygons
    for i, poly in enumerate(polys):
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.5, edgecolor="k")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.savefig(filepath, dpi=300)
    return None


if __name__ == "__main__":
    exit()
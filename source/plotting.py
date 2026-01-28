import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


def plot_2d_mesh(field, verts, faces, filepath, *, threshold=0.5) -> None:
    """
    Plot a 2D triangular mesh colored by phase (0/1),
    where phase is determined from the field at triangle centroids.
    """
    assert field.ndim == 2

    # --- determine phase per triangle
    c = verts[faces].mean(axis=1)           # centroids (x,y)
    x = np.clip(np.rint(c[:, 0]).astype(int), 0, field.shape[1] - 1)
    y = np.clip(np.rint(c[:, 1]).astype(int), 0, field.shape[0] - 1)
    face_phase = (field[y, x] >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(6, 6))

    # background (optional)
    ax.imshow(field, origin="lower", aspect="equal", alpha=0.25)

    # colored mesh
    tpc = ax.tripcolor(
        verts[:, 0],
        verts[:, 1],
        faces,
        facecolors=face_phase,
        shading="flat",
        cmap="tab10",
        edgecolors="k",
        linewidth=0.2,
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close(fig)


def plot_surface(verts, faces, values=None, filepath: str = None) -> None:

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=0.9)

    if values is not None:
        face_values = values[faces].mean(axis=1)
        mesh.set_array(face_values)
        mesh.set_cmap("viridis")
        mesh.set_edgecolor("none")
        fig.colorbar(mesh, ax=ax, shrink=0.6, label="field value")
    else:
        mesh.set_facecolor("lightgray")
        mesh.set_edgecolor("none")

    ax.add_collection3d(mesh)

    # Set equal aspect ratio
    min_xyz = verts.min(axis=0)
    max_xyz = verts.max(axis=0)
    ax.set_xlim(min_xyz[0], max_xyz[0])
    ax.set_ylim(min_xyz[1], max_xyz[1])
    ax.set_zlim(min_xyz[2], max_xyz[2])
    ax.set_box_aspect(max_xyz - min_xyz)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    return None


if __name__ == "__main__":
    exit()
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
from scipy.stats import gaussian_kde
from skimage import feature


@dataclass
class ParticleStats:
    n_particles: int
    distance: np.ndarray
    area: Sequence[float]
    aspect_ratio: Sequence[float]
    radius: Sequence[float]

    @property
    def mean_radius(self) -> float:
        return np.mean(self.radius)

    @property
    def std_radius(self) -> float:
        return np.std(self.radius)

    @property
    def mean_area(self) -> float:
        return np.mean(self.area)

    @property
    def std_area(self) -> float:
        return np.std(self.area)

    @property
    def mean_aspect_ratio(self) -> float:
        return np.mean(self.aspect_ratio)

    @property
    def std_aspect_ratio(self) -> float:
        return np.std(self.aspect_ratio)

    def to_dict(self) -> dict:
        """Flatten particle statistics into a serializable dict."""
        return {
            "n_particles": self.n_particles,
            "mean_radius": self.mean_radius,
            "std_radius": self.std_radius,
            "mean_area": self.mean_area,
            "std_area": self.std_area,
            "mean_aspect_ratio": self.mean_aspect_ratio,
            "std_aspect_ratio": self.std_aspect_ratio,
            "radius": np.asarray(self.radius),
            "area": np.asarray(self.area),
            "aspect_ratio": np.asarray(self.aspect_ratio),
            "distance": np.asarray(self.distance),
        }

    def apply_physical_scaling(self, physical_spacing: float, ndim: int):
        """Convert voxel-based quantities into physical units."""
        ps = physical_spacing
        return ParticleStats(
            n_particles=self.n_particles,
            distance=np.asarray(self.distance) * ps,
            radius=np.asarray(self.radius) * ps,
            area=np.asarray(self.area) * ps ** ndim,
            aspect_ratio=self.aspect_ratio,
        )

    def to_inclusion_set(self):
        kde = gaussian_kde(self.radius)

        samples = kde.resample(self.n_particles).ravel()
        samples = samples[samples > 0]  # keep only physical radii

        if samples.size == 0:
            raise ValueError("KDE produced no positive radius samples")

        bins = min(20, samples.size)
        counts, edges = np.histogram(samples, bins=bins)

        centers = 0.5 * (edges[:-1] + edges[1:])

        # build [(radius, amount), ...] with shape (N, 2)
        inclusion_sets = np.column_stack((centers, counts))

        # optional: drop empty bins
        inclusion_sets = inclusion_sets[inclusion_sets[:, 1] > 0]

        return inclusion_sets.tolist()


def chord_length(field: np.ndarray) -> tuple:
    """Calculates chord length of inclusions of segmented image"""
    edges = feature.canny(field)
    boundary_points = np.argwhere(edges)
    chord_lengths = []
    for i in range(boundary_points.shape[0] - 1):
        p1 = boundary_points[i]
        p2 = boundary_points[i + 1]
        chord_lengths.append(euclidean(p1, p2))
    return np.mean(chord_lengths)


def volume_fractions(x: np.ndarray) -> list:
    """Calculates volume fraction of inclusions of segmented image"""
    values, counts = np.unique(x, return_counts=True)
    return [c / x.size for c in counts]


def particle_features(x: np.ndarray, min_size: int = None) -> ParticleStats:
    min_size = 25 if not min_size else min_size
    mask, n = label(x)

    centers, area, aspect_ratio, radius = [], [], [], []
    n_particles = 0

    for i in range(n):
        points = np.array(np.where(mask == i + 1))

        if points.shape[-1] < min_size:
            continue

        n_particles += 1
        s = points.shape[-1]
        area.append(s)

        # --- skip particles touching the border ---
        touches_border = False
        for d in range(x.ndim):
            if (
                    np.any(points[d] == 0) or
                    np.any(points[d] == x.shape[d] - 1)
            ):
                touches_border = True
                break

        if touches_border:
            continue
        # ------------------------------------------

        eigvals, _ = np.linalg.eig(np.cov(points))
        ar = np.max(eigvals) / np.min(eigvals)
        c = points.mean(axis=-1)

        if ar < 0.5 or ar > 2:
            continue

        # representative (equivalent) radius
        if x.ndim == 2:
            r = np.sqrt(s / np.pi)
        elif x.ndim == 3:
            r = (3 * s / (4 * np.pi)) ** (1 / 3)

        aspect_ratio.append(ar)
        radius.append(r)
        centers.append(c)

    center_coords = np.array(centers)
    tree = cKDTree(center_coords)
    dists = tree.query(center_coords, x.ndim)

    return ParticleStats(
        n_particles=n_particles,
        distance=dists[0][:, 1],
        area=area,
        aspect_ratio=aspect_ratio,
        radius=radius,
    )




if __name__ == "__main__":
    exit()

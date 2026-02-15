from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree


ArrayLike = Union[np.ndarray, Iterable[float]]
BoxSize = Union[float, ArrayLike]


def _as_Lvec(boxsize: BoxSize, dim: Optional[int] = None) -> np.ndarray:
    """Convert boxsize into a (dim,) float array."""
    if np.isscalar(boxsize):
        if dim is None:
            raise ValueError("If boxsize is a scalar, you must provide dim=2 or dim=3 (or inferable).")
        return np.full((dim,), float(boxsize), dtype=float)
    L = np.asarray(boxsize, dtype=float).reshape(-1)
    if dim is not None and len(L) != dim:
        raise ValueError(f"boxsize has dim {len(L)} but dim={dim}.")
    return L


def wrap_points(points: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Wrap points into [0, L) in each dimension."""
    return np.mod(points, L)


def min_image_displacement(a: np.ndarray, b: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Minimum-image displacement vector(s) from b to a under periodic boundaries.

    a, b can broadcast to (..., dim). Returns (..., dim).
    """
    d = a - b
    d = d - L * np.round(d / L)
    return d


def periodic_dist(a: np.ndarray, b: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Minimum-image periodic distance(s) between a and b."""
    d = min_image_displacement(a, b, L)
    return np.linalg.norm(d, axis=-1)


def compute_nnd(points: np.ndarray, L: np.ndarray, leafsize: int = 16) -> np.ndarray:
    """
    Compute nearest-neighbor distance for each point in a periodic box using cKDTree(boxsize=L).
    Returns array of shape (n,).
    """
    tree = cKDTree(points, boxsize=L, leafsize=leafsize)
    d, _ = tree.query(points, k=2)  # k=1 is self; k=2 is nearest other
    return d[:, 1]


def match_sorted_nnd_energy(nnd: np.ndarray, target_nnd: np.ndarray) -> float:
    """
    Distribution match energy: mean squared error between sorted current NND and sorted target NND.
    """
    a = np.sort(nnd)
    b = np.sort(target_nnd)
    return float(np.mean((a - b) ** 2))


def repulsion_energy(points: np.ndarray, L: np.ndarray, r_min: float, leafsize: int = 16) -> float:
    """
    Short-range repulsion barrier: penalize any pair closer than r_min.

    Efficiently finds pairs within r_min using cKDTree.query_pairs (supports periodic via boxsize).
    Energy = mean over violating pairs of (r_min - dist)^2. If no violations -> 0.
    """
    if r_min <= 0:
        return 0.0

    tree = cKDTree(points, boxsize=L, leafsize=leafsize)
    pairs = tree.query_pairs(r=r_min, output_type="ndarray")
    if pairs.size == 0:
        return 0.0

    i = pairs[:, 0]
    j = pairs[:, 1]
    d = periodic_dist(points[i], points[j], L)
    viol = np.clip(r_min - d, 0.0, None)
    return float(np.mean(viol ** 2))


@dataclass
class AnnealConfig:
    steps: int = 50_000
    # Move scale is relative to box size: actual stddev per dim is move_scale * L[dim]
    move_scale: float = 0.02
    # Temperature schedule: exponential from T0 to T1
    T0: float = 1.0
    T1: float = 0.01
    # Weights
    repel_weight: float = 25.0
    # Repulsion radius as factor of min(target_nnd)
    r_min_factor: float = 0.85
    # cKDTree parameter
    leafsize: int = 16
    # Random seed
    seed: Optional[int] = 0
    # If True, periodically reduce move_scale as it converges (helps fine-tuning)
    adaptive_stepsize: bool = True
    # How often to print progress if verbose
    log_every: int = 2000


def _temperature(t: int, steps: int, T0: float, T1: float) -> float:
    """Exponential schedule."""
    if steps <= 1:
        return T1
    frac = t / (steps - 1)
    return float(T0 * (T1 / T0) ** frac)


def _total_energy(
    points: np.ndarray,
    L: np.ndarray,
    target_nnd: np.ndarray,
    cfg: AnnealConfig,
    r_min: float,
) -> Tuple[float, float, float]:
    """
    Returns: (E_total, E_match, E_repel)
    """
    nnd = compute_nnd(points, L, leafsize=cfg.leafsize)
    E_match = match_sorted_nnd_energy(nnd, target_nnd)
    E_repel = repulsion_energy(points, L, r_min=r_min, leafsize=cfg.leafsize)
    E_total = E_match + cfg.repel_weight * E_repel
    return E_total, E_match, E_repel


def sample_points_with_nnd_distribution(
    n: int,
    target_nnd: ArrayLike,
    boxsize: BoxSize,
    dim: Optional[int] = None,
    *,
    init_points: Optional[np.ndarray] = None,
    cfg: Optional[AnnealConfig] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Sample n points in a periodic 2D or 3D box so that the NND distribution matches target_nnd.

    Args:
        n: number of points.
        target_nnd: iterable/array of length n containing desired nearest-neighbor distances.
                    This function matches the *distribution* (sorted NNDs), not a per-point identity.
        boxsize: scalar L (same in all dims) or vector-like (Lx, Ly[, Lz]).
        dim: 2 or 3 (required if boxsize is scalar; otherwise inferred from boxsize vector length).
        init_points: optional initial point array shape (n, dim). If None, uniform random init.
        cfg: AnnealConfig.
        verbose: print progress.

    Returns:
        points: (n, dim) array within [0, L) with periodic boundary conditions.
    """
    if cfg is None:
        cfg = AnnealConfig()

    target = np.asarray(target_nnd, dtype=float).reshape(-1)
    rng = np.random.default_rng()
    target = rng.choice(target, size=n, replace=True)

    L = _as_Lvec(boxsize, dim=dim)
    dim = len(L)

    rng = np.random.default_rng(cfg.seed)

    if init_points is None:
        points = rng.random((n, dim)) * L
    else:
        points = np.asarray(init_points, dtype=float)
        if points.shape != (n, dim):
            raise ValueError(f"init_points must have shape {(n, dim)}, got {points.shape}.")
        points = wrap_points(points, L)

    # Repulsion threshold: protects against degeneracy/collisions.
    # If target.min() is zero (or near), use a small fraction of typical spacing scale.
    tmin = float(np.min(target))
    typical_spacing = float(np.sqrt(np.prod(L) / n)) if dim == 2 else float((np.prod(L) / n) ** (1.0 / dim))
    base = tmin if tmin > 1e-12 else 0.25 * typical_spacing
    r_min = cfg.r_min_factor * base

    # Initial energy
    E, Em, Er = _total_energy(points, L, target, cfg, r_min)

    # Annealing
    base_move = cfg.move_scale
    for t in range(cfg.steps):
        T = _temperature(t, cfg.steps, cfg.T0, cfg.T1)

        # Optionally shrink step size late in the run
        if cfg.adaptive_stepsize:
            # Smoothly reduce to 40% of original by the end
            move_scale = base_move * (0.4 + 0.6 * (1.0 - t / max(cfg.steps - 1, 1)))
        else:
            move_scale = base_move

        i = int(rng.integers(0, n))
        old = points[i].copy()

        # Random Gaussian move (scaled to box). Wrap to periodic domain.
        step = rng.normal(size=dim) * (move_scale * L)
        points[i] = (points[i] + step) % L

        E_new, Em_new, Er_new = _total_energy(points, L, target, cfg, r_min)
        dE = E_new - E

        accept = (dE <= 0.0) or (rng.random() < np.exp(-dE / max(T, 1e-12)))
        if accept:
            E, Em, Er = E_new, Em_new, Er_new
        else:
            points[i] = old

        if verbose and (t % cfg.log_every == 0 or t == cfg.steps - 1):
            nnd = compute_nnd(points, L, leafsize=cfg.leafsize)
            msg = (
                f"[{t:6d}/{cfg.steps}] "
                f"E={E:.6g} (match={Em:.6g}, repel={Er:.6g}), "
                f"NND: min={nnd.min():.4g}, mean={nnd.mean():.4g}, med={np.median(nnd):.4g}"
            )
            print(msg)

    return wrap_points(points, L)


def evaluate_nnd_match(points: np.ndarray, target_nnd: ArrayLike, boxsize: BoxSize) -> dict:
    """
    Utility to evaluate how well points match the target NND distribution.
    Returns summary stats + distribution error.
    """
    target = np.asarray(target_nnd, dtype=float).reshape(-1)
    L = _as_Lvec(boxsize, dim=points.shape[1])
    nnd = compute_nnd(points, L)
    return {
        "n": int(points.shape[0]),
        "dim": int(points.shape[1]),
        "boxsize": L.copy(),
        "target_min": float(np.min(target)),
        "target_mean": float(np.mean(target)),
        "target_median": float(np.median(target)),
        "nnd_min": float(np.min(nnd)),
        "nnd_mean": float(np.mean(nnd)),
        "nnd_median": float(np.median(nnd)),
        "sorted_mse": float(np.mean((np.sort(nnd) - np.sort(target)) ** 2)),
    }


if __name__ == "__main__":
    # Minimal demo / smoke test
    n = 36
    dim = 3
    L = 900

    rng = np.random.default_rng(0)

    # Example: make a target NND list roughly around a typical spacing with some spread.
    # This is just a demo target; replace with your own list.
    typical = np.sqrt((L ** dim) / n)
    target = np.clip(rng.normal(loc=typical, scale=0.25 * typical, size=n), 0.05 * typical, None)
    target = [ 85.44941283, 165.92112421, 161.55754238, 132.77025925,
        97.76532695,  97.76532695, 199.42120742, 132.77025925,
       155.93508656,  85.44941283,  86.86703702, 149.46730708,
       149.46730708, 236.85130962, 105.26740391,  97.95446255,
       105.26740391,  97.95446255,  94.60241043,  82.15557258,
       106.37506676,  82.15557258, 204.02469769, 189.44299414,
       106.37506676, 127.28734323, 179.27788162, 127.28734323,
       151.28239183, 169.63235282, 171.78196743, 128.01808783,
       207.6260156 , 133.18724009, 133.18724009, 128.01808783]

    cfg = AnnealConfig(
        steps=20_000,
        move_scale=0.05,
        repel_weight=50.0,
        r_min_factor=0.65,
        T0=3.0,
        T1=0.002,
        seed=42,
        log_every=2000,
    )

    pts = sample_points_with_nnd_distribution(
        n=n,
        target_nnd=target,
        boxsize=L,
        dim=dim,
        cfg=cfg,
        verbose=True,
    )

    stats = evaluate_nnd_match(pts, target, L)
    print("\nSummary:")
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")

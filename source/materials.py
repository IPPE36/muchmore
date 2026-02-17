from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from source.plotting import plot_material_model

warnings.simplefilter("ignore", RuntimeWarning)


def average_xy_over_full_union(
    curves: Sequence[Tuple[Sequence[float], Sequence[float]]],
    *,
    grid: str = "linspace",          # "linspace" | "union"
    n_points: int = 400,             # used if grid="linspace"
    return_std: bool = False,
    return_n: bool = False,
    return_all: bool = False,
):
    """
    Compute (x_avg, y_avg) across the FULL x-range covered by any curve.
    For each x in x_avg, y is interpolated ONLY for curves whose x-domain contains x.
    Curves that do not cover x contribute NaN at that point and are excluded from the mean.
    Notes:
      - Each curve's x is sorted (and y permuted accordingly).
      - Duplicate x values within a curve are dropped (keeping the first occurrence).
      - Linear interpolation (np.interp). No extrapolation.

    Returns:
      x_avg, y_avg
      optionally std_y (nanstd), n_contrib (count per x), ys_interp (stacked)
    """
    if not curves:
        raise ValueError("curves must contain at least one (x, y) pair.")

    xs, ys = [], []
    for i, (x, y) in enumerate(curves):
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.size != y_arr.size:
            raise ValueError(f"Curve {i}: x and y must have same length.")
        if x_arr.size < 2:
            raise ValueError(f"Curve {i}: need at least 2 points.")

        order = np.argsort(x_arr)
        x_arr = x_arr[order]
        y_arr = y_arr[order]

        # drop duplicates in x (keep first)
        keep = np.concatenate(([True], np.diff(x_arr) > 0))
        x_arr = x_arr[keep]
        y_arr = y_arr[keep]
        if x_arr.size < 2:
            raise ValueError(f"Curve {i}: not enough unique x points after dedup.")

        xs.append(x_arr)
        ys.append(y_arr)

    global_left = min(x[0] for x in xs)
    global_right = max(x[-1] for x in xs)

    if grid == "linspace":
        if n_points < 2:
            raise ValueError("n_points must be >= 2.")
        x_avg = np.linspace(global_left, global_right, int(n_points))
    elif grid == "union":
        x_avg = np.unique(np.concatenate(xs))
        if x_avg.size < 2:
            x_avg = np.linspace(global_left, global_right, 2)
    else:
        raise ValueError("grid must be 'linspace' or 'union'.")

    # Interpolate each curve onto x_avg, but exclude out-of-domain x via NaNs
    # np.interp supports left/right fill values for out-of-bounds.
    ys_interp = np.vstack([
        np.interp(x_avg, x, y, left=np.nan, right=np.nan)
        for x, y in zip(xs, ys)
    ])  # shape: (n_curves, len(x_avg))

    n_contrib = np.sum(np.isfinite(ys_interp), axis=0)
    y_avg = np.nanmean(ys_interp, axis=0)  # NaN where no curve contributes

    out = (x_avg, y_avg)

    if return_std:
        std_y = np.nanstd(ys_interp, axis=0, ddof=0)
        out = (*out, std_y)

    if return_n:
        out = (*out, n_contrib)

    if return_all:
        out = (*out, ys_interp)

    return out


class MaterialModel:
    def predict(self, strain: float) -> float:
        raise NotImplementedError

    def to_inp_str(self):
        raise NotImplementedError

    def fit(self, x, y):
        raise NotImplementedError

    def plot(self, filepath: str):
        raise NotImplementedError

    def name(self):
        return self.df_meta.iloc[0, -1].replace("/", "_")

    def read_xlsx(self, filepath: str):
        try:
            columns = pd.read_excel(filepath, sheet_name="Results", header=0).columns
            df_meta = pd.read_excel(filepath, sheet_name="Results", header=1, names=columns)
            df_meta = df_meta.rename(columns={"Unnamed: 0": "Specimen"})
            df_meta = df_meta.iloc[:, :-1]
        except ValueError:
            columns = pd.read_excel(filepath, sheet_name="Ergebnisse", header=0).columns
            df_meta = pd.read_excel(filepath, sheet_name="Ergebnisse", header=1, names=columns)
            df_meta = df_meta.rename(columns={"Unnamed: 0": "Specimen"})
            df_meta = df_meta.iloc[:, :-1]

        replicates = []
        for name in df_meta["Specimen"].values:
            df = pd.read_excel(filepath, sheet_name=name, header=1, skiprows=[2], usecols=[0, 1])
            df.columns = ["strain_pct", "stress_MPa"]
            df = df.dropna()
            eps = df["strain_pct"].astype(float) / 100.0
            sig = df["stress_MPa"].astype(float)
            x, y = eps.to_list(), sig.to_list()
            x, y = self.sort_xy(x, y)
            x, y = self.eng_to_true(x, y)
            replicates.append((x, y))

        if not replicates:
            raise ValueError("No Specimen sheets with usable data.")

        return replicates, df_meta

    @staticmethod
    def sort_xy(x, y) -> Tuple[list, list]:
        """ensure monotone rising values"""
        idx = np.argsort(x)
        return np.asarray(x)[idx].tolist(), np.asarray(y)[idx].tolist()

    @staticmethod
    def eng_to_true(eps_eng: list, sig_eng: list) -> Tuple[list, list]:
        """Convert engineering strain/stress to true strain and a necking-corrected true stress...
        eps_conv = ln(1+e)
        sig_conv = s*(1+e)
        """
        e = np.asarray(eps_eng, dtype=float)
        s = np.asarray(sig_eng, dtype=float)

        # Base conversions (uniform assumption)
        eps_conv = np.log1p(e)
        sig_conv = s * (1.0 + e)
        # A_conv = A0 / (1.0 + e)
        slope_conv = np.gradient(sig_conv, eps_conv)

        # Necking start at peak of sig_conv
        i0 = np.where(eps_conv < 0.25)[0][-1]
        i_neck_start = np.nanargmax(np.where(eps_conv < 0.2, sig_conv, 0))
        try:
            i_neck_end = np.nanargmin(sig_conv[i_neck_start:i0]) + i_neck_start
        except ValueError:
            i_neck_end = np.nanargmin(sig_conv[i_neck_start:]) + i_neck_start

        # linear hardening after necking
        if len(s) - i_neck_end > len(s)//20:
            hardening_slope = np.median(np.gradient(
                sig_conv[i_neck_end:i_neck_end+len(s)//20],
                eps_conv[i_neck_end:i_neck_end+len(s)//20])
            )
            i_trans = np.where((slope_conv < hardening_slope) & (eps_conv > 0.03))[0][0]
            sig_trans = sig_conv[i_trans]
            sig_conv[i_trans:] = sig_trans + (eps_conv[i_trans:] - eps_conv[i_trans]) * hardening_slope
        else:
            eps_conv = eps_conv[:i_neck_start]
            sig_conv = sig_conv[:i_neck_start]
        return eps_conv.tolist(), sig_conv.tolist()

    # ---------------------------
    # Serialization (dump/load)
    # ---------------------------
    def dump(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable dict with model state + data.
        Expects self.replicates (list of (x,y)) and self.df_meta (pd.DataFrame) if present.
        """
        state: Dict[str, Any] = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
        }

        # store replicates if available
        if hasattr(self, "replicates") and self.replicates is not None:
            reps = []
            for x, y in self.replicates:
                reps.append({
                    "x": [float(v) for v in x],
                    "y": [float(v) for v in y],
                })
            state["replicates"] = reps

        # store meta dataframe if available
        if hasattr(self, "df_meta") and self.df_meta is not None:
            df = self.df_meta.copy()
            state["df_meta"] = {
                "columns": [str(c) for c in df.columns],
                "records": df.astype(object).where(pd.notnull(df), None).to_dict(orient="records"),
            }

        # store any additional lightweight parameters a subclass may set
        # (override _extra_dump/_extra_load in subclasses)
        state["extra"] = self._extra_dump()
        state["poisson"] = self.poisson_ratio
        state["young_modulus"] = self.young_modulus
        state["density"] = self.density
        state["plastic_table"] = self.plastic_table

        return state

    @classmethod
    def load(cls, state: Dict[str, Any]) -> "MaterialModel":
        """
        Create an instance from a dict previously returned by dump().
        Note: this constructs `cls`, not the recorded class in state["class"].
        """
        density = state["density"]
        E = state["young_modulus"]
        poisson_ratio = state["poisson"]
        plastic_table = state["plastic_table"]
        obj = cls(density=density, young_modulus=E, poisson_ratio=poisson_ratio)
        setattr(obj, "plastic_table", plastic_table)

        # replicates
        reps_state = state.get("replicates")
        if reps_state is not None:
            obj.replicates = [
                (r.get("x", []), r.get("y", []))
                for r in reps_state
            ]

        # df_meta
        meta_state = state.get("df_meta")
        if meta_state is not None:
            records = meta_state.get("records", [])
            columns = meta_state.get("columns", None)
            df = pd.DataFrame.from_records(records)
            if columns is not None:
                # enforce original column order if provided
                df = df.reindex(columns=columns)
            obj.df_meta = df

        obj._extra_load(state.get("extra", {}))
        return obj

    def dump_json(self, path: str, *, indent: int = 2) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.dump(), f, indent=indent, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> "MaterialModel":
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return cls.load(state)

    def _extra_dump(self) -> Dict[str, Any]:
        """
        Hook for subclasses to add parameters (e.g. fitted coefficients).
        Must return JSON-serializable data.
        """
        return {}

    def _extra_load(self, extra: Dict[str, Any]) -> None:
        """
        Hook for subclasses to restore parameters.
        """
        return


@dataclass
class ElasticMM(MaterialModel):

    density: float
    young_modulus: float
    poisson_ratio: float = 0.40
    plastic_table: List[Tuple[float, float]] = field(default_factory=list)
    df_meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    replicates: List[Tuple[List[float], List[float]]] = field(default_factory=list)

    def predict(self, eps: float) -> float:
        return self.young_modulus * eps

    def plot(self, filepath: str, max_strain: float = 0.03, n: int = 400) -> None:
        """Plot model curve alongside replicates using existing plot_material_model()."""
        x = np.linspace(0.0, max_strain, n).tolist()
        y = [self.predict(x_) for x_ in x]
        plot_material_model(self.replicates, x, y, self.df_meta, max_strain, filepath)
        return None

    def to_inp_str(self, name: str = "ELASTIC") -> str:
        """Return Abaqus *Material block for linear elasticity."""
        out = list()
        out.append(f"*Material, name={name}")
        out.append("*Density")
        out.append(f"{self.density * 1e-9:.6g}")
        out.append("*Elastic")
        out.append(f"{self.young_modulus:.6g}, {self.poisson_ratio:.6g}")
        return "\n".join(out)

    @classmethod
    def from_xy(cls, eps: list, sig: list, density: float = 1.0, poisson_ratio: float = 0.40) -> "ElasticMM":
        """Generate model directly from xy data"""
        xs = [float(x) for x in eps]
        ys = [float(y) for y in sig]
        xs, ys = super().eng_to_true(xs, ys)
        E, _ = cls.fit(xs, ys)
        return cls(density=density, young_modulus=E, poisson_ratio=poisson_ratio, df_meta=pd.DataFrame(), replicates=[(xs, ys)])

    @classmethod
    def from_xlsx(cls, filepath: str, density: float = 1.0, poisson_ratio: float = 0.40) -> "ElasticMM":
        replicates, df_meta = super().read_xlsx(cls, filepath)
        E_mean = np.mean([cls.fit(x, y)[0] for x, y in replicates])
        return cls(density=density, young_modulus=E_mean, poisson_ratio=poisson_ratio, df_meta=df_meta, replicates=replicates)

    @staticmethod
    def fit(x, y, lo: float = 0.0005, hi: float = 0.002) -> Tuple[float, float]:
        """Ordinary least squares with intercept, Select ISO 527 window"""
        pts = [(x_, y_) for x_, y_ in zip(x, y) if (lo <= x_ <= hi)]
        fx, fy = zip(*pts)
        n = len(fx)
        xbar = sum(fx) / n
        ybar = sum(fy) / n
        sxx = sum((x - xbar) ** 2 for x in fx)
        sxy = sum((x - xbar) * (y - ybar) for x, y in zip(fx, fy))
        E = sxy / sxx
        b = ybar - E * xbar
        return float(E), float(b)


@dataclass
class ElasticPlasticMM(MaterialModel):

    density: float
    young_modulus: float
    poisson_ratio: float = 0.40
    plastic_table: List[Tuple[float, float]] = field(default_factory=list)
    df_meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    replicates: List[Tuple[List[float], List[float]]] = field(default_factory=list)

    def predict(self, eps_true: float) -> float:
        """
        Uniaxial stress prediction (true stress) for monotone loading.
        Assumes eps_true is TOTAL TRUE strain (consistent with read_xlsx() output).
        Uses:  eps = eps_p + sigma/E
        where sigma is obtained from the *Plastic table (sigma vs eps_p).
        For eps below first plastic point -> purely elastic: sigma = E * eps.
        """
        if eps_true <= 0.0:
            return 0.0

        E = float(self.young_modulus)
        if E <= 0:
            raise ValueError("young_modulus must be positive")

        # fallback: elastic-only
        if not self.plastic_table:
            return E * float(eps_true)

        # plastic_table: list[(sigma_true, eps_p_true)]
        sig = np.asarray([s for s, _ in self.plastic_table], dtype=float)
        ep = np.asarray([ep_ for _, ep_ in self.plastic_table], dtype=float)

        # Ensure increasing plastic strain (should already be true)
        idx = np.argsort(ep)
        ep = ep[idx]
        sig = sig[idx]

        # Total strain at table points: eps = ep + sig/E
        eps_total = ep + sig / E

        # Define the "yield" total strain as first table point
        eps_y = float(eps_total[0])

        # Elastic region
        if eps_true <= eps_y:
            return E * float(eps_true)

        # Plastic region: interpolate sigma as function of total strain
        # np.interp clamps outside bounds (left/right), which is fine for monotone extrapolation.
        sigma_pred = float(np.interp(float(eps_true), eps_total, sig))
        return sigma_pred

    def plot(self, filepath: str, max_strain: float = 0.5, n: int = 100) -> None:
        """Plot model curve alongside replicates using existing plot_material_model()."""
        max_ = self.df_meta["eB"].max() / 100
        max_strain = max_ if max_ < max_strain else max_strain
        x = np.linspace(0.0, max_strain, n).tolist()
        y = [self.predict(x_) for x_ in x]
        plot_material_model(self.replicates, x, y, self.df_meta, max_strain, filepath)
        return None

    def to_inp_str(self) -> str:
        """Return Abaqus *Material block for linear elasto-plasticity."""
        out = list()
        out.append("*Material, name=ELASTOPLASTIC")
        out.append("*Density")
        out.append(f"{self.density * 1e-9:.6g}")
        out.append("*Elastic")
        out.append(f"{self.young_modulus:.6g}, {self.poisson_ratio:.6g}")
        out.append("*Plastic")
        for s, ep in self.plastic_table:
            out.append(f"{s:.6g}, {ep:.6g}")
        return "\n".join(out)

    @classmethod
    def from_xy(cls, eps: list, sig: list, density: float = 1.0, poisson_ratio: float = 0.40) -> "ElasticPlasticMM":
        xs = [float(x) for x in eps]
        ys = [float(y) for y in sig]
        xs, ys = super().eng_to_true(xs, ys)
        E, b = ElasticMM.fit(xs, ys)
        ys = ys - b  # remove intercept
        table = cls._build_plastic_table(xs, ys, young_modulus=E)
        return cls(density=density, young_modulus=E, poisson_ratio=poisson_ratio, plastic_table=table,
                   df_meta=pd.DataFrame(), replicates=[(xs, ys)])

    @classmethod
    def from_xlsx(cls, filepath: str, density: float = 1.0, poisson_ratio: float = 0.40) -> "ElasticPlasticMM":
        replicates, df_meta = super().read_xlsx(cls, filepath)
        E_mean = np.mean([ElasticMM.fit(x_, y_)[0] for x_, y_ in replicates])

        sigs = []
        for xs, ys in replicates:
            x = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=float)
            idx = np.argsort(x)  # ensure monotone x for interp
            x, y = x[idx], y[idx]
            sigs.append((x, y))

        avg_x, avg_y = average_xy_over_full_union(sigs, grid="linspace", n_points=1000)

        table = cls._build_plastic_table(avg_x, avg_y, young_modulus=E_mean)
        return cls(density=density, young_modulus=E_mean, poisson_ratio=poisson_ratio, plastic_table=table,
                   df_meta=df_meta, replicates=replicates)

    @staticmethod
    def _build_plastic_table(
            eps_true: list,
            sig_true: list,
            young_modulus: float,
            yield_offset: float = 0.001,
            smooth: int = 1,
    ) -> List[Tuple[float, float]]:
        """
        Build Abaqus *Plastic table: (true stress, true plastic strain).
        eps_p = eps_true - sig_true/E

        Key change: ensures the table starts at eps_p = 0 (anchor point),
        which removes the common elastic->plastic "jump" when switching in predict().
        """
        eps_true = np.asarray(eps_true, dtype=float)
        sig_true = np.asarray(sig_true, dtype=float)

        if smooth and smooth > 1:
            w = int(smooth)
            kernel = np.ones(w) / w
            sig_true = np.convolve(sig_true, kernel, mode="same")

        E = float(young_modulus)
        if E <= 0:
            raise ValueError("young_modulus must be positive")

        # compute plastic strain
        eps_p = eps_true - sig_true / E

        # keep finite
        m = np.isfinite(eps_p) & np.isfinite(sig_true)
        eps_p = eps_p[m]
        sig_true = sig_true[m]

        # sort by eps_p
        idx = np.argsort(eps_p)
        eps_p = eps_p[idx]
        sig_true = sig_true[idx]

        # drop duplicate / non-increasing eps_p
        keep = np.r_[True, np.diff(eps_p) > 0]
        eps_p = eps_p[keep]
        sig_true = sig_true[keep]

        # --- NEW: anchor at eps_p = 0 for continuity ---
        # estimate stress at eps_p=0 by interpolation if possible
        if eps_p.size == 0:
            return []

        if eps_p[0] > 0.0:
            sig0 = float(sig_true[0])
        else:
            sig0 = float(np.interp(0.0, eps_p, sig_true))

        # prepend anchor
        eps_p = np.r_[0.0, eps_p]
        sig_true = np.r_[sig0, sig_true]

        # now keep only eps_p >= 0 (but keep the anchor regardless)
        m3 = eps_p >= 0.0
        eps_p = eps_p[m3]
        sig_true = sig_true[m3]

        # apply yield_offset gate, but DON'T drop the anchor at 0
        yoff = float(yield_offset)
        m2 = (eps_p == 0.0) | (eps_p >= yoff)
        eps_p = eps_p[m2]
        sig_true = sig_true[m2]

        # enforce Abaqus-friendly monotone stress
        sig_true = np.maximum.accumulate(sig_true)

        return list(zip(sig_true.astype(float), eps_p.astype(float)))


if __name__ == "__main__":
    exit()

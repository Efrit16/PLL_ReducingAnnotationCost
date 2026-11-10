import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


CLASS_NAMES = ["Empty", "Low", "Moderate", "High"]


FLOOR_S_BINS = [-0.01, 0.5, 10.0, 30.0, 1000.0]
ROBOD_BINS = {
    1: [-0.01, 0.5, 5.0, 15.0, 500.0],
    2: [-0.01, 0.5, 5.0, 15.0, 500.0],
    3: [-0.01, 0.5, 3.0, 8.0, 500.0],
    4: [-0.01, 0.5, 5.0, 12.0, 500.0],
    5: [-0.01, 0.5, 5.0, 12.0, 500.0],
}
ROBOD_NAMES = {1: "Lecture (Large)", 2: "Lecture (Small)", 3: "Admin Office",
               4: "Research Office", 5: "Library"}


FLOOR_S_LAG_CHANNELS = ["mels_S", "lig_S", "hvac_S", "air_temp_set_1",
                        "dew_point_temperature_set_1d", "relative_humidity_set_1",
                        "solar_radiation_set_1", "cool_delta_mean",
                        "cool_delta_max", "zones_above_cool_sp"]
ROBOD_LAG_CHANNELS = ["indoor_co2", "air_temperature", "illuminance",
                      "wifi_connected_devices", "lighting_energy",
                      "plug_load_energy", "voc", "sound_pressure_level"]

FLOOR_S_LAGS = (1, 4)
FLOOR_S_ROLL = 4
ROBOD_LAGS = (1, 3)
ROBOD_ROLL = 4


@dataclass
class Config:
    data_dir: str = "data"
    out_dir: str = "results_revision4_paper"
    floor_csv: str = "floor_S_df_cleaned.csv"
    robod_pattern: str = "combined_Room{rid}.csv"
    datasets: List[str] = field(default_factory=lambda: [
        "Floor_S", "ROBOD_R1", "ROBOD_R2", "ROBOD_R3", "ROBOD_R4", "ROBOD_R5"])


    protocol: str = "blocked"
    n_folds_floor: int = 10
    n_folds_robod: int = 5
    purge_hours_floor: float = 24.0
    purge_hours_robod: float = 12.0


    budgets: Tuple[float, ...] = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    n_placements: int = 3
    n_placements_floor: int = 2
    master_seed: int = 2026
    chunk_days_floor: float = 7.0
    chunk_days_robod: float = 1.0
    budget_methods: Tuple[str, ...] = ("SoDisam",)
    full_methods: Tuple[str, ...] = ("HTC-LTPLL", "SoDisam", "UCL", "TERIAL")
    cand_modes: Tuple[str, ...] = ("seed_only",)


    tau: float = 0.25
    tau_mode: str = "fixed"

    tau_select_grid: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    tau_target_cov: float = 0.90
    tau_inner_folds: int = 5
    tau_fallback: float = 0.25
    dt_depth: int = 6
    dt_min_leaf: int = 5
    dt_class_weight: Optional[str] = None
    dt_min_set_size: int = 1


    proxy_abs: float = 0.5
    proxy_rel: float = 0.25


    lr_C: float = 1.0
    lr_class_weight: Optional[str] = None


    imbalance: str = "none"
    smote_k: int = 5
    smote_ratio: float = 0.5
    imbalance_baselines: bool = True
    knn_k: int = 10


    pll_T: int = 10
    htc_alpha: float = 0.5
    htc_beta: float = 0.9
    sodisam_delta: float = 0.10
    ucl_beta: float = 0.9
    terial_gamma: float = 0.6
    terial_hidden: int = 32
    terial_steps: int = 100
    terial_test_iter: int = 5


    margin_acc: float = 0.02
    margin_f1: float = 0.03
    alpha: float = 0.05


    lag_mode: str = "segment"
    quick: bool = False
    tau_grid: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
    tau_budgets: Tuple[float, ...] = (0.10, 0.40)
    tau_run_pll: bool = True
    tau_pll_grid: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.40)
    decomp_budgets: Tuple[float, ...] = (0.10, 0.40)

    def n_folds(self, name):  return self.n_folds_floor if name == "Floor_S" else self.n_folds_robod
    def purge(self, name):    return pd.Timedelta(hours=self.purge_hours_floor if name == "Floor_S" else self.purge_hours_robod)
    def placements(self, name):
        return 1 if self.quick else (self.n_placements_floor if name == "Floor_S" else self.n_placements)
    def chunk_days(self, name): return self.chunk_days_floor if name == "Floor_S" else self.chunk_days_robod


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    times: pd.Series
    feature_names: List[str]
    segment: np.ndarray
    step: pd.Timedelta
    bins: List[float]
    info: dict
    count: Optional[np.ndarray] = None

    @property
    def steps_per_day(self): return pd.Timedelta("1D") / self.step


def _segments(times: pd.Series, step: pd.Timedelta) -> np.ndarray:
    gap = times.diff() > step * 1.5
    return gap.cumsum().to_numpy()


def _add_lags(df, channels, lags, roll, seg, mode):
    names = []
    grp = df.groupby(seg) if mode == "segment" else None
    for c in channels:
        for L in lags:
            df[f"{c}_lag{L}"] = grp[c].shift(L) if grp is not None else df[c].shift(L)
            names.append(f"{c}_lag{L}")
        if grp is not None:
            df[f"{c}_roll{roll}"] = grp[c].transform(lambda s: s.rolling(roll, min_periods=1).mean())
        else:
            df[f"{c}_roll{roll}"] = df[c].rolling(roll, min_periods=1).mean()
        names.append(f"{c}_roll{roll}")
    return names


def _cyc(df, tcol):
    df["sin_hour"] = np.sin(2 * np.pi * df[tcol].dt.hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df[tcol].dt.hour / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df[tcol].dt.dayofweek / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df[tcol].dt.dayofweek / 7)
    return ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]


def load_floor_s(cfg: Config) -> Dataset:
    df = pd.read_csv(os.path.join(cfg.data_dir, cfg.floor_csv))
    df.columns = df.columns.str.strip()
    occ_col = next(c for c in df.columns if "occ" in c.lower() and "south" in c.lower())
    tcol = "datetime"
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol).reset_index(drop=True)
    df["occ_class"] = pd.cut(df[occ_col], bins=FLOOR_S_BINS, labels=[0, 1, 2, 3]).astype(float)
    df = df.dropna(subset=["occ_class"]).copy()
    df["occ_class"] = df["occ_class"].astype(int)
    df = df.reset_index(drop=True)
    step = pd.Timedelta(df[tcol].diff().mode()[0])
    seg = _segments(df[tcol], step)

    base = [c for c in df.columns if c not in {occ_col, "occ_class", tcol}
            and pd.api.types.is_numeric_dtype(df[c])]
    lag_names = _add_lags(df, FLOOR_S_LAG_CHANNELS, FLOOR_S_LAGS, FLOOR_S_ROLL, seg, cfg.lag_mode)
    cyc = _cyc(df, tcol)
    feats = base + lag_names + cyc
    keep = df[feats].notna().all(axis=1).to_numpy()
    df, seg = df[keep].reset_index(drop=True), seg[keep]
    info = {"calendar_raw_features": [c for c in ["hour", "dayofweek", "month"] if c in feats],
            "n_raw_rows": int(keep.size), "n_rows_after_lags": int(keep.sum())}
    return Dataset("Floor_S", df[feats].to_numpy(float), df["occ_class"].to_numpy(), df[tcol],
                   feats, seg, step, FLOOR_S_BINS, info, count=df[occ_col].to_numpy(float))


def load_robod_room(cfg: Config, rid: int) -> Dataset:
    path = os.path.join(cfg.data_dir, cfg.robod_pattern.format(rid=rid))
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_raw = len(df)
    df["occ_class"] = pd.cut(df["occupant_count"], bins=ROBOD_BINS[rid], labels=[0, 1, 2, 3]).astype(float)

    df = df.dropna().reset_index(drop=True)
    n_after_nan = len(df)
    df["occ_class"] = df["occ_class"].astype(int)
    step = pd.Timedelta(minutes=5)
    seg = _segments(df["timestamp"], step)

    drop = {"timestamp", "occupant_count", "occupant_presence", "occ_class"}
    base = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    cyc = _cyc(df, "timestamp")
    chans = [c for c in ROBOD_LAG_CHANNELS if c in df.columns]
    lag_names = _add_lags(df, chans, ROBOD_LAGS, ROBOD_ROLL, seg, cfg.lag_mode)
    feats = base + cyc + lag_names
    keep = df[feats].notna().all(axis=1).to_numpy()
    df, seg = df[keep].reset_index(drop=True), seg[keep]
    info = {"n_raw_rows": n_raw, "n_after_dropna": n_after_nan, "n_after_lags": int(keep.sum()),
            "rows_lost_to_nan": n_raw - n_after_nan}
    return Dataset(f"ROBOD_R{rid}", df[feats].to_numpy(float), df["occ_class"].to_numpy(),
                   df["timestamp"], feats, seg, step, ROBOD_BINS[rid], info,
                   count=df["occupant_count"].to_numpy(float))


def load_dataset(cfg: Config, name: str) -> Dataset:
    if name == "Floor_S":
        return load_floor_s(cfg)
    return load_robod_room(cfg, int(name.split("R")[-1]))


def make_folds(ds: Dataset, cfg: Config) -> List[dict]:
    n, K = len(ds.y), cfg.n_folds(ds.name)
    if cfg.quick:
        K_run = 2
    else:
        K_run = K
    folds = []
    if cfg.protocol == "shuffled":
        skf = StratifiedKFold(K, shuffle=True, random_state=42)
        for k, (tr, te) in enumerate(skf.split(ds.X, ds.y)):
            folds.append({"fold": k, "tr": tr, "te": te})
    else:
        t = ds.times.reset_index(drop=True)
        purge = cfg.purge(ds.name)
        for k, te in enumerate(np.array_split(np.arange(n), K)):
            t0, t1 = t.iloc[te[0]], t.iloc[te[-1]]
            tr = np.where((t < t0 - purge) | (t > t1 + purge))[0]
            folds.append({"fold": k, "tr": tr, "te": te})
    for f in folds:
        te_t = ds.times.iloc[f["te"]]
        f["test_start"], f["test_end"] = str(te_t.iloc[0]), str(te_t.iloc[-1])
        f["n_train"], f["n_test"] = len(f["tr"]), len(f["te"])
        f["n_test_segments"] = int(len(np.unique(ds.segment[f["te"]])))
    return folds[:K_run]


def seed_indices(n_train: int, steps_per_day: float, chunk_days: float,
                 budget: float, rng_seed: int, return_chunks: bool = False):
    csize = max(1, int(round(chunk_days * steps_per_day)))
    bounds = list(range(0, n_train, csize)) + [n_train]
    chunks = [np.arange(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    if budget >= 1.0:
        pos = np.arange(n_train)
        cid = np.concatenate([np.full(len(c), i) for i, c in enumerate(chunks)])
        return (pos, cid) if return_chunks else pos
    order = np.random.RandomState(rng_seed).permutation(len(chunks))
    target, got, picked = budget * n_train, 0, []
    for ci in order:
        picked.append(ci); got += len(chunks[ci])
        if got >= target:
            break
    pos = np.concatenate([chunks[ci] for ci in picked])
    cid = np.concatenate([np.full(len(chunks[ci]), ci) for ci in picked])
    o = np.argsort(pos)
    return (pos[o], cid[o]) if return_chunks else pos[o]


def _fit_tree(X, y, cfg: Config, rs):
    return DecisionTreeClassifier(max_depth=cfg.dt_depth, min_samples_leaf=cfg.dt_min_leaf,
                                  class_weight=cfg.dt_class_weight, random_state=rs).fit(X, y)


def _leaf_proba(dt, X, K):
    raw = dt.predict_proba(X)
    P = np.zeros((len(X), K))
    for j, c in enumerate(dt.classes_):
        P[:, int(c)] = raw[:, j]
    return P


def _sets_from_P(P, tau, cfg: Config):
    M = P >= tau
    if cfg.dt_min_set_size > 1:
        order = np.argsort(-P, axis=1)
        for r in np.where(M.sum(1) < cfg.dt_min_set_size)[0]:
            M[r, order[r, :cfg.dt_min_set_size]] = True
    empty = ~M.any(1)
    M[empty, P[empty].argmax(1)] = True
    return M


def select_tau(X_tr, y_tr, seed_pos, seed_chunk, K, cfg: Config, rs=0):
    grid = sorted(cfg.tau_select_grid)
    chunks = np.unique(seed_chunk)
    info = {"n_seed_chunks": int(len(chunks))}
    if len(chunks) < 2:
        info.update(tau_selected=cfg.tau_fallback, rule="fallback_lt2_chunks")
        return cfg.tau_fallback, info
    n_in = min(cfg.tau_inner_folds, len(chunks))
    perm = np.random.RandomState(rs + 31).permutation(chunks)
    groups = np.array_split(perm, n_in)
    hits = np.zeros(len(grid)); size = np.zeros(len(grid)); n = 0
    Xs, ys = X_tr[seed_pos], y_tr[seed_pos]
    for g in groups:
        ho = np.isin(seed_chunk, g)
        if ho.all() or not ho.any():
            continue
        P = _leaf_proba(_fit_tree(Xs[~ho], ys[~ho], cfg, rs), Xs[ho], K)
        yh = ys[ho]; n += len(yh)
        for i, t in enumerate(grid):
            M = _sets_from_P(P, t, cfg)
            hits[i] += M[np.arange(len(yh)), yh].sum(); size[i] += M.sum()
    if n == 0:
        info.update(tau_selected=cfg.tau_fallback, rule="fallback_no_inner_rows")
        return cfg.tau_fallback, info
    cov = hits / n
    ok = [t for t, c in zip(grid, cov) if c >= cfg.tau_target_cov]
    tau = max(ok) if ok else min(grid)
    info.update(tau_selected=tau, rule="target_met" if ok else "target_not_met_min_tau",
                n_inner_folds=n_in, n_inner_heldout=int(n),
                **{f"innercov_tau{t:.2f}": float(c) for t, c in zip(grid, cov)},
                **{f"innersize_tau{t:.2f}": float(s / n) for t, s in zip(grid, size)})
    return tau, info


def dtcg_candidates(X_tr, y_tr, seed_pos, K, cfg: Config, tau=None, mode="seed_only", rs=0,
                    X_te=None):
    tau = cfg.tau if tau is None else tau
    dt = _fit_tree(X_tr[seed_pos], y_tr[seed_pos], cfg, rs)
    M = _sets_from_P(_leaf_proba(dt, X_tr, K), tau, cfg)
    M[seed_pos, y_tr[seed_pos]] = True
    if mode == "oracle_true":
        M[np.arange(len(y_tr)), y_tr] = True
    M_te = None if X_te is None else _sets_from_P(_leaf_proba(dt, X_te, K), tau, cfg).astype(np.float32)
    return M.astype(np.float32), M_te


def _classes_overlapping(lo, hi, bins):
    e = np.asarray(bins, float)
    return np.stack([(e[k] < hi) & (e[k + 1] >= lo) for k in range(len(e) - 1)], 1)


def proxy_candidates(count, y_tr, seed_pos, bins, K, q, cfg: Config, rs=0):
    rng = np.random.RandomState(rs + 7919)
    y = y_tr; n = len(y)
    w = cfg.proxy_abs + cfg.proxy_rel * count
    M = _classes_overlapping(np.maximum(0, count - w), count + w, bins)
    cap = max(float(np.nanmax(count)), bins[-2] + 1)
    e = np.asarray(bins, float)
    for i in np.where(rng.rand(n) > q)[0]:
        nb = [c for c in (y[i] - 1, y[i] + 1) if 0 <= c < K]
        wc = nb[rng.randint(len(nb))]
        lo_c = 0.0 if wc == 0 else np.floor(e[wc]) + 1
        hi_c = 0.0 if wc == 0 else min(np.floor(e[wc + 1]), max(cap, lo_c))
        c2 = float(rng.randint(int(lo_c), int(hi_c) + 1))
        w2 = cfg.proxy_abs + cfg.proxy_rel * c2
        row = _classes_overlapping(np.array([max(0, c2 - w2)]), np.array([c2 + w2]), bins)[0]
        row[y[i]] = False
        if not row.any():
            row[wc] = True
        M[i] = row
    M[seed_pos] = False
    M[seed_pos, y[seed_pos]] = True
    return M.astype(np.float32)


def candidate_diagnostics(M, y_tr, seed_pos, K) -> dict:
    unl = np.ones(len(y_tr), bool); unl[seed_pos] = False
    return set_diagnostics(M, y_tr, unl, K, suffix="unl")


def set_diagnostics(M, y, mask, K, suffix="unl") -> dict:
    out = {f"n_{suffix}": int(mask.sum())}
    if mask.sum() == 0:
        return out
    hit = M[np.arange(len(y)), y] > 0
    size = M.sum(1)
    single = mask & (size == 1)
    out[f"recall_{suffix}"] = float(hit[mask].mean())
    out[f"ambig_rate_{suffix}"] = float((size[mask] > 1).mean())
    out[f"mean_size_{suffix}"] = float(size[mask].mean())
    out[f"singleton_frac_{suffix}"] = float(single.sum() / mask.sum())
    out[f"singleton_err_{suffix}"] = float((~hit[single]).mean()) if single.sum() else np.nan
    tag = "" if suffix == "unl" else f"{suffix}_"
    for c in range(K):
        m = mask & (y == c); s = m & (size == 1)
        out[f"{tag}recall_c{c}"] = float(hit[m].mean()) if m.sum() else np.nan
        out[f"{tag}n_{suffix}_c{c}"] = int(m.sum())
        out[f"{tag}singleton_frac_c{c}"] = float(s.sum() / m.sum()) if m.sum() else np.nan
        out[f"{tag}singleton_err_c{c}"] = float((~hit[s]).mean()) if s.sum() else np.nan
        out[f"{tag}miss_wrong_singleton_c{c}"] = float((s & ~hit).sum() / m.sum()) if m.sum() else np.nan
    return out


class _Const:
    def __init__(self, c): self.classes_ = np.array([c])
    def predict(self, X): return np.full(len(X), self.classes_[0])
    def predict_proba(self, X): return np.ones((len(X), 1))


SMOTE_STATS = {"calls": 0, "synth": 0, "skipped_classes": 0, "k_reduced": 0}


def reset_smote_stats():
    for k in SMOTE_STATS:
        SMOTE_STATS[k] = 0


def smote_resample(X, y, cfg: Config, seed=0):
    rng = np.random.RandomState(seed)
    cnt = np.bincount(y)
    target = int(np.ceil(cfg.smote_ratio * cnt.max()))
    Xn, yn = [X], [y]
    SMOTE_STATS["calls"] += 1
    for c in np.where((cnt > 0) & (cnt < target))[0]:
        Xc = X[y == c]; nc = len(Xc)
        if nc < 2:
            SMOTE_STATS["skipped_classes"] += 1
            continue
        k = min(cfg.smote_k, nc - 1)
        if k < cfg.smote_k:
            SMOTE_STATS["k_reduced"] += 1
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Xc).kneighbors(Xc, return_distance=False)[:, 1:]
        m = target - nc
        base = rng.randint(nc, size=m)
        nbr = nn[base, rng.randint(k, size=m)]
        Xn.append(Xc[base] + rng.rand(m, 1) * (Xc[nbr] - Xc[base]))
        yn.append(np.full(m, c))
        SMOTE_STATS["synth"] += m
    return np.vstack(Xn), np.concatenate(yn)


def fit_lr(X, y, cfg: Config, C=None, solver="lbfgs", seed=0, imbalance=None):
    imb = cfg.imbalance if imbalance is None else imbalance
    if len(np.unique(y)) < 2:
        return _Const(int(y[0]))
    cw = cfg.lr_class_weight
    if imb == "smote":
        X, y = smote_resample(X, y, cfg, seed=seed)
    elif imb == "weight":
        cw = "balanced"
    clf = LogisticRegression(max_iter=500, C=cfg.lr_C if C is None else C, solver=solver,
                             class_weight=cw, random_state=seed)
    return clf.fit(X, y)


def safe_proba(clf, X, K):
    raw = clf.predict_proba(X)
    if len(clf.classes_) == K:
        return raw
    full = np.zeros((len(X), K))
    for j, c in enumerate(clf.classes_):
        full[:, int(c)] = raw[:, j]
    s = full.sum(1, keepdims=True); s[s == 0] = 1
    return full / s


def masked_argmax(S, M):
    return np.where(M > 0, S, -1.0).argmax(1)


class FoldContext:
    def __init__(self, X_tr, X_te, cfg: Config):
        self.scaler = StandardScaler().fit(X_tr)
        self.Xtr = self.scaler.transform(X_tr)
        self.Xte = self.scaler.transform(X_te)


LAST_PSEUDO = {"y": None}


def _rownorm(A, eps=1e-12):
    s = A.sum(1, keepdims=True); s[s < eps] = 1.0
    return A / s


def _init_q(M):
    return _rownorm(M.astype(np.float64))


def _tiebreak_pseudo(ctx, M, K, cfg, seed):
    single = M.sum(1) == 1
    if single.sum() >= 2 and len(np.unique(M[single].argmax(1))) >= 2:
        clf = fit_lr(ctx.Xtr[single], M[single].argmax(1), cfg, seed=seed)
        return masked_argmax(safe_proba(clf, ctx.Xtr, K), M)
    return masked_argmax(np.random.RandomState(seed).rand(*M.shape), M)


def _pseudo_from_q(q, M, prev):
    qm = np.where(M > 0, q, -1.0)
    top = qm.max(1, keepdims=True)
    tied = ((qm >= top - 1e-12).sum(1) > 1)
    out = qm.argmax(1)
    out[tied] = prev[tied]
    return out


def p_htc(ctx, M, K, cfg, seed):
    rng = np.random.RandomState(seed)
    q = _init_q(M)
    pseudo = _tiebreak_pseudo(ctx, M, K, cfg, seed)
    for t in range(1, cfg.pll_T + 1):
        head = fit_lr(ctx.Xtr, pseudo, cfg, seed=seed)

        cls, cnt = np.unique(pseudo, return_counts=True)
        idx = np.concatenate([rng.choice(np.where(pseudo == c)[0], cnt.max(), replace=True) for c in cls])
        tail = fit_lr(ctx.Xtr[idx], pseudo[idx], cfg, seed=seed, imbalance="none")
        pi = np.bincount(pseudo, minlength=K) / len(pseudo)
        w = pi ** cfg.htc_alpha; w = w / w.sum()
        p = w * safe_proba(head, ctx.Xtr, K) + (1 - w) * safe_proba(tail, ctx.Xtr, K)
        Tt = max(0.1, 1 - 0.06 * t)
        r = _rownorm((np.clip(p, 1e-12, None) ** (1.0 / Tt)) * M)
        q = cfg.htc_beta * q + (1 - cfg.htc_beta) * r
        pseudo = _pseudo_from_q(q, M, pseudo)
    LAST_PSEUDO["y"] = pseudo
    pt = w * safe_proba(head, ctx.Xte, K) + (1 - w) * safe_proba(tail, ctx.Xte, K)
    return pt.argmax(1)


def _prototypes(X, M, K, pseudo=None):
    mu = np.zeros((K, X.shape[1]))
    for k in range(K):
        if pseudo is not None:
            mk = pseudo == k
        else:
            mk = (M[:, k] == 1) & (M.sum(1) == 1)
            if mk.sum() < 2: mk = M[:, k] == 1
        mu[k] = X[mk].mean(0) if mk.sum() else X.mean(0)
    return mu


def p_sodisam(ctx, M, K, cfg, seed):
    X = ctx.Xtr
    mu = _prototypes(X, M, K)


    pseudo = np.zeros(len(X), int)
    pats, inv = np.unique(M.astype(np.int8), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    for g, pat in enumerate(pats):
        rows = np.where(inv == g)[0]
        cand = np.where(pat > 0)[0]
        if len(cand) == 1:
            pseudo[rows] = cand[0]; continue
        mbar = mu[cand].mean(0)
        a = 1.0 / (1.0 + np.exp(-np.abs(mu[cand] - mbar).mean(0)))
        D = np.stack([((X[rows] - mu[k]) ** 2 * a).sum(1) for k in cand], 1)
        pseudo[rows] = cand[D.argmin(1)]
    Ycur = M.astype(np.float64).copy()
    q = _init_q(M)
    for t in range(1, cfg.pll_T + 1):
        clf = fit_lr(X, pseudo, cfg, seed=seed)
        p = safe_proba(clf, X, K)
        keep = (q >= cfg.sodisam_delta) & (Ycur > 0)
        empty = keep.sum(1) == 0
        if empty.any():
            keep[empty, np.where(Ycur[empty] > 0, q[empty], -1).argmax(1)] = True
        Ycur = keep.astype(np.float64)
        q = _rownorm(p * Ycur)
        pseudo = _pseudo_from_q(q, Ycur, pseudo)
    LAST_PSEUDO["y"] = pseudo
    return safe_proba(clf, ctx.Xte, K).argmax(1)


def _ucl_graph(X, k):
    n = len(X); k = min(k, n - 1)
    d, idx = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X).kneighbors(X)
    d, idx = d[:, 1:].astype(np.float64), idx[:, 1:]
    sigma = np.median(d) + 1e-8
    w = np.exp(-d ** 2 / (2 * sigma ** 2))
    w[w.sum(1) < 1e-12] = 1.0
    w = w / w.sum(1, keepdims=True)
    rows = np.repeat(np.arange(n), k)
    return sparse.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))


def p_ucl(ctx, M, K, cfg, seed):
    W = _ucl_graph(ctx.Xtr, cfg.knn_k)
    q = _init_q(M)
    pseudo = _tiebreak_pseudo(ctx, M, K, cfg, seed)
    b = cfg.ucl_beta
    for t in range(1, cfg.pll_T + 1):
        clf = fit_lr(ctx.Xtr, pseudo, cfg, seed=seed)
        p = safe_proba(clf, ctx.Xtr, K)
        qprop = _rownorm((b * (W @ q) + (1 - b) * q) * M)
        H = -(q * np.log(q + 1e-12)).sum(1, keepdims=True)
        g = np.exp(-H)
        q = g * _rownorm(qprop * p * M) + (1 - g) * qprop
        pseudo = _pseudo_from_q(q, M, pseudo)
    LAST_PSEUDO["y"] = pseudo
    return safe_proba(clf, ctx.Xte, K).argmax(1)


class _TerialNet:
    def __init__(self, d, K, hdim, seed):
        r = np.random.RandomState(seed)
        s = lambda a, b: r.randn(a, b) * np.sqrt(2.0 / a)
        self.P = {"WX": s(d, hdim), "WY": s(d, hdim), "b1": np.zeros(hdim),
                  "Wo": s(hdim, K) * 0.5, "b2": np.zeros(K)}
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def forward(self, X, Z):
        P = self.P
        a = X @ P["WX"] + Z @ P["WY"] + P["b1"]
        h = np.maximum(a, 0)
        lo = h @ P["Wo"] + P["b2"]
        lo -= lo.max(1, keepdims=True)
        pr = np.exp(lo); pr /= pr.sum(1, keepdims=True)
        return a, h, pr

    def fit(self, X, Z, y, steps, lr=0.01, l2=1e-4):
        n = len(X); Y = np.eye(self.P["b2"].shape[0])[y]
        for _ in range(steps):
            a, h, pr = self.forward(X, Z)
            g = (pr - Y) / n
            G = {"Wo": h.T @ g + l2 * self.P["Wo"], "b2": g.sum(0)}
            dh = (g @ self.P["Wo"].T) * (a > 0)
            G["WX"] = X.T @ dh + l2 * self.P["WX"]; G["WY"] = Z.T @ dh + l2 * self.P["WY"]
            G["b1"] = dh.sum(0)
            self.t += 1
            for k in self.P:
                self.m[k] = 0.9 * self.m[k] + 0.1 * G[k]
                self.v[k] = 0.999 * self.v[k] + 0.001 * G[k] ** 2
                mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
                self.P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


def p_terial(ctx, M, K, cfg, seed):
    X = ctx.Xtr
    E = _init_q(M)
    pseudo = _tiebreak_pseudo(ctx, M, K, cfg, seed)
    net = _TerialNet(X.shape[1], K, cfg.terial_hidden, seed)
    g = cfg.terial_gamma
    for t in range(1, cfg.pll_T + 1):
        Phi = _prototypes(X, M, K, pseudo)
        net.fit(X, (E * M) @ Phi, pseudo, cfg.terial_steps)
        p = net.forward(X, (E * M) @ Phi)[2]
        pm = p * M
        E = g * E + (1 - g) * pm / (pm.max(1, keepdims=True) + 1e-12)
        E *= M
        pseudo = _pseudo_from_q(E, M, pseudo)
    LAST_PSEUDO["y"] = pseudo


    Mte = getattr(ctx, "Mte", None)
    if Mte is not None:
        Et = _init_q(Mte)
        for _ in range(cfg.terial_test_iter):
            pt = net.forward(ctx.Xte, (Et * Mte) @ Phi)[2] * Mte
            Et = (g * Et + (1 - g) * pt / (pt.max(1, keepdims=True) + 1e-12)) * Mte
        Zt = (Et * Mte) @ Phi
    else:
        Zt = np.full((len(ctx.Xte), K), 1.0 / K) @ Phi
    return net.forward(ctx.Xte, Zt)[2].argmax(1)


PLL_FNS = {"HTC-LTPLL": p_htc, "SoDisam": p_sodisam, "UCL": p_ucl, "TERIAL": p_terial}


def supervised_lr(ctx, y_tr, pos, cfg, seed):
    imb = None if cfg.imbalance_baselines else "none"
    clf = fit_lr(ctx.Xtr[pos], y_tr[pos], cfg, seed=seed, imbalance=imb)
    return clf.predict(ctx.Xte)


def _score(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro", zero_division=0)


def derived_seed(cfg: Config, ds_i: int, fold: int, placement: int) -> int:
    return int(cfg.master_seed + 100003 * ds_i + 1009 * fold + 17 * placement)


def _proxy_q(mode):
    return int(mode.split("_q")[1]) / 100.0


def build_candidates(mode, ds, tr, X_tr, y_tr, pos, K, cfg, tau, rseed, X_te=None):
    if mode.startswith("proxy_q"):
        if ds.count is None:
            raise ValueError(f"{ds.name}: no person counts loaded, cannot build {mode}")
        return proxy_candidates(ds.count[tr], y_tr, pos, ds.bins, K, _proxy_q(mode), cfg, rs=rseed), None
    return dtcg_candidates(X_tr, y_tr, pos, K, cfg, tau=tau, mode=mode, rs=rseed, X_te=X_te)


def pseudo_diagnostics(pseudo, M, y_tr, pos, K) -> dict:
    if pseudo is None:
        return {}
    unl = np.ones(len(y_tr), bool); unl[pos] = False
    if unl.sum() == 0:
        return {}
    inset = M[np.arange(len(y_tr)), y_tr] > 0
    ok = pseudo == y_tr
    out = {"pseudo_acc_unl": float(ok[unl].mean()),
           "pseudo_acc_unl_inset": float(ok[unl & inset].mean()) if (unl & inset).any() else np.nan,
           "pseudo_acc_unl_excluded": float(ok[unl & ~inset].mean()) if (unl & ~inset).any() else np.nan,
           "share_unl_excluded": float((unl & ~inset).sum() / unl.sum())}
    for c in range(K):
        m = unl & (y_tr == c)
        out[f"pseudo_recall_unl_c{c}"] = float(ok[m].mean()) if m.any() else np.nan
        out[f"pseudo_pred_share_c{c}"] = float((pseudo[unl] == c).mean())
    return out


def run_experiment(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    budgets = (0.10, 1.00) if cfg.quick else cfg.budgets
    metric_rows, cand_rows, pred_rows, fold_rows, tau_rows = [], [], [], [], []
    t_start = time.time()

    for ds_i, name in enumerate(cfg.datasets):
        ds = load_dataset(cfg, name)
        K = 4
        folds = make_folds(ds, cfg)
        spd = ds.steps_per_day
        n_pl = cfg.placements(name)
        print(f"\n[{name}] n={len(ds.y)} d={ds.X.shape[1]} folds={len(folds)} "
              f"protocol={cfg.protocol} segments={len(np.unique(ds.segment))} "
              f"imbalance={cfg.imbalance} tau_mode={cfg.tau_mode}", flush=True)

        for f in folds:
            tr, te = f["tr"], f["te"]
            X_tr, X_te, y_tr, y_te = ds.X[tr], ds.X[te], ds.y[tr], ds.y[te]
            fold_rows.append({"dataset": name, "protocol": cfg.protocol, "fold": f["fold"],
                              **{k: f[k] for k in ["test_start", "test_end", "n_train", "n_test",
                                                   "n_test_segments"]},
                              **{f"test_n_c{c}": int((y_te == c).sum()) for c in range(K)}})
            ctx = FoldContext(X_tr, X_te, cfg)

            def log_pred(method, b, pl, mode, pred, seed_days, n_seed, tau=np.nan, extra=None):
                acc, f1 = _score(y_te, pred)
                extra = extra or {}
                metric_rows.append({"dataset": name, "fold": f["fold"], "placement": pl, "budget": b,
                                    "cand_mode": mode, "method": method, "accuracy": acc,
                                    "macro_f1": f1, "n_seed": n_seed, "seed_days": seed_days,
                                    "n_train": len(tr), "n_test": len(te), "tau": tau,
                                    "imbalance": cfg.imbalance,
                                    **{f"smote_{k}": v for k, v in SMOTE_STATS.items()}, **extra})
                pred_rows.append(pd.DataFrame({
                    "dataset": name, "fold": f["fold"], "placement": pl, "budget": b,
                    "cand_mode": mode, "method": method, "row": te, "y_true": y_te, "y_pred": pred}))


            full = np.arange(len(tr))
            reset_smote_stats()
            log_pred("Supervised LR", 1.0, -1, "none",
                     supervised_lr(ctx, y_tr, full, cfg, seed=cfg.master_seed), len(tr) / spd, len(tr))

            for pl in range(n_pl):
                rseed = derived_seed(cfg, ds_i, f["fold"], pl)
                for b in budgets:
                    pos, cid = seed_indices(len(tr), spd, cfg.chunk_days(name), b, rseed,
                                            return_chunks=True)
                    sd = len(pos) / spd
                    if b < 1.0:
                        reset_smote_stats()
                        log_pred("Supervised LR (seed only)", b, pl, "none",
                                 supervised_lr(ctx, y_tr, pos, cfg, rseed), sd, len(pos))
                    if b >= 1.0 and pl > 0:
                        continue

                    if cfg.tau_mode == "select":
                        tau, tinfo = select_tau(X_tr, y_tr, pos, cid, K, cfg, rs=rseed)
                    else:
                        tau, tinfo = cfg.tau, {"tau_selected": cfg.tau, "rule": "fixed",
                                               "n_seed_chunks": int(len(np.unique(cid)))}
                    tau_rows.append({"dataset": name, "fold": f["fold"], "placement": pl,
                                     "budget": b, "n_seed": len(pos), "seed_days": sd, **tinfo})
                    methods = cfg.full_methods if b >= 1.0 else cfg.budget_methods
                    for mode in cfg.cand_modes:
                        M, M_te = build_candidates(mode, ds, tr, X_tr, y_tr, pos, K, cfg, tau, rseed,
                                                   X_te=X_te)
                        row = {"dataset": name, "fold": f["fold"], "placement": pl,
                               "budget": b, "cand_mode": mode,
                               "tau": np.nan if mode.startswith("proxy") else tau,
                               "n_seed": len(pos), "seed_days": sd,
                               **candidate_diagnostics(M, y_tr, pos, K)}
                        if M_te is not None:
                            row.update(set_diagnostics(M_te, y_te, np.ones(len(y_te), bool), K, "test"))
                        cand_rows.append(row)
                        ctx.Mte = M_te
                        for m in methods:
                            reset_smote_stats()
                            LAST_PSEUDO["y"] = None
                            pred = PLL_FNS[m](ctx, M, K, cfg, rseed)
                            log_pred(m, b, pl, mode, pred, sd, len(pos),
                                     tau=np.nan if mode.startswith("proxy") else tau,
                                     extra=pseudo_diagnostics(LAST_PSEUDO["y"], M, y_tr, pos, K))
            print(f"  fold {f['fold']+1}/{len(folds)} [{f['test_start'][:10]} .. "
                  f"{f['test_end'][:10]}] done  ({time.time()-t_start:.0f}s)", flush=True)


        pd.DataFrame(metric_rows).to_csv(os.path.join(cfg.out_dir, "fold_metrics.csv"), index=False)
        pd.DataFrame(cand_rows).to_csv(os.path.join(cfg.out_dir, "candidate_diagnostics.csv"), index=False)
        pd.DataFrame(fold_rows).to_csv(os.path.join(cfg.out_dir, "fold_dates.csv"), index=False)
        pd.DataFrame(tau_rows).to_csv(os.path.join(cfg.out_dir, "tau_selection.csv"), index=False)
        pd.concat(pred_rows).to_csv(os.path.join(cfg.out_dir, "predictions.csv.gz"), index=False)
    with open(os.path.join(cfg.out_dir, "run_config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2, default=str)
    print("\nDone. Outputs in", cfg.out_dir)


def run_tau_sweep(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    rows = []
    for ds_i, name in enumerate(cfg.datasets):
        ds = load_dataset(cfg, name); K = 4; spd = ds.steps_per_day
        for f in make_folds(ds, cfg):
            tr, te = f["tr"], f["te"]
            X_tr, y_tr, y_te = ds.X[tr], ds.y[tr], ds.y[te]
            ctx = FoldContext(X_tr, ds.X[te], cfg) if cfg.tau_run_pll else None
            for pl in range(cfg.placements(name)):
                rseed = derived_seed(cfg, ds_i, f["fold"], pl)
                for b in cfg.tau_budgets:
                    pos, cid = seed_indices(len(tr), spd, cfg.chunk_days(name), b, rseed,
                                            return_chunks=True)
                    tau_sel, _ = select_tau(X_tr, y_tr, pos, cid, K, cfg, rs=rseed)
                    for tau in sorted(set(cfg.tau_grid) | set(cfg.tau_pll_grid)):
                        M, M_te = dtcg_candidates(X_tr, y_tr, pos, K, cfg, tau=tau, rs=rseed,
                                                  X_te=ds.X[te])
                        r = {"dataset": name, "fold": f["fold"], "placement": pl, "budget": b,
                             "tau": tau, "tau_selected_rule": tau_sel,
                             **candidate_diagnostics(M, y_tr, pos, K),
                             **set_diagnostics(M_te, y_te, np.ones(len(y_te), bool), K, "test")}
                        if ctx is not None and any(np.isclose(tau, t) for t in cfg.tau_pll_grid):
                            reset_smote_stats()
                            acc, f1 = _score(y_te, PLL_FNS["SoDisam"](ctx, M, K, cfg, rseed))
                            r.update(accuracy=acc, macro_f1=f1)
                        rows.append(r)
            print(f"[tau] {name} fold {f['fold']+1} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(cfg.out_dir, "tau_sweep.csv"), index=False)


def run_audit(cfg: Config, lr_check: bool):
    os.makedirs(cfg.out_dir, exist_ok=True)
    rows, cut_rows, chan_rows = [], [], []
    for name in cfg.datasets:
        ds = load_dataset(cfg, name)
        cnt = np.bincount(ds.y, minlength=4)
        t = ds.times
        edges = ds.bins
        for c in range(4):
            lo = 0 if c == 0 else int(np.floor(edges[c])) + 1
            hi = "inf" if c == 3 else int(np.floor(edges[c + 1]))
            if c == 0: lo, hi = 0, 0
            cut_rows.append({"dataset": name, "class": CLASS_NAMES[c], "persons": f"{lo}-{hi}",
                             "n": int(cnt[c]), "pct": round(100 * cnt[c] / cnt.sum(), 1)})
        spans = pd.Series(ds.segment).value_counts().sort_index() / ds.steps_per_day
        rows.append({"dataset": name, "rows": len(ds.y), "features": ds.X.shape[1],
                     "step_min": ds.step.total_seconds() / 60,
                     "first": str(t.iloc[0]), "last": str(t.iloc[-1]),
                     "days_of_records": round(len(ds.y) / ds.steps_per_day, 1),
                     "segments": int(spans.size), "longest_segment_days": round(spans.max(), 1),
                     **{"seed10_days": round(0.1 * len(ds.y) / ds.steps_per_day, 1),
                        "seed40_days": round(0.4 * len(ds.y) / ds.steps_per_day, 1)},
                     **ds.info})

        fixed = FLOOR_S_LAG_CHANNELS if name == "Floor_S" else [c for c in ROBOD_LAG_CHANNELS]
        raw_idx = [i for i, n in enumerate(ds.feature_names) if "_lag" not in n and "_roll" not in n]
        for f in make_folds(ds, cfg):
            Xr, y = ds.X[f["tr"]][:, raw_idx], ds.y[f["tr"]]
            corr = np.array([abs(np.corrcoef(Xr[:, j], y)[0, 1]) if Xr[:, j].std() > 0 else 0
                             for j in range(Xr.shape[1])])
            top8 = [ds.feature_names[raw_idx[j]] for j in np.argsort(-corr)[:8]]
            chan_rows.append({"dataset": name, "fold": f["fold"], "top8_in_fold": "|".join(top8),
                              "overlap_with_fixed_list": len(set(top8) & set(fixed))})
        print(f"[audit] {name} ok", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(cfg.out_dir, "audit_datasets.csv"), index=False)
    pd.DataFrame(cut_rows).to_csv(os.path.join(cfg.out_dir, "audit_class_cutoffs.csv"), index=False)
    pd.DataFrame(chan_rows).to_csv(os.path.join(cfg.out_dir, "audit_channel_selection.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(cut_rows).to_string(index=False))

    if lr_check:
        print("\nLR sanity check (plan reports Floor S: shuffled 10f 0.693/0.549, blocked 10f 0.662/0.491)")
        out = []
        for name in cfg.datasets:
            ds = load_dataset(cfg, name)
            for proto, K in [("shuffled", cfg.n_folds(name)), ("blocked", cfg.n_folds(name)),
                             ("blocked", 5)]:
                c2 = Config(**{**asdict(cfg), "protocol": proto, "quick": False})
                c2.n_folds_floor = c2.n_folds_robod = K
                a, fs = [], []
                for f in make_folds(ds, c2):
                    sc = StandardScaler().fit(ds.X[f["tr"]])
                    clf = fit_lr(sc.transform(ds.X[f["tr"]]), ds.y[f["tr"]], cfg)
                    p = clf.predict(sc.transform(ds.X[f["te"]]))
                    ac, f1 = _score(ds.y[f["te"]], p); a.append(ac); fs.append(f1)
                out.append({"dataset": name, "protocol": proto, "folds": K,
                            "acc": f"{np.mean(a):.3f} +/- {np.std(a):.3f}",
                            "macro_f1": f"{np.mean(fs):.3f} +/- {np.std(fs):.3f}"})
                print(out[-1], flush=True)
        pd.DataFrame(out).to_csv(os.path.join(cfg.out_dir, "audit_lr_check.csv"), index=False)


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (n - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def paired_noninferiority(diff, n_train, n_test, margin, alpha):
    K = len(diff); m = diff.mean(); s = diff.std(ddof=1) if K > 1 else np.nan
    se = np.sqrt((1 / K + n_test / n_train) * s ** 2) if K > 1 else np.nan
    tcrit = stats.t.ppf(1 - alpha, K - 1) if K > 1 else np.nan
    t_stat = (m + margin) / se if se and se > 0 else np.nan
    p = 1 - stats.t.cdf(t_stat, K - 1) if K > 1 and np.isfinite(t_stat) else np.nan
    d = m / s if s and s > 0 else np.nan
    return {"mean_diff": m, "se": se, "lower_1sided": m - tcrit * se, "p_noninf": p, "cohens_d": d, "K": K}


def run_report(cfg: Config):
    fm = pd.read_csv(os.path.join(cfg.out_dir, "fold_metrics.csv"))
    pr = pd.read_csv(os.path.join(cfg.out_dir, "predictions.csv.gz"))

    rows = []
    for mode in [m for m in fm.cand_mode.unique() if m != "none"]:
        for ds, g in fm.groupby("dataset"):
            ref = g[g.method == "Supervised LR"].set_index("fold")
            for (method, b), h in g[(g.cand_mode == mode) & (g.method.isin(cfg.full_methods))
                                    & (g.budget < 1.0)].groupby(["method", "budget"]):
                per_fold = h.groupby("fold")[["accuracy", "macro_f1", "n_train", "n_test"]].mean()
                same_seed = g[(g.method == "Supervised LR (seed only)") & (g.budget == b)] \
                    .groupby("fold")[["accuracy", "macro_f1"]].mean()
                for metric, margin in [("accuracy", cfg.margin_acc), ("macro_f1", cfg.margin_f1)]:
                    for refname, refvals in [("LR_full", ref[metric]), ("LR_same_seed", same_seed[metric])]:
                        d = (per_fold[metric] - refvals).dropna()
                        if len(d) < 3: continue
                        r = paired_noninferiority(d.values, per_fold.n_train.mean(),
                                                  per_fold.n_test.mean(), margin, cfg.alpha)
                        n_pl = h.placement.nunique()
                        run_sd = h.groupby("placement")[metric].mean().std(ddof=1) if n_pl > 1 else np.nan
                        rows.append({"cand_mode": mode, "dataset": ds, "method": method, "budget": b,
                                     "metric": metric, "reference": refname, "margin": margin,
                                     "placement_sd_of_mean": run_sd, **r,
                                     "noninferior": bool(r["lower_1sided"] > -margin)})
    st = pd.DataFrame(rows)
    if len(st):
        st["p_holm"] = np.nan

        for _, idx in st.groupby(["cand_mode", "metric", "reference"]).groups.items():
            st.loc[idx, "p_holm"] = holm(st.loc[idx, "p_noninf"].fillna(1).values)
        st.to_csv(os.path.join(cfg.out_dir, "stats_noninferiority.csv"), index=False)

        conv = []
        for (mode, ds, method, metric), g in st[st.reference == "LR_full"].groupby(
                ["cand_mode", "dataset", "method", "metric"]):
            g = g.sort_values("budget")
            ok = g.noninferior.values; bs = g.budget.values
            first = next((bs[i] for i in range(len(bs)) if ok[i:].all()), np.nan)
            conv.append({"cand_mode": mode, "dataset": ds, "method": method, "metric": metric,
                         "convergence_budget": first})
        pd.DataFrame(conv).to_csv(os.path.join(cfg.out_dir, "convergence_budget.csv"), index=False)


    pc = []
    for keys, g in pr.groupby(["dataset", "cand_mode", "method", "budget"]):
        yt, yp = g.y_true.values, g.y_pred.values
        p, r, f1, sup = _prf(yt, yp)
        for c in range(4):
            pc.append(dict(zip(["dataset", "cand_mode", "method", "budget"], keys),
                           cls=CLASS_NAMES[c], precision=p[c], recall=r[c], f1=f1[c], support=sup[c]))
        pc.append(dict(zip(["dataset", "cand_mode", "method", "budget"], keys), cls="POOLED_macro",
                       precision=np.nanmean(p), recall=np.nanmean(r), f1=np.nanmean(f1), support=sup.sum()))
    pd.DataFrame(pc).to_csv(os.path.join(cfg.out_dir, "per_class_metrics.csv"), index=False)
    cms = {}
    for keys, g in pr.groupby(["dataset", "cand_mode", "method", "budget"]):
        cms["|".join(map(str, keys))] = confusion_matrix(g.y_true, g.y_pred, labels=range(4)).tolist()
    with open(os.path.join(cfg.out_dir, "confusion_matrices.json"), "w") as fh:
        json.dump(cms, fh)
    print("report written to", cfg.out_dir)


def _prf(yt, yp):
    from sklearn.metrics import precision_recall_fscore_support as prfs
    p, r, f, s = prfs(yt, yp, labels=range(4), zero_division=0)
    p, r, f = p.astype(float), r.astype(float), f.astype(float)
    absent = s == 0
    p[absent & (np.bincount(yp, minlength=4) == 0)] = np.nan
    r[absent] = np.nan; f[absent] = np.nan
    return p, r, f, s


def _tcrit_ci(x, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x)); t = stats.t.ppf(1 - alpha / 2, len(x) - 1)
    return x.mean() - t * se, x.mean() + t * se


def _mean_by(df, keys, cols):
    cols = [c for c in cols if c in df.columns]
    return df.groupby(keys)[cols].mean().reset_index()


def make_coverage_tables(cfg: Config):
    out = cfg.out_dir
    K = 4
    cd_all = pd.read_csv(os.path.join(out, "candidate_diagnostics.csv"))
    cd = cd_all[(cd_all.cand_mode == "seed_only") & (cd_all.budget < 1.0)]


    cols = (["tau", "recall_unl", "ambig_rate_unl", "mean_size_unl", "singleton_frac_unl",
             "singleton_err_unl", "recall_test", "singleton_err_test"]
            + [f"recall_c{c}" for c in range(K)] + [f"singleton_err_c{c}" for c in range(K)]
            + [f"singleton_frac_c{c}" for c in range(K)] + [f"n_unl_c{c}" for c in range(K)]
            + [f"test_recall_c{c}" for c in range(K)])
    cov = _mean_by(cd, ["dataset", "budget"], cols).round(3)
    cov.to_csv(os.path.join(out, "table_coverage_by_class_budget.csv"), index=False)


    long = []
    for _, r in cov.iterrows():
        for c in range(K):
            long.append({"dataset": r.dataset, "budget": r.budget, "class": CLASS_NAMES[c],
                         "n_unlabeled_per_fold": r.get(f"n_unl_c{c}"),
                         "coverage_unlabeled": r.get(f"recall_c{c}"),
                         "coverage_test_fold": r.get(f"test_recall_c{c}"),
                         "singleton_frac": r.get(f"singleton_frac_c{c}"),
                         "singleton_err": r.get(f"singleton_err_c{c}")})
    pd.DataFrame(long).to_csv(os.path.join(out, "table_coverage_long.csv"), index=False)


    pc = _mean_by(cd, ["dataset", "budget", "placement"],
                  ["recall_unl", "recall_c3", "ambig_rate_unl", "singleton_err_unl", "tau"]).round(3)
    pc.to_csv(os.path.join(out, "table_coverage_by_placement.csv"), index=False)
    if pc.placement.nunique() > 1:
        sp = pc.groupby(["dataset", "budget"])[["recall_unl", "recall_c3", "ambig_rate_unl"]] \
            .agg(["min", "max", "std"]).round(3)
        sp.columns = ["_".join(c) for c in sp.columns]
        sp.reset_index().to_csv(os.path.join(out, "table_coverage_placement_spread.csv"), index=False)


    tp = os.path.join(out, "tau_selection.csv")
    if os.path.exists(tp):
        ts = pd.read_csv(tp)
        ts = ts[ts.budget < 1.0]
        g = ts.groupby(["dataset", "budget"])
        tsum = pd.DataFrame({
            "tau_median": g.tau_selected.median(), "tau_min": g.tau_selected.min(),
            "tau_max": g.tau_selected.max(),
            "share_target_met": g.rule.apply(lambda x: (x == "target_met").mean()),
            "share_fallback": g.rule.apply(lambda x: x.str.startswith("fallback").mean()),
            "n_seed_chunks_mean": g.n_seed_chunks.mean()}).reset_index()

        def inner_at_sel(r):
            k = f"innercov_tau{r.tau_selected:.2f}"
            return r[k] if k in r and pd.notna(r[k]) else np.nan
        if any(c.startswith("innercov_") for c in ts.columns):
            ts = ts.assign(innercov_selected=ts.apply(inner_at_sel, axis=1))
            tsum = tsum.merge(ts.groupby(["dataset", "budget"]).innercov_selected.mean().reset_index(),
                              on=["dataset", "budget"])
            real = cd.groupby(["dataset", "budget"]).recall_unl.mean().reset_index() \
                .rename(columns={"recall_unl": "realised_cov_unlabeled"})
            tsum = tsum.merge(real, on=["dataset", "budget"], how="left")
        tsum.round(3).to_csv(os.path.join(out, "table_tau_selection.csv"), index=False)


    fm = pd.read_csv(os.path.join(out, "fold_metrics.csv"), keep_default_na=False)
    for c in ["placement", "accuracy", "macro_f1", "budget"]:
        fm[c] = pd.to_numeric(fm[c])
    g = fm.groupby(["dataset", "method", "cand_mode", "budget", "fold"])[["accuracy", "macro_f1"]].mean().reset_index()
    sod = g[(g.method == "SoDisam") & (g.budget < 1.0)]
    lrs = g[g.method == "Supervised LR (seed only)"].groupby(["dataset", "budget"])[["accuracy", "macro_f1"]].mean() \
        .add_suffix("_LR_same_seed").reset_index()
    lrf = g[g.method == "Supervised LR"].groupby("dataset")[["accuracy", "macro_f1"]].mean() \
        .add_suffix("_LR_full").reset_index()
    cm = sod.groupby(["dataset", "budget", "cand_mode"])[["accuracy", "macro_f1"]].mean().reset_index()
    cdm = cd_all[cd_all.budget < 1.0].groupby(["dataset", "budget", "cand_mode"])[
        ["recall_unl", "ambig_rate_unl", "mean_size_unl"]].mean().reset_index()
    cm = cm.merge(cdm, on=["dataset", "budget", "cand_mode"], how="left") \
        .merge(lrs, on=["dataset", "budget"], how="left").merge(lrf, on="dataset", how="left")
    pcol = [c for c in fm.columns if c.startswith("pseudo_acc")]
    if pcol:
        pa = fm[(fm.method == "SoDisam") & (fm.budget < 1.0)].copy()
        for c in pcol + ["share_unl_excluded"]:
            pa[c] = pd.to_numeric(pa[c], errors="coerce")
        cm = cm.merge(pa.groupby(["dataset", "budget", "cand_mode"])[pcol + ["share_unl_excluded"]].mean()
                      .reset_index(), on=["dataset", "budget", "cand_mode"], how="left")
    cm.round(4).to_csv(os.path.join(out, "table_cand_modes.csv"), index=False)


    rows = []
    for (d, b), h in sod.groupby(["dataset", "budget"]):
        o = h[h.cand_mode == "oracle_true"].set_index("fold"); so = h[h.cand_mode == "seed_only"].set_index("fold")
        if len(o) == 0 or len(so) == 0:
            continue
        r = {"dataset": d, "budget": b}
        for met in ["accuracy", "macro_f1"]:
            diff = (o[met] - so[met]).dropna()
            lo, hi = _tcrit_ci(diff.values)
            r.update({f"{met}_oracle": o[met].mean(), f"{met}_seed_only": so[met].mean(),
                      f"{met}_drop_when_excluded": diff.mean(), f"{met}_drop_ci_lo": lo,
                      f"{met}_drop_ci_hi": hi})
        rows.append(r)
    orc = pd.DataFrame(rows)
    if len(orc):
        orc = orc.merge(lrs, on=["dataset", "budget"], how="left").merge(lrf, on="dataset", how="left")
        orc = orc.rename(columns={"accuracy_LR_same_seed": "accuracy_LR_same_seed",
                                  "macro_f1_LR_same_seed": "macro_f1_LR_same_seed"})
    orc.round(4).to_csv(os.path.join(out, "table_oracle_vs_seed_only.csv"), index=False)


    sc = [c for c in fm.columns if c.startswith("smote_")]
    if sc:
        for c in sc:
            fm[c] = pd.to_numeric(fm[c], errors="coerce")
        fm.groupby(["dataset", "method", "budget"])[sc].mean().round(1).reset_index() \
            .to_csv(os.path.join(out, "table_smote_log.csv"), index=False)
    return cov, cm


def make_figures(cfg: Config):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available, skipping figures:", e); return
    out = cfg.out_dir
    cov = pd.read_csv(os.path.join(out, "table_coverage_by_class_budget.csv"))
    cm = pd.read_csv(os.path.join(out, "table_cand_modes.csv"))
    names = list(cov.dataset.unique())
    colors = dict(zip(names, plt.cm.tab10(np.arange(len(names)))))
    sel = "seed-only tau rule" if cfg.tau_mode == "select" else f"tau = {cfg.tau}"


    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for d in names:
        h = cov[cov.dataset == d]
        ax[0].plot(h.budget * 100, h.recall_unl, "o-", color=colors[d], label=d)
        ax[1].plot(h.budget * 100, h.recall_c3, "o-", color=colors[d], label=d)
        if "recall_test" in h:
            ax[0].plot(h.budget * 100, h.recall_test, "--", color=colors[d], alpha=.6)
        if "test_recall_c3" in h:
            ax[1].plot(h.budget * 100, h.test_recall_c3, "--", color=colors[d], alpha=.6)
    ax[0].set_title("True-label coverage, all classes"); ax[1].set_title("True-label coverage, High class")
    for a in ax:
        a.set_xlabel("Labeled seed (% of training fold)"); a.set_ylim(0, 1.02); a.grid(alpha=.3)
        a.axhline(1.0, color="k", ls=":", lw=.8)
    ax[0].set_ylabel("Coverage"); ax[0].legend(fontsize=8)
    fig.suptitle(f"DT-CG coverage ({sel}); solid = unlabeled training rows, dashed = test fold")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_coverage_recall.png"), dpi=200); plt.close(fig)


    styles = {"oracle_true": ("s--", "tab:gray", "SoDisam, true label in every set (old code)"),
              "seed_only": ("o-", "tab:red", "SoDisam, DT-CG seed-only sets")}
    pal = iter(["tab:green", "tab:purple", "tab:olive", "tab:cyan"])
    for m in sorted(cm.cand_mode.unique()):
        if m.startswith("proxy_q"):
            styles[m] = ("D-", next(pal), f"SoDisam, independent proxy sets, coverage {_proxy_q(m):.0%}")
    n = len(names); ncol = 3; nrow = int(np.ceil(n / ncol))
    for met in ["accuracy", "macro_f1"]:
        fig, axs = plt.subplots(nrow, ncol, figsize=(13, 3.6 * nrow)); axs = np.atleast_1d(axs).ravel()
        for i, d in enumerate(names):
            a = axs[i]; h = cm[cm.dataset == d]
            for m, (mk, col, lab) in styles.items():
                hm = h[h.cand_mode == m].sort_values("budget")
                if len(hm):
                    a.plot(hm.budget * 100, hm[met], mk, color=col, label=lab)
            lr = h[h.cand_mode == "seed_only"].sort_values("budget")
            if len(lr):
                a.plot(lr.budget * 100, lr[f"{met}_LR_same_seed"], "^-", color="tab:blue", label="LR, same labeled seed")
                a.axhline(lr[f"{met}_LR_full"].iloc[0], color="k", ls=":", label="LR, all labels")
            a.set_title(d); a.set_xlabel("Labeled seed (%)"); a.set_ylabel(met); a.grid(alpha=.3)
        for j in range(i + 1, len(axs)): axs[j].axis("off")
        axs[0].legend(fontsize=6, loc="lower right")
        fig.suptitle(f"SoDisam by candidate-set source, {met} (separate y-axis per panel)")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_cand_modes_{met}.png"), dpi=200); plt.close(fig)


    tp = os.path.join(out, "tau_sweep.csv")
    if os.path.exists(tp):
        ts = pd.read_csv(tp)
        vals = [c for c in ["recall_unl", "ambig_rate_unl", "mean_size_unl", "accuracy", "macro_f1"] if c in ts]
        tg = ts.groupby(["dataset", "budget", "tau"])[vals].mean().reset_index()
        panels = [c for c in ["recall_unl", "ambig_rate_unl", "mean_size_unl", "macro_f1"] if c in tg]
        fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
        for d in tg.dataset.unique():
            for b, ls in zip(sorted(tg.budget.unique()), ["-", "--"]):
                h = tg[(tg.dataset == d) & (tg.budget == b)]
                for k, col in enumerate(panels):
                    hh = h.dropna(subset=[col])
                    ax[k].plot(hh.tau, hh[col], ls, marker="o", ms=3, color=colors.get(d, None),
                               label=f"{d} ({int(b*100)}%)" if k == 0 else None)
        titles = {"recall_unl": "True-label coverage (unlabeled)", "ambig_rate_unl": "Ambiguity rate (|set|>1)",
                  "mean_size_unl": "Mean candidate-set size", "macro_f1": "SoDisam macro-F1 (test)"}
        for k, c in enumerate(panels):
            ax[k].set_title(titles[c]); ax[k].set_xlabel("tau"); ax[k].grid(alpha=.3)
        ax[0].legend(fontsize=6, ncol=2)
        fig.suptitle("Tau sensitivity: solid = 10% seed, dashed = 40% seed")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_tau_sweep.png"), dpi=200); plt.close(fig)
        tg.round(3).to_csv(os.path.join(out, "table_tau_sweep_summary.csv"), index=False)
    print("figures + tables written to", out)


def _subcfg(cfg: Config, sub: str, **kw) -> Config:
    c = Config(**{**asdict(cfg), **kw})
    c.out_dir = os.path.join(cfg.out_dir, sub)
    return c


def run_decomposition(cfg: Config):
    sc = _subcfg(cfg, "decomp_shuffled", protocol="shuffled", budgets=tuple(cfg.decomp_budgets),
                 cand_modes=("seed_only", "oracle_true"))
    run_experiment(sc)
    rows = []
    for proto, d in [("blocked", cfg.out_dir), ("shuffled", sc.out_dir)]:
        fm = pd.read_csv(os.path.join(d, "fold_metrics.csv"), keep_default_na=False)
        for c in ["accuracy", "macro_f1", "budget"]:
            fm[c] = pd.to_numeric(fm[c])
        fm = fm[fm.budget.isin(list(cfg.decomp_budgets)) | (fm.method == "Supervised LR")]
        g = fm.groupby(["dataset", "method", "cand_mode", "budget"])[["accuracy", "macro_f1"]].mean().reset_index()
        for _, r in g[g.method == "SoDisam"].iterrows():
            lr = g[(g.dataset == r.dataset) & (g.method == "Supervised LR")]
            ls = g[(g.dataset == r.dataset) & (g.method == "Supervised LR (seed only)") & (g.budget == r.budget)]
            rows.append({"dataset": r.dataset, "budget": r.budget, "protocol": proto,
                         "cand_mode": r.cand_mode, "sodisam_acc": r.accuracy, "sodisam_f1": r.macro_f1,
                         "LR_full_acc": lr.accuracy.mean(), "LR_full_f1": lr.macro_f1.mean(),
                         "LR_same_seed_acc": ls.accuracy.mean(), "LR_same_seed_f1": ls.macro_f1.mean()})
    t = pd.DataFrame(rows).sort_values(["dataset", "budget", "protocol", "cand_mode"])
    t.round(4).to_csv(os.path.join(cfg.out_dir, "table_protocol_decomposition.csv"), index=False)
    print("decomposition written to", cfg.out_dir)


def run_imbalance_sensitivity(cfg: Config, modes=None):
    if modes is None:
        modes = [m for m in ("none", "smote", "weight") if m != cfg.imbalance]
    dirs = [(cfg.imbalance, cfg.out_dir)]
    for imb in modes:
        sc = _subcfg(cfg, f"imbalance_{imb}", imbalance=imb, cand_modes=("seed_only",))
        run_experiment(sc); run_report(sc)
        dirs.append((imb, sc.out_dir))
    rows = []
    for imb, d in dirs:
        pc = pd.read_csv(os.path.join(d, "per_class_metrics.csv"), keep_default_na=False)
        pc = pc[pc.cand_mode.isin(["none", "seed_only"])]
        for c in ["budget", "precision", "recall", "f1"]:
            pc[c] = pd.to_numeric(pc[c], errors="coerce")
        piv = pc.pivot_table(index=["dataset", "method", "budget"], columns="cls", values="f1").reset_index()
        piv.columns.name = None
        fm = pd.read_csv(os.path.join(d, "fold_metrics.csv"), keep_default_na=False)
        fm = fm[fm.cand_mode.isin(["none", "seed_only"])]
        for c in ["accuracy", "macro_f1", "budget"]:
            fm[c] = pd.to_numeric(fm[c])
        g = fm.groupby(["dataset", "method", "budget"])[["accuracy", "macro_f1"]].mean().reset_index()
        g = g.merge(piv, on=["dataset", "method", "budget"], how="left")
        g.insert(0, "imbalance", imb)
        rows.append(g)
    pd.concat(rows).round(4).to_csv(os.path.join(cfg.out_dir, "table_imbalance_sensitivity.csv"), index=False)
    print("imbalance sensitivity written to", cfg.out_dir)


def run_tau_select_sensitivity(cfg: Config):
    sc = _subcfg(cfg, "tau_select", tau_mode="select", cand_modes=("seed_only",))
    run_experiment(sc); run_report(sc)
    rows = []
    for lab, d in [(f"fixed_{cfg.tau}", cfg.out_dir), ("seed_only_rule", sc.out_dir)]:
        fm = pd.read_csv(os.path.join(d, "fold_metrics.csv"), keep_default_na=False)
        fm = fm[(fm.cand_mode == "seed_only") & (fm.method == "SoDisam")]
        for c in ["accuracy", "macro_f1", "budget"]:
            fm[c] = pd.to_numeric(fm[c])
        cd = pd.read_csv(os.path.join(d, "candidate_diagnostics.csv"))
        cd = cd[cd.cand_mode == "seed_only"]
        g = fm.groupby(["dataset", "budget"])[["accuracy", "macro_f1"]].mean() \
            .join(cd.groupby(["dataset", "budget"])[["tau", "recall_unl", "ambig_rate_unl"]].mean())
        g = g.reset_index(); g.insert(0, "tau_setting", lab); rows.append(g)
    pd.concat(rows).round(4).to_csv(os.path.join(cfg.out_dir, "table_tau_fixed_vs_rule.csv"), index=False)
    print("tau rule sensitivity written to", cfg.out_dir)


def run_replication(cfg: Config):
    sc = _subcfg(cfg, "replication_old_settings", protocol="shuffled", lag_mode="legacy",
                 cand_modes=("oracle_true", "seed_only"), tau_mode="fixed", tau=0.25,
                 imbalance="none", budgets=tuple(cfg.decomp_budgets) + (1.0,),
                 chunk_days_floor=1e-9, chunk_days_robod=1e-9)
    run_experiment(sc); run_report(sc)
    fm = pd.read_csv(os.path.join(sc.out_dir, "fold_metrics.csv"), keep_default_na=False)
    for c in ["accuracy", "macro_f1", "budget"]:
        fm[c] = pd.to_numeric(fm[c])
    fm.groupby(["dataset", "method", "cand_mode", "budget"])[["accuracy", "macro_f1"]] \
        .agg(["mean", "std"]).round(4).to_csv(os.path.join(cfg.out_dir, "table_replication.csv"))
    print("replication written to", cfg.out_dir)


def run_coverage_pack(cfg: Config):
    if tuple(cfg.cand_modes) == ("seed_only",):
        cfg.cand_modes = ("seed_only", "oracle_true", "proxy_q90", "proxy_q95")
    run_experiment(cfg)
    run_report(cfg)
    run_tau_sweep(cfg)
    make_coverage_tables(cfg)
    make_figures(cfg)


IDE_MODE       = "revision"
IDE_DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd(), "data")
IDE_OUT_DIR    = os.path.join(os.path.dirname(IDE_DATA_DIR), "results_revision4_paper")
IDE_QUICK      = False
IDE_LR_CHECK   = False
IDE_PLACEMENTS = 2
IDE_DATASETS   = ["Floor_S", "ROBOD_R1", "ROBOD_R2", "ROBOD_R3", "ROBOD_R4", "ROBOD_R5"]
IDE_BUDGETS    = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
IDE_CAND_MODES = "seed_only,oracle_true,proxy_q90,proxy_q95"
IDE_IMBALANCE  = "none"
IDE_TAU_MODE   = "fixed"
IDE_TAU        = 0.25
IDE_TAU_TARGET = 0.90


def run_hardlabel(cfg: Config, main_dir: Optional[str] = None):
    out = os.path.join(cfg.out_dir, "hardlabel"); os.makedirs(out, exist_ok=True)
    rows = []; t0 = time.time()
    budgets = [b for b in cfg.budgets if b < 1.0]
    for ds_i, name in enumerate(cfg.datasets):
        ds = load_dataset(cfg, name); K = 4; spd = ds.steps_per_day
        for f in make_folds(ds, cfg):
            tr, te = f["tr"], f["te"]
            X_tr, X_te, y_tr, y_te = ds.X[tr], ds.X[te], ds.y[tr], ds.y[te]
            sc = StandardScaler().fit(X_tr); Ztr, Zte = sc.transform(X_tr), sc.transform(X_te)
            for pl in range(cfg.placements(name)):
                rseed = derived_seed(cfg, ds_i, f["fold"], pl)
                for b in budgets:
                    pos = seed_indices(len(tr), spd, cfg.chunk_days(name), b, rseed)
                    dt = _fit_tree(X_tr[pos], y_tr[pos], cfg, rseed)
                    P = _leaf_proba(dt, X_tr, K)
                    yh = P.argmax(1); yh[pos] = y_tr[pos]
                    M = _sets_from_P(P, cfg.tau, cfg); M[pos, y_tr[pos]] = True
                    unl = np.ones(len(tr), bool); unl[pos] = False
                    single = (M.sum(1) == 1) | ~unl
                    base = {"dataset": name, "fold": f["fold"], "placement": pl, "budget": b,
                            "n_seed": len(pos), "n_train": len(tr), "n_test": len(te),
                            "hard_label_acc_unl": float((yh[unl] == y_tr[unl]).mean()),
                            "share_unl_singleton": float(single[unl].mean())}
                    for meth, idx, yy in [("Supervised LR (seed only)", pos, y_tr[pos]),
                                          ("LR, DT-CG hard labels", np.arange(len(tr)), yh),
                                          ("LR, DT-CG hard labels (singleton)", np.where(single)[0], yh[single])]:
                        clf = fit_lr(Ztr[idx], yy, cfg, seed=rseed, imbalance="none")
                        acc, f1 = _score(y_te, clf.predict(Zte))
                        rows.append({**base, "method": meth, "accuracy": acc, "macro_f1": f1})
            print(f"[hardlabel] {name} fold {f['fold']+1} done ({time.time()-t0:.0f}s)", flush=True)
    hl = pd.DataFrame(rows); hl.to_csv(os.path.join(out, "hardlabel_fold_metrics.csv"), index=False)


    main_dir = main_dir or cfg.out_dir
    fm_path = os.path.join(main_dir, "fold_metrics.csv")
    allm = hl
    if os.path.exists(fm_path):
        fm = pd.read_csv(fm_path)
        so = fm[(fm.cand_mode == "seed_only") & (fm.method == "SoDisam") & (fm.budget < 1.0)]
        allm = pd.concat([hl, so[["dataset", "fold", "placement", "budget", "method", "accuracy",
                                  "macro_f1", "n_train", "n_test"]]], ignore_index=True)
        chk = fm[(fm.method == "Supervised LR (seed only)") & (fm.budget < 1.0)]
        m = hl[hl.method == "Supervised LR (seed only)"].merge(
            chk, on=["dataset", "fold", "placement", "budget"], suffixes=("", "_main"))
        if len(m):
            print(f"[hardlabel] check vs main run: max |acc diff| seed-only LR = "
                  f"{(m.accuracy - m.accuracy_main).abs().max():.4f} (should be ~0)")
    else:
        print(f"[hardlabel] {fm_path} not found: SoDisam columns omitted")
    fold_mean = allm.groupby(["dataset", "budget", "method", "fold"])[["accuracy", "macro_f1"]].mean().reset_index()
    summ = fold_mean.groupby(["dataset", "budget", "method"])[["accuracy", "macro_f1"]].mean().reset_index()
    summ.to_csv(os.path.join(out, "hardlabel_summary_by_dataset.csv"), index=False)
    overall = summ.groupby(["budget", "method"])[["accuracy", "macro_f1"]].mean().unstack("method")
    overall.to_csv(os.path.join(out, "hardlabel_summary_mean.csv"))
    diag = hl.drop_duplicates(["dataset", "fold", "placement", "budget"]).groupby(
        ["dataset", "budget"])[["hard_label_acc_unl", "share_unl_singleton"]].mean()
    diag.to_csv(os.path.join(out, "hardlabel_diagnostics.csv"))


    pr = []
    piv = fold_mean.pivot_table(index=["dataset", "budget", "fold"], columns="method",
                                values=["accuracy", "macro_f1"])
    ntr = hl.groupby(["dataset", "fold"]).n_train.first(); nte = hl.groupby(["dataset", "fold"]).n_test.first()
    for (d, b), g in piv.groupby(level=[0, 1]):
        folds = g.index.get_level_values(2)
        n_tr = ntr.loc[d].loc[folds].mean(); n_te = nte.loc[d].loc[folds].mean()
        for a_, r_ in [("SoDisam", "LR, DT-CG hard labels"),
                       ("LR, DT-CG hard labels", "Supervised LR (seed only)")]:
            for met in ["accuracy", "macro_f1"]:
                if (met, a_) not in g.columns or (met, r_) not in g.columns:
                    continue
                diff = (g[(met, a_)] - g[(met, r_)]).dropna().values
                if len(diff) < 2: continue
                res = paired_noninferiority(diff, n_tr, n_te, 0.0, cfg.alpha)
                tq = stats.t.ppf(0.975, len(diff) - 1)
                pr.append({"dataset": d, "budget": b, "comparison": f"{a_} - {r_}", "metric": met,
                           "mean_diff": res["mean_diff"], "ci95_lo": res["mean_diff"] - tq * res["se"],
                           "ci95_hi": res["mean_diff"] + tq * res["se"], "d_z": res["cohens_d"],
                           "n_folds": res["K"]})
    pd.DataFrame(pr).to_csv(os.path.join(out, "hardlabel_paired.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n[hardlabel] mean over datasets (accuracy / macro-F1):"); print(overall.round(3))
    print(f"[hardlabel] done in {time.time()-t0:.0f}s -> {out}")


def parse_args():
    ap = argparse.ArgumentParser(description="Partial label learning with DT-CG candidate sets for building occupancy estimation.")
    ap.add_argument("mode", nargs="?", default=None,
                    choices=["all", "revision", "coverage", "decomp", "imbalance-sens", "tau-select-sens",
                             "replicate", "audit", "run", "tau-sweep", "report", "tables", "hardlabel"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--protocol", choices=["blocked", "shuffled"])
    ap.add_argument("--folds-floor", type=int); ap.add_argument("--folds-robod", type=int)
    ap.add_argument("--purge-hours-floor", type=float); ap.add_argument("--purge-hours-robod", type=float)
    ap.add_argument("--budgets", type=float, nargs="+")
    ap.add_argument("--placements", type=int)
    ap.add_argument("--master-seed", type=int)
    ap.add_argument("--tau", type=float, help="fixed tau (only with --tau-mode fixed)")
    ap.add_argument("--tau-mode", choices=["select", "fixed"])
    ap.add_argument("--tau-target", type=float, help="coverage target of the seed-only rule")
    ap.add_argument("--cand-modes", default=None,
                    help="comma list: seed_only,oracle_true,proxy_q90,proxy_q95 (any proxy_qNN)")
    ap.add_argument("--budget-methods", default=None, help="comma list, default SoDisam")
    ap.add_argument("--dt-class-weight", choices=["none", "balanced"])
    ap.add_argument("--min-set-size", type=int)
    ap.add_argument("--imbalance", choices=["smote", "none", "weight"])
    ap.add_argument("--smote-k", type=int); ap.add_argument("--smote-ratio", type=float)
    ap.add_argument("--proxy-abs", type=float); ap.add_argument("--proxy-rel", type=float)
    ap.add_argument("--lr-class-weight", choices=["none", "balanced"])
    ap.add_argument("--lag-mode", choices=["segment", "legacy"])
    ap.add_argument("--no-tau-pll", action="store_true", help="tau sweep without downstream SoDisam")
    ap.add_argument("--lr-check", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--main-dir", default=None, help="hardlabel: folder with the main run's fold_metrics.csv")
    a, _unknown = ap.parse_known_args()
    if a.mode is None:
        a.mode = IDE_MODE
        a.data_dir = a.data_dir or IDE_DATA_DIR
        a.out_dir = a.out_dir or IDE_OUT_DIR
        a.quick = IDE_QUICK
        a.lr_check = IDE_LR_CHECK
        a.datasets = a.datasets or IDE_DATASETS
        a.placements = a.placements or IDE_PLACEMENTS
        a.budgets = a.budgets or IDE_BUDGETS
        a.cand_modes = a.cand_modes or IDE_CAND_MODES
        a.imbalance = a.imbalance or IDE_IMBALANCE
        a.tau_mode = a.tau_mode or IDE_TAU_MODE
        a.tau = a.tau if a.tau is not None else IDE_TAU
        a.tau_target = a.tau_target if a.tau_target is not None else IDE_TAU_TARGET
    a.data_dir = a.data_dir or Config.data_dir
    a.out_dir = a.out_dir or Config.out_dir
    return a


def build_cfg(a) -> Config:
    c = Config(data_dir=a.data_dir, out_dir=a.out_dir, quick=a.quick)
    if a.datasets: c.datasets = a.datasets
    if a.protocol: c.protocol = a.protocol
    if a.folds_floor: c.n_folds_floor = a.folds_floor
    if a.folds_robod: c.n_folds_robod = a.folds_robod
    if a.purge_hours_floor is not None: c.purge_hours_floor = a.purge_hours_floor
    if a.purge_hours_robod is not None: c.purge_hours_robod = a.purge_hours_robod
    if a.budgets: c.budgets = tuple(a.budgets)
    if a.placements: c.n_placements = a.placements
    if a.master_seed is not None: c.master_seed = a.master_seed
    if a.tau is not None: c.tau = a.tau
    if a.tau_mode: c.tau_mode = a.tau_mode
    if a.tau_target is not None: c.tau_target_cov = a.tau_target
    if a.cand_modes: c.cand_modes = tuple(a.cand_modes.split(","))
    if a.budget_methods: c.budget_methods = tuple(a.budget_methods.split(","))
    if a.dt_class_weight: c.dt_class_weight = None if a.dt_class_weight == "none" else a.dt_class_weight
    if a.min_set_size: c.dt_min_set_size = a.min_set_size
    if a.imbalance: c.imbalance = a.imbalance
    if a.smote_k: c.smote_k = a.smote_k
    if a.smote_ratio: c.smote_ratio = a.smote_ratio
    if a.proxy_abs is not None: c.proxy_abs = a.proxy_abs
    if a.proxy_rel is not None: c.proxy_rel = a.proxy_rel
    if a.lr_class_weight: c.lr_class_weight = None if a.lr_class_weight == "none" else a.lr_class_weight
    if a.lag_mode: c.lag_mode = a.lag_mode
    if a.no_tau_pll: c.tau_run_pll = False
    if c.quick:
        c.tau_budgets = (0.10,); c.decomp_budgets = (0.10,)
    return c


if __name__ == "__main__":
    args = parse_args()
    cfg = build_cfg(args)
    if args.mode == "revision":
        run_coverage_pack(cfg)
        run_decomposition(cfg)
        run_imbalance_sensitivity(cfg)
        run_tau_select_sensitivity(cfg)
        run_replication(cfg)
    elif args.mode == "coverage":
        run_coverage_pack(cfg)
    elif args.mode == "decomp":
        run_decomposition(cfg)
    elif args.mode == "imbalance-sens":
        run_imbalance_sensitivity(cfg)
    elif args.mode == "tau-select-sens":
        run_tau_select_sensitivity(cfg)
    elif args.mode == "replicate":
        run_replication(cfg)
    elif args.mode == "hardlabel":
        run_hardlabel(cfg, main_dir=args.main_dir)
    elif args.mode == "tables":
        make_coverage_tables(cfg); make_figures(cfg)
    elif args.mode == "all":
        run_audit(cfg, args.lr_check)
        run_experiment(cfg)
        run_report(cfg)
    elif args.mode == "audit":
        run_audit(cfg, args.lr_check)
    elif args.mode == "run":
        run_experiment(cfg)
    elif args.mode == "tau-sweep":
        run_tau_sweep(cfg)
    elif args.mode == "report":
        run_report(cfg)

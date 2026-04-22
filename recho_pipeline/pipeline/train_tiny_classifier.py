"""
Tiny-feature / tiny-classifier baseline for sheep-vs-not-sheep.

Instead of feeding the full (200, 100) reservoir image into a CNN, extract a
handful of cheap scalar features from the (x(t), y(t)) trajectory of each
clip and classify with a lightweight sklearn model. The whole pipeline is
designed to fit on a microcontroller — feature extraction is O(N) over the
time series, and the classifier is either a linear model, a tiny tree, or a
diagonal-Gaussian naive Bayes.

Features (per clip, concatenated into one vector):
  - mean/variance of x and y
  - signal energy  sum x^2, sum y^2
  - radius    r(t) = sqrt(x^2 + y^2)    — mean, var, min, max
  - phase     theta(t) = atan2(y, x)    — wrapped mean/var, unwrapped drift rate
  - derivatives  dx, dy, dr, d(theta_unwrapped)  — mean |.| and var
  - zero-crossing counts for x and y
  - dominant oscillation frequency (rFFT peak) and its power
  - spectral centroid and spectral energy concentration of r(t)
  - autocorrelation of x and y at geometric lags (1, 2, 4, 8, 16, 32, 64)
  - cross-correlation x<->y at lags (1, 4, 16, 64)
  - log-power in 8 bands of the r(t) spectrum
  - 4x4 phase-space occupancy histogram (16 bins, normalised)

Classifier options (same feature vector, pick one at the CLI):
  - logistic regression
  - linear SVM (LinearSVC)
  - small decision tree  (max_depth=4)
  - tiny random forest   (n_estimators=20, max_depth=5)
  - gaussian naive bayes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


ESC50_HOPF_TEXT_CACHE: Path = Path(
    "/Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50/hopf_text"
)
TARGET_CLASS: str = "sheep"
# Positive fraction after balancing. 0.5 = 50/50 (drops negatives to match the
# ~40 sheep clips, giving ~80 total). None = keep every clip and rely on the
# sklearn classifier's class_weight="balanced" to compensate — ~2000 samples,
# tighter CV stds, but much more imbalanced confusion matrices.
TARGET_POSITIVE_RATE: float | None = 0.5
SUBTRACT_COMMON_MODE: bool = True
VAL_SPLIT: float = 0.2
# Classifier: one of {"logreg", "svm", "tree", "forest", "nb"}, or "all" to
# fit every option on the same train/val split and print a comparison table.
CLASSIFIER: str = "svm"
# When CLASSIFIER == "all", prefer K-fold cross-validation over a single split.
# With ~80 balanced sheep-vs-rest samples, a single 20% val set has only ~16
# rows and the per-model metrics are within noise of each other — CV gives
# every sample a turn in the held-out fold and reports mean ± std per model.
USE_CROSS_VAL: bool = True
N_FOLDS: int = 5
# When True, grid-search C for the selected CLASSIFIER (must be "logreg" or
# "svm"). Runs K-fold CV at every C value and reports the best by TUNE_METRIC.
TUNE_C: bool = False
TUNE_C_VALUES: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
TUNE_METRIC: str = "f1"  # one of: "f1", "auroc", "acc", "prec", "rec"
# When True, grid-search (C, gamma) for the RBF-kernel SVM. Overrides TUNE_C
# and CLASSIFIER. gamma="scale" uses sklearn's heuristic 1/(n_features * var).
TUNE_RBF: bool = True
TUNE_RBF_C_VALUES: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
TUNE_RBF_GAMMA_VALUES: tuple[str | float, ...] = ("scale", 0.001, 0.01, 0.1, 1.0)
# Occupancy histogram resolution (N_HIST x N_HIST bins in phase space).
N_HIST: int = 4
# Autocorrelation lags — geometric spacing covers fine (1-sample) to coarse
# (multi-cycle) structure cheaply. Applied to both x(t) and y(t).
AC_LAGS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
# Cross-correlation lags between x(t) and y(t). Catches phase-lag structure
# that pure-x and pure-y autocorrelation cannot — Hopf keeps y ~pi/2 out of
# phase with x at the natural orbit, but forced classes deviate differently.
XCORR_LAGS: tuple[int, ...] = (1, 4, 16, 64)
# Number of log-power bands to split the r(t) rFFT into. Each band
# contributes one feature: log(sum(|R(f)|^2) + eps) inside the band.
SPEC_BANDS: int = 8


def _autocorr(x: NDArray[np.float64], lag: int) -> float:
    if lag <= 0 or lag >= len(x):
        return 0.0
    a = x[:-lag] - x[:-lag].mean()
    b = x[lag:] - x[lag:].mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return 0.0
    return float((a * b).sum() / denom)


def _xcorr(x: NDArray[np.float64], y: NDArray[np.float64], lag: int) -> float:
    """
    Pearson cross-correlation between x and y with y shifted by `lag` samples
    (positive lag = y lags x). Returns 0 for out-of-range lags.
    """
    if abs(lag) >= len(x):
        return 0.0
    if lag >= 0:
        a = x[:len(x) - lag] - x[:len(x) - lag].mean()
        b = y[lag:] - y[lag:].mean()
    else:
        a = x[-lag:] - x[-lag:].mean()
        b = y[:len(y) + lag] - y[:len(y) + lag].mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return 0.0
    return float((a * b).sum() / denom)


def _log_spectral_bands(
    x: NDArray[np.float64], n_bands: int
) -> NDArray[np.float64]:
    """
    Partition the rFFT power spectrum of x into `n_bands` contiguous bins
    (skipping DC) and return the log10 of the summed power per band.

    Log transform compresses the wide dynamic range of audio-driven spectra
    so no single band dominates downstream standard-scaling.
    """
    r0 = x - x.mean()
    spec = np.abs(np.fft.rfft(r0))
    power = (spec * spec)[1:]  # drop DC
    if len(power) == 0 or n_bands <= 0:
        return np.zeros(n_bands, dtype=np.float64)
    edges = np.linspace(0, len(power), n_bands + 1, dtype=np.int64)
    out = np.empty(n_bands, dtype=np.float64)
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        band = power[lo:hi] if hi > lo else power[lo:lo + 1]
        out[i] = np.log10(float(band.sum()) + 1e-12)
    return out


def _zero_crossings(x: NDArray[np.float64]) -> float:
    s = np.signbit(x - x.mean())
    return float(np.count_nonzero(s[1:] != s[:-1]))


def _occupancy_hist(x: NDArray[np.float64], y: NDArray[np.float64], n_bins: int) -> NDArray[np.float64]:
    # Normalise per clip so the histogram describes shape, not amplitude.
    sx = x.std() + 1e-9
    sy = y.std() + 1e-9
    xn = np.clip((x - x.mean()) / sx, -3.0, 3.0)
    yn = np.clip((y - y.mean()) / sy, -3.0, 3.0)
    h, _, _ = np.histogram2d(
        xn, yn, bins=n_bins, range=[[-3.0, 3.0], [-3.0, 3.0]]
    )
    total = h.sum()
    return (h / total if total > 0 else h).ravel()


def extract_features_single(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the full feature vector for one clip. x and y are 1-D of equal
    length — the downsampled-and-flattened Hopf trajectory.
    """
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    theta_u = np.unwrap(theta)

    dx = np.diff(x)
    dy = np.diff(y)
    dr = np.diff(r)
    dth = np.diff(theta_u)

    feats: list[float] = []

    # Moments and energy of x, y.
    feats += [float(x.mean()), float(x.var()), float(y.mean()), float(y.var())]
    feats += [float((x * x).sum()), float((y * y).sum())]

    # Radius statistics.
    feats += [float(r.mean()), float(r.var()), float(r.min()), float(r.max())]

    # Phase statistics. Use circular mean/var for the wrapped angle so it isn't
    # meaningless on [-pi, pi], and the slope of the unwrapped phase as the
    # mean rotation rate (cycles/sample).
    feats += [
        float(np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())),
        float(1.0 - np.hypot(np.sin(theta).mean(), np.cos(theta).mean())),
        float((theta_u[-1] - theta_u[0]) / max(1, len(theta_u) - 1)),
    ]

    # Derivative magnitudes.
    for d in (dx, dy, dr, dth):
        feats += [float(np.mean(np.abs(d))), float(np.var(d))]

    # Zero-crossings.
    feats += [_zero_crossings(x), _zero_crossings(y)]

    # Dominant oscillation frequency of r(t): peak bin of rFFT magnitude and
    # its relative power. Normalised index so it's sample-rate agnostic.
    r0 = r - r.mean()
    spec = np.abs(np.fft.rfft(r0))
    total_power = float((spec * spec).sum()) + 1e-12
    if len(spec) > 1:
        peak = int(1 + np.argmax(spec[1:]))  # skip DC
        peak_freq_norm = peak / len(spec)
        peak_power_frac = float(spec[peak] ** 2) / total_power
        # Spectral centroid (weighted by power).
        idx = np.arange(len(spec))
        centroid = float((idx * spec * spec).sum() / total_power) / max(1, len(spec) - 1)
    else:
        peak_freq_norm = 0.0
        peak_power_frac = 0.0
        centroid = 0.0
    feats += [peak_freq_norm, peak_power_frac, centroid]

    # Autocorrelation of x and y over a geometric lag grid.
    for lag in AC_LAGS:
        feats.append(_autocorr(x, lag))
    for lag in AC_LAGS:
        feats.append(_autocorr(y, lag))

    # Cross-correlation x<->y at a few lags (captures phase-lag structure
    # that per-channel autocorrelation cannot see).
    for lag in XCORR_LAGS:
        feats.append(_xcorr(x, y, lag))

    # Log-power in contiguous bands of the r(t) spectrum — fuller spectral
    # fingerprint than just peak + centroid.
    feats += _log_spectral_bands(r, SPEC_BANDS).tolist()

    # Phase-space occupancy histogram.
    feats += _occupancy_hist(x, y, N_HIST).tolist()

    return np.asarray(feats, dtype=np.float64)


def extract_features_dataset(
    x_processed: NDArray[np.float64], y_processed: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Map a batch of (x, y) trajectories to a feature matrix of shape
    (n_clips, n_features). Input shape is (n_clips, n_time, n_nodes) — the
    last two axes are flattened into one long time series per clip.
    """
    n = x_processed.shape[0]
    x_flat = x_processed.reshape(n, -1)
    y_flat = y_processed.reshape(n, -1)
    feats = [extract_features_single(x_flat[i], y_flat[i]) for i in range(n)]
    return np.stack(feats, axis=0)


def balance_binary(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    target_positive_rate: float,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    if len(pos) == 0:
        raise ValueError("no positives — cannot balance")
    n_neg_target = int(round(len(pos) * (1.0 - target_positive_rate) / target_positive_rate))
    n_neg_keep = min(n_neg_target, len(neg))
    rng = np.random.default_rng(seed)
    neg_keep = rng.choice(neg, size=n_neg_keep, replace=False)
    keep = np.concatenate([pos, neg_keep])
    rng.shuffle(keep)
    print(
        f"[train] Balanced: kept {len(pos)} pos + {n_neg_keep}/{len(neg)} neg "
        f"(target={target_positive_rate:.2f}, actual="
        f"{len(pos) / (len(pos) + n_neg_keep):.2f})"
    )
    return features[keep], labels[keep]


def stratified_split(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    val_split: float,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    val_idx_parts, train_idx_parts = [], []
    for c in np.unique(labels):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        n_val = int(round(len(class_idx) * val_split))
        val_idx_parts.append(class_idx[:n_val])
        train_idx_parts.append(class_idx[n_val:])
    val_idx = np.concatenate(val_idx_parts)
    train_idx = np.concatenate(train_idx_parts)
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    print(
        f"[train] Stratified split — train: {len(train_idx)} "
        f"(pos={int((labels[train_idx] == 1).sum())}), "
        f"val: {len(val_idx)} (pos={int((labels[val_idx] == 1).sum())})"
    )
    return features[train_idx], labels[train_idx], features[val_idx], labels[val_idx]


CLASSIFIER_FACTORIES: dict[str, Callable[[], object]] = {
    # Linear models get standard-scaled features; trees/forests/NB do not need
    # scaling but it's harmless, so we wrap all of them in a pipeline with a
    # StandardScaler for consistency.
    "logreg": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ]),
    "svm": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf", LinearSVC(class_weight="balanced", C=1.0, dual=False, max_iter=10000)),
    ]),
    "tree": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=0)),
    ]),
    "forest": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=20, max_depth=5, class_weight="balanced", random_state=0, n_jobs=1,
        )),
    ]),
    "nb": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf", GaussianNB()),
    ]),
    "rbf": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", C=1.0, gamma="scale")),
    ]),
}


def _scores_for(clf, x_val: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Return continuous scores for ROC-AUC, or None if unavailable."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x_val)[:, 1]
    if hasattr(clf, "decision_function"):
        return clf.decision_function(x_val)
    return None


def _fit_and_score(
    classifier_key: str,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64],
    y_val: NDArray[np.int64],
) -> dict[str, float]:
    """Fit one classifier and return a small metrics dict (plus fit seconds)."""
    import time

    clf = CLASSIFIER_FACTORIES[classifier_key]()
    t0 = time.perf_counter()
    clf.fit(x_train, y_train)
    fit_s = time.perf_counter() - t0

    y_pred = clf.predict(x_val)
    acc = accuracy_score(y_val, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    scores = _scores_for(clf, x_val)
    auroc = (
        float(roc_auc_score(y_val, scores))
        if scores is not None and len(np.unique(y_val)) == 2
        else float("nan")
    )
    return {
        "acc": float(acc), "prec": float(p), "rec": float(r), "f1": float(f1),
        "auroc": auroc, "fit_s": float(fit_s),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def train_and_evaluate(
    classifier_key: str,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64],
    y_val: NDArray[np.int64],
) -> None:
    if classifier_key not in CLASSIFIER_FACTORIES:
        raise ValueError(
            f"unknown classifier {classifier_key!r}; choose one of "
            f"{sorted(CLASSIFIER_FACTORIES)}"
        )
    print(f"[train] Fitting classifier: {classifier_key}")
    m = _fit_and_score(classifier_key, x_train, y_train, x_val, y_val)
    auroc_str = f"{m['auroc']:.3f}" if not np.isnan(m["auroc"]) else "  n/a"
    print(f"\n--- Validation metrics ({classifier_key}) ---")
    print(f"  accuracy : {m['acc']:.3f}")
    print(f"  precision: {m['prec']:.3f}   recall: {m['rec']:.3f}   f1: {m['f1']:.3f}")
    print(f"  ROC AUC  : {auroc_str}")
    print(f"  fit time : {m['fit_s']:.2f}s")
    print(f"  confusion matrix (rows=true, cols=pred):")
    print(f"      pred=0   pred=1")
    print(f"true=0 {m['tn']:>6d} {m['fp']:>8d}")
    print(f"true=1 {m['fn']:>6d} {m['tp']:>8d}")


def _build_with_c(classifier_key: str, c: float) -> Pipeline:
    """Re-build a logreg or svm pipeline at a specific C value."""
    if classifier_key == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=c)),
        ])
    if classifier_key == "svm":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearSVC(class_weight="balanced", C=c, dual=False, max_iter=10000)),
        ])
    raise ValueError(
        f"C tuning only supports logreg / svm, got {classifier_key!r}"
    )


def tune_c(
    classifier_key: str,
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    c_values: tuple[float, ...] = TUNE_C_VALUES,
    n_folds: int = 5,
    metric: str = "f1",
    seed: int = 0,
) -> float:
    """
    K-fold CV grid-search over C for logreg / svm.

    Every C value is evaluated on the same K splits so comparisons are
    apples-to-apples. Prints mean ± std per metric per C, then returns the
    C with the best mean `metric`.
    """
    import time

    if classifier_key not in {"logreg", "svm"}:
        raise ValueError(
            f"C tuning only supports logreg / svm, got {classifier_key!r}"
        )
    if metric not in {"f1", "auroc", "acc", "prec", "rec"}:
        raise ValueError(f"metric must be f1/auroc/acc/prec/rec, got {metric!r}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    print(
        f"[tune] {classifier_key} C-grid {list(c_values)} with "
        f"{n_folds}-fold CV on {len(features)} samples "
        f"({int((labels == 1).sum())} pos, {int((labels == 0).sum())} neg) ..."
    )

    per_c: dict[float, dict[str, list[float]]] = {
        c: {m: [] for m in ("acc", "prec", "rec", "f1", "auroc", "fit_s")}
        for c in c_values
    }

    for tr_idx, va_idx in skf.split(features, labels):
        x_tr, x_va = features[tr_idx], features[va_idx]
        y_tr, y_va = labels[tr_idx], labels[va_idx]
        for c in c_values:
            clf = _build_with_c(classifier_key, c)
            t0 = time.perf_counter()
            clf.fit(x_tr, y_tr)
            fit_s = time.perf_counter() - t0
            y_pred = clf.predict(x_va)
            p, r, f1, _ = precision_recall_fscore_support(
                y_va, y_pred, average="binary", zero_division=0
            )
            scores = _scores_for(clf, x_va)
            auroc = (
                float(roc_auc_score(y_va, scores))
                if scores is not None and len(np.unique(y_va)) == 2
                else float("nan")
            )
            per_c[c]["acc"].append(float(accuracy_score(y_va, y_pred)))
            per_c[c]["prec"].append(float(p))
            per_c[c]["rec"].append(float(r))
            per_c[c]["f1"].append(float(f1))
            per_c[c]["auroc"].append(auroc)
            per_c[c]["fit_s"].append(fit_s)

    stats: dict[float, dict[str, tuple[float, float]]] = {}
    for c, m in per_c.items():
        s: dict[str, tuple[float, float]] = {}
        for name, vals in m.items():
            arr = np.asarray([v for v in vals if not np.isnan(v)], dtype=np.float64)
            s[name] = (
                (float(arr.mean()), float(arr.std()))
                if arr.size else (float("nan"), float("nan"))
            )
        stats[c] = s

    def _fmt(s: dict[str, tuple[float, float]], name: str) -> str:
        mean, std = s[name]
        return "         n/a" if np.isnan(mean) else f"{mean:.3f}±{std:.3f}"

    print(f"\n--- {classifier_key} C-grid results (sorted by C asc) ---")
    header = (f"{'C':>8}  {'acc':>13}  {'prec':>13}  {'rec':>13}  "
              f"{'f1':>13}  {'auroc':>13}  {'fit(s)':>8}")
    print(header)
    print("-" * len(header))
    for c in c_values:
        s = stats[c]
        print(
            f"{c:>8.3g}  {_fmt(s, 'acc'):>13}  {_fmt(s, 'prec'):>13}  "
            f"{_fmt(s, 'rec'):>13}  {_fmt(s, 'f1'):>13}  "
            f"{_fmt(s, 'auroc'):>13}  {s['fit_s'][0]:>8.3f}"
        )

    best_c = max(
        c_values,
        key=lambda c: (
            stats[c][metric][0] if not np.isnan(stats[c][metric][0]) else -np.inf
        ),
    )
    best_mean, best_std = stats[best_c][metric]
    print(
        f"\n[tune] Best C for {classifier_key} by {metric}: "
        f"C={best_c} ({metric}={best_mean:.3f} ± {best_std:.3f})"
    )
    return best_c


def tune_rbf_svm(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    c_values: tuple[float, ...] = TUNE_RBF_C_VALUES,
    gamma_values: tuple[str | float, ...] = TUNE_RBF_GAMMA_VALUES,
    n_folds: int = 5,
    metric: str = "f1",
    seed: int = 0,
) -> tuple[float, str | float]:
    """
    K-fold CV grid-search over (C, gamma) for an RBF-kernel SVM.

    Prints a 2-D grid of the selected `metric` (mean ± std) for every
    (C, gamma) combination, then returns the best pair.

    gamma accepts either a float or the string "scale" (sklearn default:
    1 / (n_features * X.var())), which is a sensible zero-cost baseline.
    """
    import time

    if metric not in {"f1", "auroc", "acc", "prec", "rec"}:
        raise ValueError(f"metric must be f1/auroc/acc/prec/rec, got {metric!r}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    print(
        f"[tune] RBF-SVM (C, gamma) grid — C={list(c_values)}, "
        f"gamma={list(gamma_values)} — {n_folds}-fold CV on {len(features)} "
        f"samples ({int((labels == 1).sum())} pos, "
        f"{int((labels == 0).sum())} neg) ..."
    )

    metric_grid: dict[tuple[float, object], list[float]] = {
        (c, g): [] for c in c_values for g in gamma_values
    }
    fit_grid: dict[tuple[float, object], list[float]] = {
        (c, g): [] for c in c_values for g in gamma_values
    }

    for tr_idx, va_idx in skf.split(features, labels):
        x_tr, x_va = features[tr_idx], features[va_idx]
        y_tr, y_va = labels[tr_idx], labels[va_idx]
        for c in c_values:
            for g in gamma_values:
                # gamma is typed as Literal["scale", "auto"] | float; our
                # config tuple legitimately holds either, so cast to Any.
                clf = Pipeline([
                    ("scale", StandardScaler()),
                    ("clf", SVC(
                        kernel="rbf", class_weight="balanced",
                        C=c, gamma=cast(Any, g),
                    )),
                ])
                t0 = time.perf_counter()
                clf.fit(x_tr, y_tr)
                fit_grid[(c, g)].append(time.perf_counter() - t0)

                y_pred = clf.predict(x_va)
                if metric == "auroc":
                    scores = _scores_for(clf, x_va)
                    val = (
                        float(roc_auc_score(y_va, scores))
                        if scores is not None and len(np.unique(y_va)) == 2
                        else float("nan")
                    )
                elif metric == "acc":
                    val = float(accuracy_score(y_va, y_pred))
                else:
                    p, r, f1, _ = precision_recall_fscore_support(
                        y_va, y_pred, average="binary", zero_division=0
                    )
                    val = {"prec": float(p), "rec": float(r), "f1": float(f1)}[metric]
                metric_grid[(c, g)].append(val)

    print(f"\n--- RBF-SVM {metric} grid (rows=C, cols=gamma; mean ± std) ---")
    gamma_labels = [str(g) for g in gamma_values]
    corner = "C / gamma"
    header = f"{corner:>10}  " + "  ".join(f"{lab:>13}" for lab in gamma_labels)
    print(header)
    print("-" * len(header))
    for c in c_values:
        cells: list[str] = []
        for g in gamma_values:
            vals = np.asarray(
                [v for v in metric_grid[(c, g)] if not np.isnan(v)],
                dtype=np.float64,
            )
            if vals.size == 0:
                cells.append("         n/a")
            else:
                cells.append(f"{vals.mean():.3f}±{vals.std():.3f}")
        print(f"{c:>10.3g}  " + "  ".join(f"{cell:>13}" for cell in cells))

    means: dict[tuple[float, object], float] = {}
    for key, vals in metric_grid.items():
        arr = np.asarray([v for v in vals if not np.isnan(v)], dtype=np.float64)
        means[key] = float(arr.mean()) if arr.size else -np.inf
    best_key = max(means, key=means.get)  # type: ignore[arg-type]
    best_c, best_g = best_key
    best_vals = np.asarray(
        [v for v in metric_grid[best_key] if not np.isnan(v)], dtype=np.float64,
    )
    print(
        f"\n[tune] Best (C, gamma) for RBF-SVM by {metric}: "
        f"C={best_c}, gamma={best_g} "
        f"({metric}={best_vals.mean():.3f} ± {best_vals.std():.3f}, "
        f"mean fit={np.mean(fit_grid[best_key]):.3f}s)"
    )
    return best_c, best_g


def compare_all_cv(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    n_folds: int = 5,
    sort_by: str = "f1",
    seed: int = 0,
) -> None:
    """
    Fit every classifier under stratified K-fold cross-validation and print a
    comparison table with mean ± std for each metric.

    Every sample serves in exactly one held-out fold, so the reported metrics
    are computed across all N samples rather than a single ~16-row validation
    set. With small datasets (e.g. 80 balanced clips) this is the only way to
    tell model differences apart from sampling noise.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    print(f"[train] {n_folds}-fold CV on {len(features)} samples "
          f"({int((labels == 1).sum())} pos, {int((labels == 0).sum())} neg) ...")

    agg: dict[str, dict[str, list[float]]] = {
        k: {m: [] for m in ("acc", "prec", "rec", "f1", "auroc", "fit_s")}
        for k in CLASSIFIER_FACTORIES
    }
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(features, labels)):
        x_tr, x_va = features[tr_idx], features[va_idx]
        y_tr, y_va = labels[tr_idx], labels[va_idx]
        for key in CLASSIFIER_FACTORIES:
            try:
                m = _fit_and_score(key, x_tr, y_tr, x_va, y_va)
            except Exception as e:
                print(f"  fold {fold_i} {key} failed: {type(e).__name__}: {e}")
                continue
            for metric_name in agg[key]:
                agg[key][metric_name].append(m[metric_name])

    # Reduce to mean/std per model.
    rows: list[tuple[str, dict[str, tuple[float, float]]]] = []
    for key, metrics in agg.items():
        stats: dict[str, tuple[float, float]] = {}
        for metric_name, vals in metrics.items():
            arr = np.asarray([v for v in vals if not np.isnan(v)], dtype=np.float64)
            if arr.size == 0:
                stats[metric_name] = (float("nan"), float("nan"))
            else:
                stats[metric_name] = (float(arr.mean()), float(arr.std()))
        rows.append((key, stats))

    if sort_by not in {"f1", "auroc", "acc", "prec", "rec"}:
        sort_by = "f1"
    rows.sort(
        key=lambda kv: (kv[1][sort_by][0] if not np.isnan(kv[1][sort_by][0]) else -np.inf),
        reverse=True,
    )

    print(f"\n--- {n_folds}-fold CV results (mean ± std, sorted by {sort_by} desc) ---")
    header = (f"{'model':<8}  {'acc':>13}  {'prec':>13}  {'rec':>13}  "
              f"{'f1':>13}  {'auroc':>13}  {'fit(s)':>8}")
    print(header)
    print("-" * len(header))
    for key, s in rows:
        def _fmt(name: str) -> str:
            mean, std = s[name]
            if np.isnan(mean):
                return "         n/a"
            return f"{mean:.3f}±{std:.3f}"
        fit_mean = s["fit_s"][0]
        print(
            f"{key:<8}  {_fmt('acc'):>13}  {_fmt('prec'):>13}  {_fmt('rec'):>13}  "
            f"{_fmt('f1'):>13}  {_fmt('auroc'):>13}  {fit_mean:>8.3f}"
        )

    if rows:
        best_key, best_s = rows[0]
        mean, std = best_s[sort_by]
        print(f"\n[train] Best by {sort_by}: {best_key} "
              f"({sort_by}={mean:.3f} ± {std:.3f})")


def compare_all(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64],
    y_val: NDArray[np.int64],
    sort_by: str = "f1",
) -> None:
    """
    Fit every classifier in CLASSIFIER_FACTORIES on the same split and print
    a one-row-per-model comparison table, sorted by `sort_by` descending.

    sort_by: column to rank rows by — "f1", "auroc", "acc", "prec", or "rec".
    """
    print(f"[train] Comparing {len(CLASSIFIER_FACTORIES)} classifiers on the "
          f"same train/val split ...")
    rows: list[tuple[str, dict[str, float]]] = []
    for key in CLASSIFIER_FACTORIES:
        print(f"  fitting {key} ...")
        try:
            rows.append((key, _fit_and_score(key, x_train, y_train, x_val, y_val)))
        except Exception as e:
            print(f"    {key} failed: {type(e).__name__}: {e}")

    if sort_by not in {"f1", "auroc", "acc", "prec", "rec"}:
        sort_by = "f1"
    # NaN sorts last on descending order.
    rows.sort(
        key=lambda kv: (kv[1][sort_by] if not np.isnan(kv[1][sort_by]) else -np.inf),
        reverse=True,
    )

    print("\n--- Model comparison (sorted by " + sort_by + " desc) ---")
    header = f"{'model':<8}  {'acc':>5}  {'prec':>5}  {'rec':>5}  {'f1':>5}  {'auroc':>5}  {'fit(s)':>6}  {'tp/fp/fn/tn':>15}"
    print(header)
    print("-" * len(header))
    for key, m in rows:
        auroc_str = f"{m['auroc']:.3f}" if not np.isnan(m["auroc"]) else "  n/a"
        cm_str = f"{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}"
        print(
            f"{key:<8}  {m['acc']:.3f}  {m['prec']:.3f}  {m['rec']:.3f}  "
            f"{m['f1']:.3f}  {auroc_str:>5}  {m['fit_s']:>6.2f}  {cm_str:>15}"
        )

    if rows:
        best_key, best_m = rows[0]
        print(f"\n[train] Best by {sort_by}: {best_key} ({sort_by}={best_m[sort_by]:.3f})")


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.sample_data import load_xy_dataset_from_text_cache
    from pipeline.ingest import process_dataset, FS_HW, FS_TARGET

    print(
        f"[train] Loading hopf_text cache from {ESC50_HOPF_TEXT_CACHE} "
        f"(binary: {TARGET_CLASS} vs rest) ..."
    )
    raw_x, raw_y, labels, class_names, fs = load_xy_dataset_from_text_cache(
        cache_dir=ESC50_HOPF_TEXT_CACHE,
        target_class=TARGET_CLASS,
    )
    print(f"  raw_x: {raw_x.shape}, fs={fs} Hz, n_classes={len(class_names)}")

    ds_factor = 1 if fs == FS_TARGET else FS_HW // fs
    print(
        f"[train] Processing clips (downsample_factor={ds_factor}, "
        f"subtract_common_mode={SUBTRACT_COMMON_MODE}) ..."
    )
    x_processed = process_dataset(
        raw_x, downsample_factor=ds_factor, subtract_common_mode=SUBTRACT_COMMON_MODE,
    )
    y_processed = process_dataset(
        raw_y, downsample_factor=ds_factor, subtract_common_mode=SUBTRACT_COMMON_MODE,
    )

    print(f"[train] Extracting trajectory features ...")
    features = extract_features_dataset(x_processed, y_processed)
    print(f"  features: shape={features.shape} (clips x feature_dim)")

    if TARGET_POSITIVE_RATE is not None:
        features, labels = balance_binary(features, labels, TARGET_POSITIVE_RATE)

    if TUNE_RBF:
        tune_rbf_svm(
            features, labels,
            c_values=TUNE_RBF_C_VALUES,
            gamma_values=TUNE_RBF_GAMMA_VALUES,
            n_folds=N_FOLDS,
            metric=TUNE_METRIC,
        )
        return

    if TUNE_C:
        tune_c(
            CLASSIFIER, features, labels,
            c_values=TUNE_C_VALUES, n_folds=N_FOLDS, metric=TUNE_METRIC,
        )
        return

    if CLASSIFIER == "all" and USE_CROSS_VAL:
        compare_all_cv(features, labels, n_folds=N_FOLDS)
        return

    x_train, y_train, x_val, y_val = stratified_split(features, labels, VAL_SPLIT)
    if CLASSIFIER == "all":
        compare_all(x_train, y_train, x_val, y_val)
    else:
        train_and_evaluate(CLASSIFIER, x_train, y_train, x_val, y_val)


if __name__ == "__main__":
    main()

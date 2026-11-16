"""Signal processing for the Spatial Analysis module.

All functions work on plain NumPy arrays shaped (trials, channels, times) so
they are easy to test and to reuse outside Streamlit. MNE is used where it
adds real value (surface Laplacian / CSD, CSP).
"""

from __future__ import annotations

import warnings

import mne
import numpy as np
from scipy.stats import binom
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

REFERENCES = [
    "As recorded",
    "Common average (CAR)",
    "Single electrode",
    "Hjorth Laplacian",
    "Surface Laplacian (CSD)",
]

BAND_PRESETS = {
    "Mu (8-13 Hz)": (8.0, 13.0),
    "Beta (13-30 Hz)": (13.0, 30.0),
    "Mu + beta (8-30 Hz)": (8.0, 30.0),
    "Alpha (8-12 Hz)": (8.0, 12.0),
    "Theta (4-8 Hz)": (4.0, 8.0),
    "Custom": None,
}


# --------------------------------------------------------------------------
# Re-referencing and spatial filters
# --------------------------------------------------------------------------

def channel_positions(info: mne.Info) -> np.ndarray:
    """3-D positions (metres) of each channel, shape (n_channels, 3)."""
    return np.array([ch["loc"][:3] for ch in info["chs"]])


def neighbours(positions: np.ndarray, k: int = 4) -> np.ndarray:
    """Indices of the k nearest electrodes for each electrode."""
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def common_average(X: np.ndarray) -> np.ndarray:
    """Subtract the mean over channels at every time point."""
    return X - X.mean(axis=-2, keepdims=True)


def single_reference(X: np.ndarray, ref_idx: int) -> np.ndarray:
    """Re-reference every channel to one electrode (it becomes flat zero)."""
    return X - X[..., ref_idx:ref_idx + 1, :]


def hjorth_laplacian(X: np.ndarray, positions: np.ndarray, k: int = 4) -> np.ndarray:
    """Hjorth (1975) small Laplacian: each channel minus the mean of its k
    nearest neighbours. Acts as a spatial high-pass filter."""
    nb = neighbours(positions, k)
    return X - X[..., nb, :].mean(axis=-2)


def surface_laplacian(X: np.ndarray, info: mne.Info, lambda2: float = 1e-5,
                      stiffness: int = 4) -> np.ndarray:
    """Spherical-spline surface Laplacian (current source density), via MNE.
    Output units are V/m^2, not volts."""
    epochs = mne.EpochsArray(X, info, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csd = mne.preprocessing.compute_current_source_density(
            epochs, lambda2=lambda2, stiffness=stiffness, verbose=False)
    return csd.get_data(copy=True)


def apply_reference(X: np.ndarray, info: mne.Info, method: str,
                    ref_channel: str | None = None, n_neighbours: int = 4) -> np.ndarray:
    """Dispatch to one of the reference schemes listed in ``REFERENCES``."""
    if method == "As recorded":
        return X.copy()
    if method == "Common average (CAR)":
        return common_average(X)
    if method == "Single electrode":
        names = info["ch_names"]
        if ref_channel not in names:
            raise ValueError(f"Reference channel {ref_channel!r} not in data")
        return single_reference(X, names.index(ref_channel))
    if method == "Hjorth Laplacian":
        return hjorth_laplacian(X, channel_positions(info), n_neighbours)
    if method == "Surface Laplacian (CSD)":
        return surface_laplacian(X, info)
    raise ValueError(f"Unknown reference method: {method}")


# --------------------------------------------------------------------------
# Temporal steps
# --------------------------------------------------------------------------

def bandpass(X: np.ndarray, sfreq: float, l_freq: float, h_freq: float,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    h_freq = min(h_freq, sfreq / 2 - 1)
    return mne.filter.filter_data(
        X.astype(np.float64), sfreq, l_freq, h_freq, method="iir",
        iir_params=dict(order=order, ftype="butter"), verbose=False)


def crop(X: np.ndarray, times: np.ndarray, tmin: float, tmax: float):
    mask = (times >= tmin) & (times <= tmax)
    if mask.sum() < 2:
        raise ValueError("Time window is empty, widen it.")
    return X[..., mask], times[mask]


def band_power(X: np.ndarray) -> np.ndarray:
    """Mean power per trial and channel (variance of a band-passed signal)."""
    return np.mean(X ** 2, axis=-1)


def envelope_power(X: np.ndarray, sfreq: float, smooth_s: float = 0.25) -> np.ndarray:
    """Instantaneous power smoothed with a moving average (for ERD time courses)."""
    win = max(1, int(smooth_s * sfreq))
    kernel = np.ones(win) / win
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), -1, X ** 2)


# --------------------------------------------------------------------------
# Topomap metrics
# --------------------------------------------------------------------------

def topo_metric(X_band: np.ndarray, y: np.ndarray, times: np.ndarray,
                window: tuple[float, float], metric: str,
                baseline: tuple[float, float] = (-1.0, 0.0)):
    """Return (values per channel, unit label) for the chosen metric.

    metric is one of: "difference", "power_a", "power_b", "erd_a", "erd_b".
    """
    Xw, _ = crop(X_band, times, *window)
    p = band_power(Xw)
    pa, pb = p[y == 0].mean(axis=0), p[y == 1].mean(axis=0)
    if metric == "difference":
        return 10 * np.log10(np.maximum(pa, 1e-30) / np.maximum(pb, 1e-30)), "dB (A / B)"
    if metric == "power_a":
        return 10 * np.log10(np.maximum(pa, 1e-30)), "dB"
    if metric == "power_b":
        return 10 * np.log10(np.maximum(pb, 1e-30)), "dB"
    if metric in ("erd_a", "erd_b"):
        if times[0] > baseline[0] or times[-1] < baseline[1]:
            raise ValueError("Epochs do not include the baseline window (-1 to 0 s).")
        Xb, _ = crop(X_band, times, *baseline)
        cls = 0 if metric == "erd_a" else 1
        task = band_power(Xw[y == cls]).mean(axis=0)
        base = band_power(Xb[y == cls]).mean(axis=0)
        return 100 * (task - base) / np.maximum(base, 1e-30), "% change vs baseline"
    raise ValueError(f"Unknown metric {metric}")


# --------------------------------------------------------------------------
# CSP
# --------------------------------------------------------------------------

def _rescale(X: np.ndarray) -> np.ndarray:
    """CSP is scale-invariant, but its covariance maths is not happy with
    values around 1e-12 (volts squared) or 1e-3 (CSD units). Bring data to
    unit scale first."""
    s = X.std()
    return X / s if s > 0 else X


def _make_csp(n_components: int, reg):
    from mne.decoding import CSP
    # Rank-deficient data (after CAR, Laplacians or CSD) can make the
    # generalised eigen-problem fail. A tiny diagonal loading keeps it solvable
    # without changing the result in any visible way.
    if reg is None:
        reg = 1e-6
    return CSP(n_components=n_components, reg=reg, log=True,
               norm_trace=False, component_order="alternate")


def drop_flat_channels(X: np.ndarray, names: list[str]):
    """Remove channels that are all zeros (e.g. the single-electrode reference)."""
    keep = np.where(X.std(axis=(0, 2)) > 0)[0]
    return X[:, keep, :], [names[i] for i in keep], keep


def fit_csp(X: np.ndarray, y: np.ndarray, n_components: int = 4, reg=None):
    """Fit CSP on all trials (for visualising patterns, not for scoring)."""
    csp = _make_csp(n_components, reg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = csp.fit_transform(_rescale(X), y)
    return csp, feats


def csp_cross_val(X: np.ndarray, y: np.ndarray, n_components: int = 4, reg=None,
                  n_splits: int = 5, seed: int = 42):
    """Leak-free cross-validated accuracy of CSP + LDA.

    CSP sits inside the pipeline, so in every fold it is fitted on the training
    trials only. Fitting CSP on all trials first and then cross-validating the
    classifier would leak test information and inflate the score.
    """
    n_splits = int(min(n_splits, np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError("Need at least 2 trials per class for cross-validation.")
    pipe = Pipeline([("csp", _make_csp(n_components, reg)),
                     ("lda", LinearDiscriminantAnalysis())])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(pipe, _rescale(X), y, cv=cv)
    return scores


def channel_power_cross_val(X: np.ndarray, y: np.ndarray, names: list[str],
                            channels=("C3", "C4"), n_splits: int = 5, seed: int = 42):
    """Baseline without CSP: log band power of a few named channels + LDA.

    Unlike CSP, this feature depends directly on the reference, so it shows
    why re-referencing matters for simple BCIs."""
    idx = [names.index(c) for c in channels if c in names]
    if not idx:
        raise ValueError("None of the baseline channels are present.")
    feats = np.log(np.maximum(band_power(X[:, idx, :]), 1e-30))
    if np.any(feats.std(axis=0) == 0):
        return np.full(2, 0.5)  # e.g. C3 itself is the reference
    n_splits = int(min(n_splits, np.bincount(y).min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_score(LinearDiscriminantAnalysis(), feats, y, cv=cv)


def chance_level(n_trials: int, n_classes: int = 2, alpha: float = 0.05) -> float:
    """Accuracy needed to beat chance with p < alpha for a finite number of
    trials (binomial test, Combrisson & Jerbi 2015)."""
    return float(binom.ppf(1 - alpha, n_trials, 1.0 / n_classes) / n_trials)


def parse_reg(reg_choice: str, shrinkage: float):
    if reg_choice == "None":
        return None
    if reg_choice == "Shrinkage (manual)":
        return float(shrinkage)
    return reg_choice  # 'ledoit_wolf' or 'oas'


# --------------------------------------------------------------------------
# Whole pipeline (used by the exported script and the tests)
# --------------------------------------------------------------------------

def run_pipeline(epochs: mne.Epochs, params: dict) -> dict:
    """Run every step for one parameter set and return the key results."""
    from .data import epochs_to_arrays

    X, y, times, names = epochs_to_arrays(epochs)
    Xr = apply_reference(X, epochs.info, params["reference"],
                         params.get("ref_channel"), params.get("n_neighbours", 4))
    Xf = bandpass(Xr, epochs.info["sfreq"], params["l_freq"], params["h_freq"])
    topo, unit = topo_metric(Xf, y, times, (params["tmin"], params["tmax"]),
                             params.get("metric", "difference"))
    picks = [i for i, ch in enumerate(epochs.ch_names)
             if ch in params.get("csp_channels", epochs.ch_names)]
    Xw, _ = crop(Xf[:, picks, :], times, params["tmin"], params["tmax"])
    Xw, kept, _ = drop_flat_channels(Xw, [epochs.ch_names[i] for i in picks])
    reg = parse_reg(params.get("csp_reg", "None"), params.get("shrinkage", 0.1))
    scores = csp_cross_val(Xw, y, params.get("n_components", 4), reg)
    csp, _ = fit_csp(Xw, y, params.get("n_components", 4), reg)
    return dict(topo=topo, unit=unit, scores=scores, csp=csp, csp_channels=kept,
                class_names=names, chance=chance_level(len(y)))

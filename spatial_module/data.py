"""Data sources for the Spatial Analysis module.

Every source ends up as an ``mne.Epochs`` object with:

* only EEG channels that have a known scalp position (needed for topomaps,
  Laplacians and CSD),
* exactly two classes (two event types), because the module compares two
  conditions such as left-hand vs right-hand motor imagery.

Three sources are supported:

1. ``make_synthetic_raw`` - a built-in simulated motor-imagery recording. It
   needs no download and we know the ground truth, which makes it ideal for
   teaching.
2. ``load_eegbci`` - the PhysioNet EEG Motor Movement/Imagery dataset
   (downloaded by MNE on first use, needs internet once).
3. ``load_uploaded`` - any file MNE can read (.fif, .edf, .bdf, .gdf, .set).
"""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

mne.set_log_level("ERROR")

# 36 electrodes of the 10-20 / 10-10 system, dense around the motor strip so
# that the Hjorth Laplacian around C3/C4 has proper neighbours.
SYNTHETIC_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "CP5", "CP3", "CP1", "CP2", "CP4", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]

SENSORIMOTOR_CHANNELS = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
]

CLASS_NAMES = ("left_hand", "right_hand")


def standard_montage() -> mne.channels.DigMontage:
    """Return the standard 10-20 montage under whichever name this MNE uses."""
    names = mne.channels.get_builtin_montages()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "colin27_1005" in names:  # MNE >= 1.13 renamed standard_1005
            return mne.channels.make_standard_montage("colin27_1005")
        return mne.channels.make_standard_montage("standard_1005")


@dataclass
class SyntheticParams:
    n_trials: int = 80
    erd_strength: float = 0.35  # 0 = no ERD, 1 = mu rhythm fully suppressed
    common_noise_uv: float = 20.0  # noise shared by every channel (bad reference)
    occipital_alpha_uv: float = 12.0  # strong, task-unrelated alpha at the back
    seed: int = 42


def _pink_noise(rng: np.random.Generator, n_signals: int, n_times: int) -> np.ndarray:
    """1/f noise with unit standard deviation per row."""
    white = rng.standard_normal((n_signals, n_times))
    spec = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(n_times)
    freqs[0] = freqs[1]
    spec /= np.sqrt(freqs)
    pink = np.fft.irfft(spec, n=n_times, axis=-1)
    pink -= pink.mean(axis=-1, keepdims=True)
    pink /= pink.std(axis=-1, keepdims=True)
    return pink


def _band_noise(rng, n_times, sfreq, lo, hi):
    sos = butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal(n_times))
    return x / x.std()


def _smooth_gate(t, start, stop, ramp=0.3):
    """1 inside [start, stop], 0 outside, with cosine ramps."""
    g = np.zeros_like(t)
    up = (t >= start - ramp) & (t < start)
    g[up] = 0.5 - 0.5 * np.cos(np.pi * (t[up] - (start - ramp)) / ramp)
    g[(t >= start) & (t <= stop)] = 1.0
    down = (t > stop) & (t <= stop + ramp)
    g[down] = 0.5 + 0.5 * np.cos(np.pi * (t[down] - stop) / ramp)
    return g


def make_synthetic_raw(params: SyntheticParams | None = None, sfreq: float = 160.0) -> mne.io.RawArray:
    """Simulate a continuous two-class motor-imagery EEG recording.

    Ground truth built into the simulation:

    * Two mu-rhythm (10-12 Hz, plus weaker 20 Hz beta) sources sit under C3
      (left motor cortex) and C4 (right motor cortex).
    * Imagining the RIGHT hand suppresses the LEFT (C3) source between 0.5 s
      and 3 s after the cue, and vice versa. This is event-related
      desynchronisation (ERD).
    * A strong occipital alpha source (around Oz) is unrelated to the task but
      spreads widely across the scalp (volume conduction).
    * A noisy reference adds the same signal to every channel. Re-referencing
      to the common average removes it exactly.
    """
    p = params or SyntheticParams()
    rng = np.random.default_rng(p.seed)
    montage = standard_montage()
    pos = montage.get_positions()["ch_pos"]
    xyz = np.array([pos[ch] for ch in SYNTHETIC_CHANNELS])
    n_ch = len(SYNTHETIC_CHANNELS)

    trial_len, iti, lead = 6.0, 1.0, 2.0  # seconds; cue at 1 s into each trial block
    n_times = int((lead + p.n_trials * (trial_len + iti) + lead) * sfreq)
    t = np.arange(n_times) / sfreq

    labels = np.array([0, 1] * (p.n_trials // 2) + [0] * (p.n_trials % 2))
    rng.shuffle(labels)
    cues = lead + 1.0 + np.arange(p.n_trials) * (trial_len + iti)

    # Amplitude envelopes of the two motor sources (1 = rest, lower = ERD).
    env_left = np.ones(n_times)   # source under C3
    env_right = np.ones(n_times)  # source under C4
    for cue, lab in zip(cues, labels):
        gate = _smooth_gate(t, cue + 0.5, cue + 3.0)
        if lab == 1:  # right-hand imagery -> left hemisphere ERD
            env_left *= 1 - p.erd_strength * gate
        else:         # left-hand imagery -> right hemisphere ERD
            env_right *= 1 - p.erd_strength * gate

    def motor_source():
        mu = _band_noise(rng, n_times, sfreq, 9.5, 12.5)
        beta = _band_noise(rng, n_times, sfreq, 18.0, 24.0)
        return mu + 0.4 * beta

    src_left = 7e-6 * env_left * motor_source()
    src_right = 7e-6 * env_right * motor_source()
    slow_mod = 0.6 + 0.4 * _band_noise(rng, n_times, sfreq, 0.05, 0.3)
    src_alpha = p.occipital_alpha_uv * 1e-6 * slow_mod * _band_noise(rng, n_times, sfreq, 9.0, 11.0)

    def gain(center, sigma):
        d2 = np.sum((xyz - center) ** 2, axis=1)
        return np.exp(-d2 / (2 * sigma ** 2))

    g_left = gain(pos["C3"], 0.025)
    g_right = gain(pos["C4"], 0.025)
    g_alpha = gain(pos["Oz"], 0.06)

    # Background activity, blurred across the scalp to mimic volume conduction.
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    mix = np.exp(-dist / 0.03)
    mix /= np.linalg.norm(mix, axis=1, keepdims=True)
    background = 5e-6 * mix @ _pink_noise(rng, n_ch, n_times)

    # Broadband (1-40 Hz) noise picked up by a bad reference electrode. Because
    # it is added to every channel it also lands inside the mu band.
    common = p.common_noise_uv * 1e-6 * _band_noise(rng, n_times, sfreq, 1.0, 40.0)[None, :]

    data = (
        background
        + np.outer(g_left, src_left)
        + np.outer(g_right, src_right)
        + np.outer(g_alpha, src_alpha)
        + common
    )

    info = mne.create_info(SYNTHETIC_CHANNELS, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(montage, on_missing="ignore")
    raw.set_annotations(
        mne.Annotations(onset=cues, duration=np.full(p.n_trials, 4.0),
                        description=[CLASS_NAMES[i] for i in labels])
    )
    return raw


def load_eegbci(subject: int = 1, runs=(4, 8, 12)) -> mne.io.BaseRaw:
    """PhysioNet EEG Motor Movement/Imagery: runs 4, 8, 12 are left vs right
    hand motor imagery (T1 = left fist, T2 = right fist). Needs internet the
    first time; files are cached in ~/mne_data afterwards."""
    from mne.datasets import eegbci

    files = eegbci.load_data(subject, list(runs), update_path=True, verbose=False)
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in files])
    eegbci.standardize(raw)
    raw.set_montage(standard_montage(), on_missing="ignore")
    raw.annotations.rename({"T1": "left_hand", "T2": "right_hand"})
    return raw


def load_uploaded(file_bytes: bytes, filename: str):
    """Read an uploaded file. Returns ``mne.io.Raw`` or ``mne.Epochs``."""
    suffix = "".join(s for s in [os.path.splitext(filename)[1]])
    if filename.endswith("-epo.fif") or filename.endswith("_epo.fif"):
        suffix = "-epo.fif"
    elif suffix == ".fif":
        suffix = "_raw.fif"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload" + suffix)
        with open(path, "wb") as fh:
            fh.write(file_bytes)
        if suffix == "-epo.fif":
            return mne.read_epochs(path, preload=True, verbose=False)
        return mne.io.read_raw(path, preload=True, verbose=False)



def ensure_positions(inst):
    """Keep EEG channels only and make sure each has a scalp position."""
    inst = inst.copy()

    # Keep EEG channels only
    eeg_like = [
        ch for ch, kind in zip(inst.ch_names, inst.get_channel_types())
        if kind == "eeg"
    ]

    if not eeg_like:
        raise ValueError("No EEG channels found in this file.")

    inst.pick(eeg_like)

    # Normalise channel names before applying the montage
    montage = standard_montage()
    montage_names = {
        name.lower(): name for name in montage.ch_names
    }

    rename_mapping = {}

    for old_name in inst.ch_names:
        clean_name = old_name.strip().rstrip(".").strip()
        standard_name = montage_names.get(clean_name.lower())

        if standard_name:
            rename_mapping[old_name] = standard_name

    if rename_mapping:
        inst.rename_channels(rename_mapping)

    # Apply standard electrode positions
    inst.set_montage(
        montage,
        match_case=False,
        on_missing="ignore"
    )

    # Retain channels with valid electrode positions
    good = [
        ch["ch_name"]
        for ch in inst.info["chs"]
        if (
            np.all(np.isfinite(ch["loc"][:3]))
            and np.any(ch["loc"][:3] != 0)
        )
    ]

    if len(good) < 8:
        raise ValueError(
            f"Fewer than 8 EEG channels have known 10-20 positions. "
            f"Only {len(good)} channels were recognised."
        )

    inst.pick(good)
    return inst



def event_labels(raw: mne.io.BaseRaw) -> dict[str, int]:
    """Event names and their counts, from annotations or a stim channel."""
    if len(raw.annotations):
        names, counts = np.unique(raw.annotations.description, return_counts=True)
        return {str(n): int(c) for n, c in zip(names, counts)}
    stim = mne.pick_types(raw.info, stim=True)
    if len(stim):
        ev = mne.find_events(raw, verbose=False)
        ids, counts = np.unique(ev[:, 2], return_counts=True)
        return {str(i): int(c) for i, c in zip(ids, counts)}
    return {}


def raw_to_epochs(raw: mne.io.BaseRaw, class_a: str, class_b: str,
                  tmin: float = -1.0, tmax: float = 4.0,
                  max_sfreq: float = 250.0) -> mne.Epochs:
    """Cut a continuous recording into epochs of two classes."""
    if len(raw.annotations):
        events, event_id = mne.events_from_annotations(raw, verbose=False)
    else:
        events = mne.find_events(raw, verbose=False)
        event_id = {str(i): int(i) for i in np.unique(events[:, 2])}
    keep = {class_a: event_id[class_a], class_b: event_id[class_b]}
    raw = ensure_positions(raw)
    epochs = mne.Epochs(raw, events, keep, tmin=tmin, tmax=tmax, baseline=None,
                        preload=True, verbose=False, event_repeated="drop")
    if epochs.info["sfreq"] > max_sfreq:
        epochs.resample(max_sfreq, verbose=False)
    return epochs


def prepare_epochs(epochs: mne.Epochs, class_a: str | None = None,
                   class_b: str | None = None) -> mne.Epochs:
    """Use already-epoched data (e.g. an uploaded -epo.fif) with two classes."""
    names = list(epochs.event_id)
    class_a = class_a or names[0]
    class_b = class_b or names[1]
    epochs = epochs[[class_a, class_b]]
    return ensure_positions(epochs)


def epochs_to_arrays(epochs: mne.Epochs):
    """Return (X, y, times, class_names). X is trials x channels x times (volts)."""
    names = list(epochs.event_id)
    inv = {v: k for k, v in epochs.event_id.items()}
    y = np.array([names.index(inv[c]) for c in epochs.events[:, 2]])
    return epochs.get_data(copy=True), y, epochs.times.copy(), names


def save_sample_file(path: str = "sample_data/synthetic_mi_raw.fif", n_trials: int = 40) -> str:
    """Write a small synthetic recording to disk, handy for trying the upload path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw = make_synthetic_raw(SyntheticParams(n_trials=n_trials))
    raw.save(path, overwrite=True, fmt="single", verbose=False)
    return path


if __name__ == "__main__":
    print("Saved", save_sample_file())

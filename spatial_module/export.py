"""Export the current settings as JSON and as a runnable Python script."""

from __future__ import annotations

import json
from datetime import datetime

PARAM_KEYS = [
    "source", "synthetic", "eegbci_subject", "upload_name", "class_a", "class_b",
    "reference", "ref_channel", "n_neighbours", "l_freq", "h_freq", "tmin", "tmax",
    "metric", "csp_channels", "n_components", "csp_reg", "shrinkage",
]


def params_to_json(params: dict) -> str:
    clean = {k: params.get(k) for k in PARAM_KEYS}
    clean["exported_at"] = datetime.now().isoformat(timespec="seconds")
    clean["module"] = "spatial_analysis"
    return json.dumps(clean, indent=2, default=str)


def params_from_json(text: str) -> dict:
    data = json.loads(text)
    return {k: data[k] for k in PARAM_KEYS if k in data}


_TEMPLATE = '''"""Spatial analysis pipeline exported from the SYNAPSE BCI teaching app.

Run from the repository root:
    python {filename}

It repeats exactly what the app did with these settings and saves two
figures (topomap and CSP patterns) next to this script.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from spatial_module import data as D
from spatial_module import processing as P

PARAMS = {params_literal}

# 1. Load data -------------------------------------------------------------
{load_block}
X, y, times, class_names = D.epochs_to_arrays(epochs)
print(epochs)

# 2. Re-reference / spatial filter -----------------------------------------
X_ref = P.apply_reference(X, epochs.info, PARAMS["reference"],
                          PARAMS["ref_channel"], PARAMS["n_neighbours"])

# 3. Band-pass (zero-phase Butterworth, order 4) ---------------------------
X_band = P.bandpass(X_ref, epochs.info["sfreq"], PARAMS["l_freq"], PARAMS["h_freq"])

# 4. Topomap of the chosen metric inside the time window -------------------
topo, unit = P.topo_metric(X_band, y, times, (PARAMS["tmin"], PARAMS["tmax"]),
                           PARAMS["metric"])
fig, ax = plt.subplots(figsize=(4, 4))
mne.viz.plot_topomap(topo, epochs.info, axes=ax, show=False, cmap="RdBu_r")
ax.set_title(f"{{PARAMS['metric']}} ({{unit}})")
fig.savefig("export_topomap.png", dpi=150, bbox_inches="tight")

# 5. CSP + LDA, cross-validated without leakage ---------------------------
picks = [i for i, ch in enumerate(epochs.ch_names) if ch in PARAMS["csp_channels"]]
X_win, _ = P.crop(X_band[:, picks, :], times, PARAMS["tmin"], PARAMS["tmax"])
X_win, csp_names, _ = P.drop_flat_channels(X_win, [epochs.ch_names[i] for i in picks])
reg = P.parse_reg(PARAMS["csp_reg"], PARAMS["shrinkage"])
scores = P.csp_cross_val(X_win, y, PARAMS["n_components"], reg)
chance = P.chance_level(len(y))
print(f"CSP+LDA accuracy: {{scores.mean():.3f}} +/- {{scores.std():.3f}} "
      f"(chance at p<0.05: {{chance:.3f}})")

csp, _ = P.fit_csp(X_win, y, PARAMS["n_components"], reg)
info_csp = mne.pick_info(epochs.info, [epochs.ch_names.index(c) for c in csp_names])
fig, axes = plt.subplots(1, PARAMS["n_components"], figsize=(3 * PARAMS["n_components"], 3))
for k, ax in enumerate(np.atleast_1d(axes)):
    mne.viz.plot_topomap(csp.patterns_[k], info_csp, axes=ax, show=False, cmap="RdBu_r")
    ax.set_title(f"CSP pattern {{k + 1}}")
fig.savefig("export_csp_patterns.png", dpi=150, bbox_inches="tight")
print("Saved export_topomap.png and export_csp_patterns.png")
'''


def _load_block(params: dict) -> str:
    src = params.get("source")
    if src == "synthetic":
        return ('raw = D.make_synthetic_raw(D.SyntheticParams(**PARAMS["synthetic"]))\n'
                'epochs = D.raw_to_epochs(raw, "left_hand", "right_hand")')
    if src == "eegbci":
        return ('raw = D.load_eegbci(PARAMS["eegbci_subject"])  # downloads once\n'
                'epochs = D.raw_to_epochs(raw, "left_hand", "right_hand")')
    name = params.get("upload_name") or "your_file.fif"
    return (f'# Put your file next to this script (or change the path).\n'
            f'path = "{name}"\n'
            'if path.endswith("-epo.fif"):\n'
            '    epochs = D.prepare_epochs(mne.read_epochs(path, preload=True),\n'
            '                              PARAMS["class_a"], PARAMS["class_b"])\n'
            'else:\n'
            '    raw = mne.io.read_raw(path, preload=True)\n'
            '    epochs = D.raw_to_epochs(raw, PARAMS["class_a"], PARAMS["class_b"])')


def params_to_script(params: dict, filename: str = "spatial_pipeline.py") -> str:
    clean = {k: params.get(k) for k in PARAM_KEYS}
    return _TEMPLATE.format(filename=filename, params_literal=repr(clean),
                            load_block=_load_block(params))

"""Tests for the Streamlit app, file upload path and code export."""

import json
import os
import subprocess
import sys

import mne
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from spatial_module import data as D
from spatial_module import export as X_

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARAMS = dict(source="synthetic",
              synthetic=dict(n_trials=40, erd_strength=0.35, common_noise_uv=20.0,
                             occipital_alpha_uv=12.0, seed=3),
              reference="Hjorth Laplacian", ref_channel="Cz", n_neighbours=4,
              l_freq=8.0, h_freq=13.0, tmin=0.5, tmax=3.0, metric="difference",
              csp_channels=D.SENSORIMOTOR_CHANNELS, n_components=4, csp_reg="ledoit_wolf",
              shrinkage=0.1)


def _app():
    return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120)


def test_app_starts_without_errors():
    at = _app().run()
    assert not at.exception
    assert not at.error
    labels = [m.label for m in at.metric]
    assert "CSP + LDA accuracy" in labels


def test_app_reacts_to_reference_change():
    at = _app().run()
    before = next(m.value for m in at.metric if m.label == "C3 + C4 power accuracy")
    at.selectbox(key="ref").select("Common average (CAR)").run()
    assert not at.exception
    after = next(m.value for m in at.metric if m.label == "C3 + C4 power accuracy")
    # Removing the noisy reference should help the plain channel-power feature.
    assert int(after.rstrip("%")) > int(before.rstrip("%"))


def test_quiz_gives_feedback():
    at = _app().run()
    at.radio(key="quiz_1").set_value("Over C3 (left hemisphere)").run()
    assert any("Correct" in s.value for s in at.success)


def test_params_json_roundtrip():
    text = X_.params_to_json(PARAMS)
    data = json.loads(text)
    assert data["module"] == "spatial_analysis"
    back = X_.params_from_json(text)
    assert back["reference"] == "Hjorth Laplacian" and back["l_freq"] == 8.0


def test_exported_script_runs(tmp_path):
    script = tmp_path / "spatial_pipeline.py"
    script.write_text(X_.params_to_script(PARAMS))
    env = dict(os.environ, PYTHONPATH=ROOT, MPLBACKEND="Agg")
    out = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "CSP+LDA accuracy" in out.stdout
    assert (tmp_path / "export_topomap.png").exists()
    assert (tmp_path / "export_csp_patterns.png").exists()


def test_upload_path_with_fif_file(tmp_path):
    path = D.save_sample_file(str(tmp_path / "sample_raw.fif"), n_trials=20)
    with open(path, "rb") as fh:
        inst = D.load_uploaded(fh.read(), "sample_raw.fif")
    labels = D.event_labels(inst)
    assert labels == {"left_hand": 10, "right_hand": 10}
    epochs = D.raw_to_epochs(inst, "left_hand", "right_hand")
    assert len(epochs) == 20


def test_upload_without_positions_gets_standard_montage(tmp_path):
    raw = D.make_synthetic_raw(D.SyntheticParams(n_trials=10))
    raw.set_montage(None)
    raw.rename_channels({c: c.upper() for c in raw.ch_names})  # e.g. 'FP1', 'CZ'
    path = tmp_path / "nomontage_raw.fif"
    raw.save(path, verbose=False)
    inst = D.load_uploaded(path.read_bytes(), "nomontage_raw.fif")
    ep = D.raw_to_epochs(inst, "left_hand", "right_hand")
    locs = np.array([ch["loc"][:3] for ch in ep.info["chs"]])
    assert np.all(np.isfinite(locs)) and np.all(np.abs(locs).sum(axis=1) > 0)


def test_upload_epochs_file(tmp_path):
    raw = D.make_synthetic_raw(D.SyntheticParams(n_trials=10))
    ep = D.raw_to_epochs(raw, "left_hand", "right_hand")
    path = tmp_path / "x-epo.fif"
    ep.save(path, verbose=False)
    inst = D.load_uploaded(path.read_bytes(), "x-epo.fif")
    assert isinstance(inst, mne.BaseEpochs)
    assert len(D.prepare_epochs(inst)) == 10

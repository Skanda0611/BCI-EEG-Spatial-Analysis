"""Unit tests for the spatial processing steps. Run with:  pytest -q"""

import numpy as np
import pytest

from spatial_module import data as D
from spatial_module import processing as P
from spatial_module import plots as PL


@pytest.fixture(scope="module")
def epochs():
    raw = D.make_synthetic_raw(D.SyntheticParams(n_trials=40, seed=1))
    return D.raw_to_epochs(raw, "left_hand", "right_hand")


@pytest.fixture(scope="module")
def arrays(epochs):
    return D.epochs_to_arrays(epochs)


def test_synthetic_data_shape(epochs, arrays):
    X, y, times, names = arrays
    assert X.shape == (40, len(D.SYNTHETIC_CHANNELS), len(times))
    assert names == ["left_hand", "right_hand"]
    assert np.bincount(y).tolist() == [20, 20]
    assert times[0] == pytest.approx(-1.0) and times[-1] == pytest.approx(4.0)


def test_car_removes_common_signal(arrays):
    X = arrays[0]
    Xc = P.common_average(X)
    # Mean across channels is zero at every sample after CAR.
    assert np.allclose(Xc.mean(axis=1), 0, atol=1e-15)
    # Adding the same signal to every channel changes nothing after CAR.
    shared = np.random.default_rng(0).standard_normal(X.shape[-1]) * 1e-5
    assert np.allclose(P.common_average(X + shared), Xc, atol=1e-15)


def test_single_reference_zeroes_reference(epochs, arrays):
    X = arrays[0]
    Xr = P.apply_reference(X, epochs.info, "Single electrode", "Cz")
    assert np.allclose(Xr[:, epochs.ch_names.index("Cz")], 0)
    kept_X, kept, _ = P.drop_flat_channels(Xr, epochs.ch_names)
    assert "Cz" not in kept and len(kept) == len(epochs.ch_names) - 1


def test_hjorth_removes_uniform_field(epochs):
    pos = P.channel_positions(epochs.info)
    flat = np.ones((2, len(pos), 50))
    assert np.allclose(P.hjorth_laplacian(flat, pos, k=4), 0)
    nb = P.neighbours(pos, k=4)
    c3 = epochs.ch_names.index("C3")
    neighbour_names = {epochs.ch_names[i] for i in nb[c3]}
    assert neighbour_names <= {"C5", "C1", "FC3", "CP3", "FC5", "CP5", "FC1", "CP1"}


def test_csd_runs_and_changes_data(epochs, arrays):
    X = arrays[0][:4]
    Xcsd = P.apply_reference(X, epochs.info, "Surface Laplacian (CSD)")
    assert Xcsd.shape == X.shape and np.all(np.isfinite(Xcsd))
    assert not np.allclose(Xcsd, X)


def test_bandpass_keeps_band_and_rejects_outside():
    sfreq = 160.0
    t = np.arange(0, 4, 1 / sfreq)
    sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 35 * t)
    out = P.bandpass(sig[None, None, :], sfreq, 8, 13)[0, 0]
    mid = slice(160, -160)
    ten = np.sin(2 * np.pi * 10 * t)
    assert np.corrcoef(out[mid], ten[mid])[0, 1] > 0.98


def test_topomap_shows_contralateral_erd(epochs, arrays):
    """Right-hand imagery suppresses mu at C3, left-hand at C4, so the
    left-vs-right power difference must be positive at C3 and negative at C4."""
    X, y, times, _ = arrays
    Xf = P.bandpass(P.common_average(X), epochs.info["sfreq"], 8, 13)
    diff, unit = P.topo_metric(Xf, y, times, (0.5, 3.0), "difference")
    names = epochs.ch_names
    assert diff[names.index("C3")] > 1.0
    assert diff[names.index("C4")] < -1.0
    erd_right, _ = P.topo_metric(Xf, y, times, (0.5, 3.0), "erd_b")
    assert erd_right[names.index("C3")] < -15  # % drop vs baseline


def test_csp_cross_validation_beats_chance(arrays):
    X, y, times, _ = arrays
    Xf = P.bandpass(P.common_average(X), 160.0, 8, 13)
    Xw, _ = P.crop(Xf, times, 0.5, 3.0)
    scores = P.csp_cross_val(Xw, y, n_components=4)
    assert scores.mean() > P.chance_level(len(y))


def test_csp_components_alternate_between_classes(arrays):
    X, y, times, _ = arrays
    Xw, _ = P.crop(P.bandpass(P.common_average(X), 160.0, 8, 13), times, 0.5, 3.0)
    csp, feats = P.fit_csp(Xw, y, 4)
    assert feats[y == 0, 0].mean() > feats[y == 1, 0].mean()  # CSP 1 favours class A
    assert feats[y == 1, 1].mean() > feats[y == 0, 1].mean()  # CSP 2 favours class B
    assert csp.patterns_.shape[1] == X.shape[1]


@pytest.mark.parametrize("ref", P.REFERENCES)
def test_every_reference_runs_through_pipeline(epochs, ref):
    out = P.run_pipeline(epochs, dict(reference=ref, ref_channel="Cz", l_freq=8, h_freq=13,
                                      tmin=0.5, tmax=3.0, n_components=4))
    assert 0 <= out["scores"].mean() <= 1
    assert np.all(np.isfinite(out["topo"]))


def test_chance_level_is_stricter_for_few_trials():
    assert P.chance_level(20) > P.chance_level(200) > 0.5
    assert P.chance_level(20) == pytest.approx(0.7)


def test_projection_puts_cz_at_centre(epochs):
    xy = PL.project_2d(P.channel_positions(epochs.info))
    names = epochs.ch_names
    assert np.hypot(*xy[names.index("Cz")]) < 0.15
    assert xy[names.index("C3"), 0] < 0 < xy[names.index("C4"), 0]   # left is left
    assert xy[names.index("Fz"), 1] > 0 > xy[names.index("Oz"), 1]   # nose is up

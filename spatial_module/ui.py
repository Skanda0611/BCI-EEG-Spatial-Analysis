"""Streamlit user interface for the Spatial Analysis module.

``render()`` draws the whole module, so the shared course app can call it
from its own page (see docs/DEVELOPER_GUIDE.md). ``app.py`` calls it for
stand-alone use.
"""

from __future__ import annotations

import hashlib

import numpy as np
import streamlit as st

from . import data as D
from . import explain as E
from . import export as X_
from . import plots as PL
from . import processing as P

MODULE_INFO = dict(
    id="spatial_analysis",
    title="Spatial analysis",
    team="SYNAPSE",
    summary="Re-referencing, scalp topographies, Laplacians and CSP spatial filters.",
)

METRICS = {
    "Class difference (dB)": "difference",
    "ERD % - class A": "erd_a",
    "ERD % - class B": "erd_b",
    "Band power - class A": "power_a",
    "Band power - class B": "power_b",
}

DEFAULTS = dict(
    src="Synthetic motor imagery (built-in)", syn_trials=80, syn_erd=0.35, syn_common=20.0,
    syn_alpha=12.0, syn_seed=42, subject=1, ref="As recorded", ref_ch="Cz", nbrs=4,
    band_preset="Mu (8-13 Hz)", band=(8.0, 13.0), win=(0.5, 3.0),
    metric="Class difference (dB)", csp_set="All channels", ncomp=4, reg="None",
    shrink=0.1, trial=0,
)

SOURCES = ["Synthetic motor imagery (built-in)", "PhysioNet EEGBCI (download)", "Upload a file"]


# --------------------------------------------------------------------------
# Cached computations
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Simulating EEG...", max_entries=4)
def _synthetic_epochs(n_trials, erd, common, alpha, seed):
    raw = D.make_synthetic_raw(D.SyntheticParams(n_trials, erd, common, alpha, seed))
    return D.raw_to_epochs(raw, "left_hand", "right_hand")


@st.cache_resource(show_spinner="Downloading / loading PhysioNet data...", max_entries=3)
def _eegbci_epochs(subject):
    return D.raw_to_epochs(D.load_eegbci(subject), "left_hand", "right_hand")


@st.cache_resource(show_spinner="Reading file...", max_entries=2)
def _uploaded(digest, _bytes, name):
    return D.load_uploaded(_bytes, name)


@st.cache_resource(show_spinner="Epoching...", max_entries=4)
def _uploaded_epochs(digest, class_a, class_b, tmin, tmax, _inst):
    if hasattr(_inst, "event_id"):
        return D.prepare_epochs(_inst, class_a, class_b)
    return D.raw_to_epochs(_inst, class_a, class_b, tmin, tmax)


@st.cache_data(show_spinner=False, max_entries=8)
def _referenced(key, ref, ref_ch, nbrs, _ep):
    X, y, times, names = D.epochs_to_arrays(_ep)
    return P.apply_reference(X, _ep.info, ref, ref_ch, nbrs)


@st.cache_data(show_spinner=False, max_entries=8)
def _banded(key, ref, ref_ch, nbrs, lo, hi, _ep):
    Xr = _referenced(key, ref, ref_ch, nbrs, _ep)
    return P.bandpass(Xr, _ep.info["sfreq"], lo, hi)


@st.cache_data(show_spinner="Running CSP with cross-validation...", max_entries=16)
def _csp(key, ref, ref_ch, nbrs, lo, hi, tmin, tmax, channels, ncomp, reg, _ep):
    Xf = _banded(key, ref, ref_ch, nbrs, lo, hi, _ep)
    _, y, times, _ = D.epochs_to_arrays(_ep)
    names = _ep.ch_names
    picks = [i for i, c in enumerate(names) if c in channels]
    Xw, _ = P.crop(Xf[:, picks, :], times, tmin, tmax)
    Xw, kept, kept_idx = P.drop_flat_channels(Xw, [names[i] for i in picks])
    ncomp = min(ncomp, len(kept))
    scores = P.csp_cross_val(Xw, y, ncomp, reg)
    csp, feats = P.fit_csp(Xw, y, ncomp, reg)
    Xall, _ = P.crop(Xf, times, tmin, tmax)
    chan = P.channel_power_cross_val(Xall, y, names)
    return dict(scores=scores, chan_scores=chan, patterns=csp.patterns_[:ncomp],
                feats=feats, kept=kept, y=y)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _init_state():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)


def _on_preset():
    rng = P.BAND_PRESETS.get(st.session_state.band_preset)
    if rng:
        st.session_state.band = rng


def _on_band():
    if P.BAND_PRESETS.get(st.session_state.band_preset) != tuple(st.session_state.band):
        st.session_state.band_preset = "Custom"


def _on_params_upload():
    f = st.session_state.get("params_file")
    if f is None:
        return
    try:
        p = X_.params_from_json(f.getvalue().decode())
    except Exception:  # noqa: BLE001
        st.session_state["params_error"] = "That file is not a valid parameter export."
        return
    syn = p.get("synthetic") or {}
    mapping = dict(reference="ref", ref_channel="ref_ch", n_neighbours="nbrs",
                   n_components="ncomp", csp_reg="reg", shrinkage="shrink")
    for src_key, widget in mapping.items():
        if p.get(src_key) is not None:
            st.session_state[widget] = p[src_key]
    if p.get("l_freq") is not None:
        st.session_state.band = (float(p["l_freq"]), float(p["h_freq"]))
        st.session_state.band_preset = "Custom"
    if p.get("tmin") is not None:
        st.session_state.win = (float(p["tmin"]), float(p["tmax"]))
    inv = {v: k for k, v in METRICS.items()}
    if p.get("metric") in inv:
        st.session_state.metric = inv[p["metric"]]
    for k, w in dict(n_trials="syn_trials", erd_strength="syn_erd",
                     common_noise_uv="syn_common", occipital_alpha_uv="syn_alpha",
                     seed="syn_seed").items():
        if k in syn:
            st.session_state[w] = syn[k]
    st.session_state["params_error"] = None


def _sidebar():
    """All user controls. Returns (epochs or None, data_key, params dict)."""
    sb = st.sidebar
    sb.header("01  ·  EEG DATA")
    src = sb.radio("Source", SOURCES, key="src")
    params = dict(synthetic=None, eegbci_subject=None, upload_name=None,
                  class_a=None, class_b=None)
    epochs, key = None, None

    if src == SOURCES[0]:
        params["source"] = "synthetic"
        with sb.expander("Simulation settings", expanded=False):
            n = st.slider("Trials", 20, 200, step=10, key="syn_trials")
            erd = st.slider("ERD strength", 0.0, 0.9, step=0.05, key="syn_erd",
                            help="How much the mu rhythm drops over the contralateral motor cortex.")
            common = st.slider("Noisy-reference amplitude (µV)", 0.0, 60.0, step=5.0,
                               key="syn_common",
                               help="Noise added identically to every channel. CAR removes it.")
            alpha = st.slider("Occipital alpha amplitude (µV)", 0.0, 40.0, step=2.0,
                              key="syn_alpha",
                              help="Task-unrelated alpha that spreads across the scalp.")
            seed = st.number_input("Random seed", 0, 9999, key="syn_seed")
        params["synthetic"] = dict(n_trials=n, erd_strength=erd, common_noise_uv=common,
                                   occipital_alpha_uv=alpha, seed=int(seed))
        epochs = _synthetic_epochs(n, erd, common, alpha, int(seed))
        key = f"syn|{n}|{erd}|{common}|{alpha}|{seed}"

    elif src == SOURCES[1]:
        params["source"] = "eegbci"
        subject = sb.number_input("Subject (1-109)", 1, 109, key="subject")
        params["eegbci_subject"] = int(subject)
        sb.caption("Runs 4, 8, 12: imagined left vs right fist. About 7 MB per subject, "
                   "cached in ~/mne_data.")
        try:
            epochs = _eegbci_epochs(int(subject))
            key = f"eegbci|{subject}"
        except Exception as exc:  # noqa: BLE001
            sb.error(f"Could not load PhysioNet data (internet needed the first time): {exc}")

    else:
        params["source"] = "upload"
        f = sb.file_uploader("EEG file", type=["fif", "edf", "bdf", "gdf", "set"],
                             key="upload",
                             help="Continuous recordings need events (annotations or a stim "
                                  "channel). Epoched files must be *-epo.fif.")
        if f is not None:
            raw_bytes = f.getvalue()
            digest = hashlib.md5(raw_bytes).hexdigest()
            params["upload_name"] = f.name
            try:
                inst = _uploaded(digest, raw_bytes, f.name)
                if hasattr(inst, "event_id"):
                    labels = {k: len(inst[k]) for k in inst.event_id}
                else:
                    labels = D.event_labels(inst)
                if len(labels) < 2:
                    sb.error("This file needs at least two event types.")
                else:
                    opts = [f"{k} ({v})" for k, v in labels.items()]
                    names = list(labels)
                    a = sb.selectbox("Class A", opts, index=0, key="cls_a")
                    b = sb.selectbox("Class B", opts, index=1, key="cls_b")
                    ca, cb = names[opts.index(a)], names[opts.index(b)]
                    tmin_e, tmax_e = -1.0, 4.0
                    if not hasattr(inst, "event_id"):
                        tmin_e, tmax_e = sb.slider("Epoch from/to cue (s)", -2.0, 6.0,
                                                   (-1.0, 4.0), 0.5, key="ep_win")
                    if ca == cb:
                        sb.warning("Pick two different classes.")
                    else:
                        params["class_a"], params["class_b"] = ca, cb
                        epochs = _uploaded_epochs(digest, ca, cb, tmin_e, tmax_e, inst)
                        key = f"up|{digest}|{ca}|{cb}|{tmin_e}|{tmax_e}"
            except Exception as exc:  # noqa: BLE001
                sb.error(f"Could not read this file: {exc}")
        else:
            sb.info("Upload a file, or run `python -m spatial_module.data` to create "
                    "`sample_data/synthetic_mi_raw.fif` to try this path.")

    if epochs is None:
        return None, None, params

    names = epochs.ch_names
    sb.header("02  ·  REFERENCE & FILTER")
    ref = sb.selectbox("Scheme", P.REFERENCES, key="ref")
    if st.session_state.ref_ch not in names:
        st.session_state.ref_ch = "Cz" if "Cz" in names else names[0]
    ref_ch = sb.selectbox("Reference electrode", names, key="ref_ch",
                          disabled=ref != "Single electrode")
    nbrs = sb.slider("Laplacian neighbours", 3, 8, key="nbrs",
                     disabled=ref != "Hjorth Laplacian")

    sb.header("03  ·  FREQUENCY & TIME")
    sb.selectbox("Band preset", list(P.BAND_PRESETS), key="band_preset", on_change=_on_preset)
    nyq = float(epochs.info["sfreq"]) / 2 - 1
    lo, hi = sb.slider("Band (Hz)", 1.0, min(45.0, nyq), step=0.5, key="band",
                       on_change=_on_band)
    if hi - lo < 1:
        hi = lo + 1
    t0, t1 = float(epochs.times[0]), float(epochs.times[-1])
    w = st.session_state.win
    if w[0] < t0 or w[1] > t1:
        st.session_state.win = (max(t0, 0.5 if t1 > 0.6 else t0), min(t1, 3.0))
    tmin, tmax = sb.slider("Analysis window (s from cue)", t0, t1, step=0.1, key="win")

    sb.header("04  ·  SCALP TOPOGRAPHY")
    metric_label = sb.selectbox("Show", list(METRICS), key="metric")

    sb.header("05  ·  CSP SETTINGS")
    sets = ["All channels", "Sensorimotor strip", "Custom"]
    csp_set = sb.radio("Channels for CSP", sets, key="csp_set", horizontal=True)
    if csp_set == "All channels":
        chans = list(names)
    elif csp_set == "Sensorimotor strip":
        chans = [c for c in names if c in D.SENSORIMOTOR_CHANNELS] or list(names)
    else:
        chans = sb.multiselect("Pick channels", names,
                               default=[c for c in names if c in D.SENSORIMOTOR_CHANNELS][:8]
                               or names[:8], key="csp_custom")
    ncomp = sb.select_slider("Components", [2, 4, 6, 8], key="ncomp")
    reg_choice = sb.selectbox("Covariance regularisation",
                              ["None", "ledoit_wolf", "oas", "Shrinkage (manual)"], key="reg")
    shrink = sb.slider("Shrinkage", 0.0, 1.0, step=0.05, key="shrink",
                       disabled=reg_choice != "Shrinkage (manual)")

    params.update(reference=ref, ref_channel=ref_ch, n_neighbours=int(nbrs),
                  l_freq=float(lo), h_freq=float(hi), tmin=float(tmin), tmax=float(tmax),
                  metric=METRICS[metric_label], metric_label=metric_label,
                  csp_channels=list(chans), n_components=int(ncomp),
                  csp_reg=reg_choice, shrinkage=float(shrink))
    return epochs, key, params


# --------------------------------------------------------------------------
# Page sections
# --------------------------------------------------------------------------

def _topography_tab(epochs, key, params, ctx):
    names = epochs.ch_names
    X, y, times, classes = D.epochs_to_arrays(epochs)
    xy = PL.project_2d(np.array([ch["loc"][:3] for ch in epochs.info["chs"]]))
    sfreq = epochs.info["sfreq"]
    ref, ref_ch, nbrs = params["reference"], params["ref_channel"], params["n_neighbours"]
    lo, hi, tmin, tmax = params["l_freq"], params["h_freq"], params["tmin"], params["tmax"]

    Xr = _referenced(key, ref, ref_ch, nbrs, epochs)
    Xf = _banded(key, ref, ref_ch, nbrs, lo, hi, epochs)

    metric = params["metric"]
    label = params["metric_label"].replace("class A", classes[0]).replace("class B", classes[1])
    try:
        values, unit = P.topo_metric(Xf, y, times, (tmin, tmax), metric)
    except ValueError as exc:
        st.warning(f"{exc} Showing the class difference instead.")
        values, unit = P.topo_metric(Xf, y, times, (tmin, tmax), "difference")
        label = "Class difference (dB)"
    ctx["topo"] = dict(zip(names, values))

    if st.session_state.get("sel_ch") not in names:
        st.session_state.sel_ch = "C3" if "C3" in names else names[0]
    sel = st.session_state.sel_ch

    left, right = st.columns([1.10, 1.00], gap="large")
    with left:
        st.caption("Scalp distribution · select an electrode to inspect its signal")
        title = f"{label} · {lo:g}-{hi:g} Hz · {tmin:g}-{tmax:g} s"
        if metric == "difference":
            title = f"{classes[0]} vs {classes[1]} power (dB) · {lo:g}-{hi:g} Hz"
        fig = PL.topomap_figure(
    xy, values, names, title,
    unit=unit,
    selected=sel,
    height=520
)
        event = st.plotly_chart(fig, key="topo", on_select="rerun",
                                selection_mode=("points",), width="stretch",
                                config={"displayModeBar": False})
        clicked = [p.get("customdata") for p in (event.selection.points if event else [])
                   if isinstance(p.get("customdata"), str)]
        # The figure changes after a click (the new channel is highlighted), so
        # Plotly clears the selection on the next run and this cannot loop.
        if clicked and clicked[-1] != sel:
            st.session_state.sel_ch = clicked[-1]
            st.rerun()
        st.caption("Click an electrode to inspect it on the right. Red means more power "
                   f"for **{classes[0]}**, blue more for **{classes[1]}** (for the difference map).")

    with right:
        c1, c2 = st.columns([1, 1])
        c1.selectbox("Channel", names, key="sel_ch")
        n_tr = len(y)
        if st.session_state.trial >= n_tr:
            st.session_state.trial = 0
        trial = c2.slider("Example trial", 0, n_tr - 1, key="trial")
        sel = st.session_state.sel_ch
        ci = names.index(sel)
        before = X[trial, ci] * 1e6
        after = Xr[trial, ci]
        unit_after = "µV"
        if ref == "Surface Laplacian (CSD)":
            s = after.std()
            after = after / s * before.std() if s > 0 else after
            unit_after = "µV (CSD rescaled)"
        else:
            after = after * 1e6
        env = P.envelope_power(Xf[:, ci, :], sfreq)
        # Drop 0.3 s at each end: filtering and smoothing distort epoch edges.
        edge = (times >= times[0] + 0.3) & (times <= times[-1] - 0.3)
        step = max(1, int(sfreq // 40))
        env_t = times[edge][::step]
        ea = env[y == 0].mean(0)[edge][::step]
        eb = env[y == 1].mean(0)[edge][::step]
        base = (times >= -1.0) & (times <= 0.0) & edge
        if base.sum() > 5:
            ba, bb = env[y == 0][:, base].mean(), env[y == 1][:, base].mean()
            ea, eb = 100 * ea / ba, 100 * eb / bb
        st.plotly_chart(PL.signal_figure(times, before, after, env_t, ea, eb, classes, sel,
                                         (tmin, tmax), unit_after,
                                         ref_label=ref.split(" (")[0], height=420),
                        width="stretch", key="sig")

    Xw, _ = P.crop(Xr[:, ci, :], times, tmin, tmax)
    st.plotly_chart(PL.spectrum_figure(Xw[y == 0], Xw[y == 1], sfreq, classes, sel, (lo, hi)),
                    width="stretch", key="psd")


def _csp_tab(epochs, key, params, ctx):
    names = epochs.ch_names
    classes = list(epochs.event_id)
    if len(params["csp_channels"]) < 2:
        st.warning("Pick at least two channels for CSP.")
        return
    reg = P.parse_reg(params["csp_reg"], params["shrinkage"])
    args = (key, params["reference"], params["ref_channel"], params["n_neighbours"],
            params["l_freq"], params["h_freq"], params["tmin"], params["tmax"],
            tuple(params["csp_channels"]), params["n_components"], reg)
    try:
        res = _csp(*args, epochs)
    except Exception as exc:  # noqa: BLE001
        st.error(f"CSP failed with these settings: {exc}. Try adding regularisation.")
        return
    chance = P.chance_level(len(res["y"]))
    acc, chan = float(res["scores"].mean()), float(res["chan_scores"].mean())
    ctx.update(csp_acc=acc, chan_acc=chan, chance=chance, n_trials=len(res["y"]))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CSP + LDA accuracy", f"{acc:.0%}", f"± {res['scores'].std():.0%} over folds",
              delta_color="off")
    m2.metric("C3 + C4 power accuracy", f"{chan:.0%}", "no spatial filter learned",
              delta_color="off")
    m3.metric("Chance level (p < 0.05)", f"{chance:.0%}", f"{len(res['y'])} trials",
              delta_color="off")
    m4.metric("Channels used", len(res["kept"]))

    kept = res["kept"]
    idx = [names.index(c) for c in kept]
    xy = PL.project_2d(np.array([epochs.info["chs"][i]["loc"][:3] for i in idx]))
    ncomp = len(res["patterns"])
    titles = []
    for k in range(ncomp):
        favours = classes[0] if k % 2 == 0 else classes[1]
        titles.append(f"CSP {k + 1} · {favours} ↑")
    st.markdown("**CSP patterns**: how each learned source projects onto the scalp. "
                "“↑” marks the class with the higher variance for that component. "
                "Colour sign is arbitrary, so only the shape matters.")
    st.plotly_chart(PL.topomap_grid(xy, list(res["patterns"]), kept, titles,
                                    cols=min(4, ncomp)), width="stretch", key="pat")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(PL.csp_scatter(res["feats"], res["y"], classes),
                        width="stretch", key="scat")
    with c2:
        st.markdown("**Which reference helps which feature?**")
        st.caption("Runs CSP + LDA and the C3 + C4 baseline for every reference scheme, "
                   "with all your other settings unchanged.")
        if st.button("Compare all reference schemes", key="cmp"):
            csp_acc, chan_acc = [], []
            with st.spinner("Running 5 × cross-validation..."):
                for r in P.REFERENCES:
                    rc = params["ref_channel"] if r == "Single electrode" else None
                    rr = _csp(key, r, rc, params["n_neighbours"], *args[4:], epochs)
                    csp_acc.append(float(rr["scores"].mean()))
                    chan_acc.append(float(rr["chan_scores"].mean()))
            st.session_state["cmp_result"] = (key, args[4:], csp_acc, chan_acc)
        cmp = st.session_state.get("cmp_result")
        if cmp and cmp[0] == key and cmp[1] == args[4:]:
            labels = [r.split(" (")[0] for r in P.REFERENCES]
            st.plotly_chart(PL.accuracy_bar(labels, cmp[2], cmp[3], chance),
                            width="stretch", key="bar")


def _explain_tab(params, ctx):
    st.subheader("What is happening right now")
    for note in E.live_notes(params, ctx):
        st.info(note)
    topo = ctx.get("topo") or {}
    if "C3" in topo and "C4" in topo and params["metric"] == "difference":
        st.write(f"Class difference at **C3 = {topo['C3']:+.2f} dB** and **C4 = {topo['C4']:+.2f} "
                 "dB**. Opposite signs over the two hemispheres are the classic motor-imagery "
                 "signature.")
    st.subheader("Concepts")
    for title, body in E.CONCEPTS.items():
        with st.expander(title):
            st.markdown(body)
    st.subheader("Check yourself")
    correct = 0
    for i, q in enumerate(E.QUESTIONS):
        ans = st.radio(f"**Q{i + 1}.** {q['q']}", q["options"], index=None, key=f"quiz_{i}")
        if ans is not None:
            if q["options"].index(ans) == q["answer"]:
                correct += 1
                st.success(f"Correct. {q['why']}")
            else:
                st.error(f"Not quite. {q['why']}")
    st.progress(correct / len(E.QUESTIONS), text=f"{correct} / {len(E.QUESTIONS)} correct")


def _export_tab(params):
    st.subheader("Take this analysis with you")
    st.write("Download the current settings, or a Python script that reruns the same "
             "pipeline outside the app and saves the figures.")
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Parameters (JSON)", X_.params_to_json(params),
                       file_name="spatial_params.json", mime="application/json",
                       width="stretch")
    script = X_.params_to_script(params)
    c2.download_button("⬇ Python script", script, file_name="spatial_pipeline.py",
                       mime="text/x-python", width="stretch")
    with st.expander("Preview script"):
        st.code(script, language="python")
    st.divider()
    st.file_uploader("Load parameters from a JSON export", type=["json"], key="params_file",
                     on_change=_on_params_upload)
    if st.session_state.get("params_error"):
        st.error(st.session_state.params_error)



def _apply_theme():
    """Apply the visual theme without changing analysis behaviour."""
    st.markdown(
        """
        <style>
        :root {
            --bci-ink: #14243a;
            --bci-muted: #64748b;
            --bci-teal: #0e9f9a;
            --bci-cyan: #20b8c5;
            --bci-line: #dce6ef;
            --bci-panel: #ffffff;
            --bci-bg: #f3f7fb;
        }
        .stApp {
            background: var(--bci-bg);
            color: var(--bci-ink);
        }
        [data-testid="stHeader"] {
            background: rgba(243, 247, 251, 0.92);
        }
        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: #edf3f8;
            border-right: 1px solid var(--bci-line);
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.8rem;
        }
        h1, h2, h3 {
            color: var(--bci-ink);
            letter-spacing: -0.035em;
        }
        h1 { font-weight: 750 !important; }
        h2, h3 { font-weight: 680 !important; }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"] {
            color: #52647a;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.45rem;
            border-bottom: 1px solid var(--bci-line);
        }
        [data-testid="stTabs"] button[role="tab"] {
            background: transparent;
            border-radius: 10px 10px 0 0;
            padding: 0.75rem 1rem;
            color: #53657a;
            font-weight: 600;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #087f86;
            border-bottom: 3px solid var(--bci-teal);
        }
        [data-testid="stMetric"] {
            background: var(--bci-panel);
            border: 1px solid var(--bci-line);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 3px 12px rgba(24, 52, 78, 0.045);
        }
        [data-testid="stMetricLabel"] {
            color: var(--bci-muted);
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: var(--bci-ink);
            font-weight: 750;
        }
        [data-testid="stPlotlyChart"] {
            background: var(--bci-panel);
            border: 1px solid var(--bci-line);
            border-radius: 14px;
            padding: 0.35rem;
            box-shadow: 0 3px 12px rgba(24, 52, 78, 0.035);
        }
        [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid #b8d3dc;
            font-weight: 650;
            transition: all 120ms ease-in-out;
        }
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
            background: var(--bci-teal);
            border-color: var(--bci-teal);
            color: white;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--bci-teal);
            color: #087f86;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--bci-line);
            border-radius: 12px;
            background: rgba(255,255,255,0.72);
        }
        [data-testid="stAlert"] {
            border-radius: 12px;
        }
        .bci-hero {
            position: relative;
            overflow: hidden;
            padding: 1.35rem 1.6rem;
            margin: 0.15rem 0 1.15rem 0;
            border: 1px solid #d7e7ef;
            border-radius: 18px;
            background: linear-gradient(115deg, #ffffff 0%, #f0faff 58%, #e8f7f7 100%);
            box-shadow: 0 5px 18px rgba(24, 52, 78, 0.055);
        }
        .bci-hero:after {
            content: "";
            position: absolute;
            width: 230px;
            height: 230px;
            right: -75px;
            top: -125px;
            border: 1px solid rgba(14,159,154,0.18);
            border-radius: 50%;
            box-shadow: 0 0 0 22px rgba(14,159,154,0.035),
                        0 0 0 48px rgba(14,159,154,0.025);
        }
        .bci-kicker {
            color: #087f86;
            text-transform: uppercase;
            letter-spacing: 0.13em;
            font-size: 0.72rem;
            font-weight: 750;
            margin-bottom: 0.4rem;
        }
        .bci-title {
            color: #14243a;
            font-size: clamp(1.8rem, 3vw, 2.5rem);
            font-weight: 800;
            letter-spacing: -0.045em;
            line-height: 1.1;
            margin: 0;
        }
        .bci-subtitle {
            color: #52647a;
            font-size: 0.98rem;
            margin-top: 0.65rem;
            max-width: 850px;
        }
        .bci-status {
            display: inline-block;
            margin-top: 0.9rem;
            padding: 0.28rem 0.65rem;
            border: 1px solid #c6e8e4;
            border-radius: 999px;
            background: #effbf9;
            color: #087f75;
            font-size: 0.75rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render():
    """Draw the full module. Safe to call from a multi-page course app."""
    _init_state()
    _apply_theme()
    st.markdown(
        """
        <section class="bci-hero">
          <div class="bci-kicker">SYNAPSE · BRAIN–COMPUTER INTERFACE</div>
          <h1 class="bci-title">🧠 EEG Spatial Analysis</h1>
          <div class="bci-subtitle">
            Explore how brain signals are distributed across the scalp.
            Compare referencing methods, frequency bands and spatial filters.
          </div>
          <span class="bci-status">● EEG analysis workspace</span>
        </section>
        """,
        unsafe_allow_html=True,
    )
    epochs, key, params = _sidebar()
    if epochs is None:
        st.info("Choose or upload a dataset in the sidebar to begin.")
        return
    classes = list(epochs.event_id)
    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; gap:0.55rem; margin:0.25rem 0 1.15rem 0;">
          <span style="background:#e7f3f8;color:#31566b;padding:0.38rem 0.7rem;border-radius:8px;font-size:0.82rem;">
            <b>{len(epochs)}</b> trials
          </span>
          <span style="background:#e7f3f8;color:#31566b;padding:0.38rem 0.7rem;border-radius:8px;font-size:0.82rem;">
            <b>{len(epochs.ch_names)}</b> EEG channels
          </span>
          <span style="background:#e7f3f8;color:#31566b;padding:0.38rem 0.7rem;border-radius:8px;font-size:0.82rem;">
            <b>{epochs.info['sfreq']:g} Hz</b> sampling rate
          </span>
          <span style="background:#eaf8f5;color:#087f75;padding:0.38rem 0.7rem;border-radius:8px;font-size:0.82rem;">
            {classes[0]} vs {classes[1]}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ctx: dict = {}
    t1, t2, t3, t4 = st.tabs(["① Topography & signals", "② CSP spatial filters",
                              "③ Explain & quiz", "④ Export"])
    with t1:
        _topography_tab(epochs, key, params, ctx)
    with t2:
        _csp_tab(epochs, key, params, ctx)
    with t3:
        _explain_tab(params, ctx)
    with t4:
        _export_tab(params)

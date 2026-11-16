# Developer guide: Spatial Analysis module

## 1. Setup

```bash
git clone https://github.com/krishna385/BCI-project.git && cd BCI-project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # all tests should pass in about 15 s
streamlit run app.py        # stand-alone module
```

Conda users: `conda env create -f environment.yml && conda activate bci-spatial`.

## 2. Architecture

```
               ┌──────────── spatial_module/ui.py (Streamlit only) ────────────┐
 sidebar  ──►  │ _sidebar() → epochs + params dict                             │
               │ cached steps: _referenced → _banded → _csp                    │
               │ tabs: _topography_tab, _csp_tab, _explain_tab, _export_tab    │
               └──────┬───────────────┬────────────────┬───────────────┬───────┘
                      │               │                │               │
                 data.py        processing.py       plots.py      explain.py / export.py
             (MNE objects)    (pure NumPy + MNE)   (Plotly figs)   (text, JSON, script)
```

* **`processing.py` never imports Streamlit.** Every function takes NumPy arrays
  shaped `(trials, channels, times)`. That keeps it testable and reusable, and the
  exported script calls these same functions.
* **`ui.py` is the only Streamlit file.** Heavy steps are cached with
  `st.cache_resource` (MNE objects) and `st.cache_data` (arrays). Arguments with a
  leading underscore (`_ep`) are not hashed. The `key` string identifies the dataset instead.
* **Linked plots:** the topomap uses `st.plotly_chart(on_select="rerun")`. A click
  writes the electrode name (from `customdata`) to `st.session_state.sel_ch`, which
  drives the channel view and spectrum. The "Channel" dropdown uses the same key, so
  both stay in sync.

## 3. Data flow for one parameter change

1. `_sidebar()` reads the widgets and returns `epochs`, `key`, `params`.
2. `_referenced(key, ref, …)` → `processing.apply_reference` (cached per reference).
3. `_banded(…)` → zero-phase 4th-order Butterworth band-pass (cached per band).
4. `topo_metric` → values per channel → `plots.topomap_figure`.
5. `_csp(…)` → crop to window → drop flat channels → `csp_cross_val` (CSP inside a
   `Pipeline`, 5-fold stratified CV, seed 42) → `fit_csp` on all trials, used only to
   *visualise* patterns.

## 4. Design decisions

| Decision | Reason |
|---|---|
| Re-reference **before** band-pass | Both are linear, so the order doesn't change the result, but referencing first lets the trial plot show broadband before/after |
| IIR (Butterworth) filter rather than FIR | Epochs are 5 s long. A long FIR filter for low cut-offs would not fit, and IIR is fast |
| Topomaps drawn in Plotly, not `mne.viz.plot_topomap` | Needed to be clickable. Positions use an azimuthal-equidistant projection of a least-squares sphere fit, as MNE does |
| `component_order="alternate"` in CSP | Odd components favour class A and even ones class B, which makes the pattern titles meaningful |
| Tiny 1e-6 shrinkage when "None" is chosen | After CAR, Laplacian or CSD the covariance is rank-deficient and the eigen-solver can fail. The loading has no visible effect |
| Data rescaled before CSP | CSP is scale-invariant, but volts² (~1e-12) and CSD units upset numerical tolerances |
| Chance level from a binomial test | 50 % is misleading for small trial counts (Combrisson & Jerbi, 2015) |

## 5. Plugging into the shared course app

The module is self-contained in `spatial_module/`. Its public hooks are:

```python
from spatial_module.ui import render, MODULE_INFO
MODULE_INFO   # {'id': 'spatial_analysis', 'title': 'Spatial analysis', 'team': 'SYNAPSE', ...}
render()      # draws the whole module in the current page
```

For a Streamlit multi-page course app (`st.navigation`):

```python
import streamlit as st
from spatial_module.ui import render as spatial_page

pages = [
    # ... other groups' modules
    st.Page(spatial_page, title="Spatial analysis", icon="🧠", url_path="spatial"),
]
st.navigation(pages).run()
```

Notes for integration:

* `render()` does **not** call `st.set_page_config`. The host app should call it once.
* All session-state keys are module-specific (`ref`, `band`, `sel_ch`, `quiz_*`, ...). If
  the host app's keys clash, prefix them in `DEFAULTS` and the widget `key=` arguments.
* Add `spatial_module/`'s requirements to the shared `requirements.txt` (they are standard).

## 6. Extending

* **New reference scheme:** add a function in `processing.py`, add its name to
  `REFERENCES`, and add a branch in `apply_reference`. It then appears in the sidebar,
  the comparison chart and the tests (`test_every_reference_runs_through_pipeline` is
  parametrised over `REFERENCES`).
* **New quiz question:** append a dict to `explain.QUESTIONS`.
* **New topomap metric:** add a branch in `topo_metric` and an entry in `ui.METRICS`.

## 7. Tests

| File | What it covers |
|---|---|
| `tests/test_processing.py` | CAR, single reference, Hjorth, CSD, band-pass, contralateral ERD in the topomap, CSP accuracy above chance, CSP component order, every reference through the full pipeline, chance level, electrode projection |
| `tests/test_app_and_export.py` | The app starts without errors, reacts to a reference change, the quiz gives feedback, JSON round trip, the exported script runs end-to-end, upload of raw / montage-less / epoched `.fif` files |

`pytest` runs them all. The app tests use Streamlit's `AppTest`, so no browser is needed.

## 8. Reproducibility

* Simulation: `numpy.random.default_rng(seed)`, seed shown in the UI and exported.
* Cross-validation: `StratifiedKFold(shuffle=True, random_state=42)`.
* Versions: see `requirements.txt` (tested with MNE 1.13, Streamlit 1.67, scikit-learn 1.9).

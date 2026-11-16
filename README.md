# 🧠 Spatial Analysis: BCI Teaching Module

**Course:** Brain–Computer Interfaces (CS5109), IIITDM Kancheepuram · Option A, Web-App Module  
**Team SYNAPSE:** Skanda S Bhat (CS22B1067) · Krishan Kant Mishra (CS22B1031)  
**Repository:** [BCI-EEG-Spatial-Analysis](https://github.com/Skanda0611/BCI-EEG-Spatial-Analysis)

An interactive Streamlit module that teaches **spatial analysis of EEG**: why the reference matters, what a scalp topography shows, how Laplacian filters sharpen local activity, and how **Common Spatial Patterns (CSP)** learn the best channel combination for a motor-imagery BCI.

![The module: click an electrode on the topomap and the signal views follow](docs/images/02_car_topography.png)

---

## Run It

### Prerequisites

- Python 3.10–3.14
- Git
- Internet connection for cloning the repository and downloading dependencies

The built-in synthetic motor-imagery dataset is generated locally. No EEG dataset download is required to start the application.

### 1. Clone the repository

```bash
git clone https://github.com/Skanda0611/BCI-EEG-Spatial-Analysis.git
cd BCI-EEG-Spatial-Analysis
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install application dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Open the application in your browser:

[http://localhost:8501](http://localhost:8501)

### 5. Run automated tests

Install the development and testing dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run the test suite:

```bash
python -m pytest -v
```

The tests cover the spatial module's processing, application, upload and export functionality.

---

## What the Module Covers

| Requirement from the brief | Where it is in the app |
|---|---|
| Sample data and own-file upload | Sidebar → **1 · Data**: built-in simulated motor imagery, PhysioNet EEGBCI (download), or upload `.fif / .edf / .bdf / .gdf / .set` |
| 5+ adjustable parameters | Reference scheme, reference electrode, Laplacian neighbours, frequency band, time window, topomap metric, CSP channels, CSP components, regularisation, shrinkage, example trial and simulation settings |
| 2 linked, live-updating plots | **Topomap ↔ channel view**: clicking an electrode updates the trial trace, ERD time course and spectrum |
| Explain panel with questions | Tab **③ Explain & quiz**: setting-dependent notes, concept cards and six self-check questions with feedback |
| Export of code and parameters | Tab **④ Export**: parameters as JSON (and re-import), plus a runnable Python script |
| One-line application command | `streamlit run app.py` after dependency installation |
| At least 3 automated tests | Pytest suite in `tests/` |

---

## Repository Layout

```text
BCI-EEG-Spatial-Analysis/
│
├── app.py                      Stand-alone entry point
├── requirements.txt            Application dependencies
├── requirements-dev.txt        Development and testing dependencies
├── README.md                   Project documentation
│
├── spatial_module/
│   ├── ui.py                    Streamlit page and render() entry point
│   ├── data.py                  Synthetic data, PhysioNet loader,
│   │                            file upload and epoching
│   ├── processing.py            References, Laplacians, CSD,
│   │                            filtering, band power and CSP
│   ├── plots.py                 Plotly topomaps and linked signal plots
│   ├── explain.py               Explanation panel and quiz questions
│   └── export.py                JSON and Python-script export
│
├── tests/                       Automated pytest suite
├── docs/
│   ├── USER_GUIDE.md             Instructions and suggested experiments
│   └── DEVELOPER_GUIDE.md        Architecture and extension guide
│
└── .gitignore
```

---

## Data

### 1. Synthetic Motor Imagery

The default dataset is generated locally and contains:

- 36 electrodes
- 160 Hz sampling frequency
- Left-hand versus right-hand motor imagery
- Mu-rhythm sources under C3 and C4
- Contralateral event-related desynchronization (ERD) between 0.5 and 3 seconds after the cue

The recording also includes a noisy reference shared by all channels and a strong occipital alpha source, allowing users to observe the effects of different reference schemes.

The known ground truth makes this dataset useful for teaching and experimentation.

### 2. PhysioNet EEG Motor Movement/Imagery

The application supports PhysioNet EEG Motor Movement/Imagery data (Schalk et al., 2004; Goldberger et al., 2000), including runs 4, 8 and 12.

Data is downloaded through `mne.datasets.eegbci` on first use. An internet connection is required for the initial download.

### 3. Your Own EEG File

The application supports continuous EEG recordings with two event types or an epoched `-epo.fif` file.

Supported formats:

- `.edf`
- `.bdf`
- `.gdf`
- `.set`
- `.fif`

Channels are matched to standard 10-20 electrode positions when the recording does not contain a montage.

To generate the synthetic sample file manually:

```bash
python -m spatial_module.data
```

This writes:

```text
sample_data/synthetic_mi_raw.fif
```

---

## References

1. Hjorth, B. (1975). An on-line transformation of EEG scalp potentials into orthogonal source derivations. *EEG and Clinical Neurophysiology*, 39(5), 526–530.
2. McFarland, D. J., et al. (1997). Spatial filter selection for EEG-based communication. *EEG and Clinical Neurophysiology*, 103(3), 386–394.
3. Perrin, F., et al. (1989). Spherical splines for scalp potential and current density mapping. *EEG and Clinical Neurophysiology*, 72(2), 184–187.
4. Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization. *Clinical Neurophysiology*, 110(11), 1842–1857.
5. Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. *IEEE Transactions on Rehabilitation Engineering*, 8(4), 441–446.
6. Blankertz, B., et al. (2008). Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Signal Processing Magazine*, 25(1), 41–56.
7. Haufe, S., et al. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. *NeuroImage*, 87, 96–110.
8. Combrisson, E., & Jerbi, K. (2015). Exceeding chance level by chance. *Journal of Neuroscience Methods*, 250, 126–136.
9. Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.
10. Schalk, G., et al. (2004). BCI2000: a general-purpose brain-computer interface system. *IEEE Transactions on Biomedical Engineering*, 51(6), 1034–1043.

---

## Tools and AI Use

Built with MNE-Python, scikit-learn, SciPy, Plotly and Streamlit.

An AI assistant (Claude, by Anthropic) helped draft parts of the code and documentation. The team reviewed, tested and is responsible for the project.


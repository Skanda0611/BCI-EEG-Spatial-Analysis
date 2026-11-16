"""Text for the "Explain" panel: concept notes, notes that react to the
current settings, and self-check questions."""

from __future__ import annotations

CONCEPTS = {
    "Why spatial analysis?": """
Every EEG electrode records a **voltage difference** between itself and a
reference. It never sees one brain area on its own. Activity from a source
spreads through the skull and scalp (*volume conduction*), so neighbouring
electrodes pick up much the same thing. Spatial analysis asks two questions:

1. **Where on the scalp** is a signal strongest? (topomaps)
2. **How can we combine channels** so the signal we care about stands out?
   (re-referencing, Laplacians, CSP)
""",
    "Reference schemes": """
| Scheme | What it does | Good for | Watch out for |
|---|---|---|---|
| **As recorded** | Keeps the hardware reference (often an ear or mastoid) | Nothing to compute | Any noise on the reference shows up on *every* channel |
| **Common average (CAR)** | Subtracts the mean of all channels at each time point | Removing anything shared by all channels | Needs good scalp coverage. The result depends on which channels you include |
| **Single electrode** | Subtracts one chosen channel (e.g. Cz) | Matching an older study's setup | That electrode becomes flat zero, and activity under it leaks into every channel |
| **Hjorth Laplacian** | Each channel minus the mean of its nearest neighbours | Sharpening focal sources such as the motor mu rhythm | Edge channels have one-sided neighbours |
| **Surface Laplacian (CSD)** | Spherical-spline estimate of current flowing through the scalp | The most "local" view, independent of the reference | Units become V/m², and it is sensitive to bad channels |
""",
    "Event-related desynchronisation (ERD)": """
When you imagine moving your **right** hand, the 8-13 Hz **mu rhythm** over the
**left** motor cortex (around **C3**) drops in power. This is event-related
desynchronisation. Imagining the left hand does the same over **C4**.
ERD % = (power during task − baseline power) / baseline power × 100. Negative
values mean desynchronisation.
""",
    "Common Spatial Patterns (CSP)": """
CSP learns spatial filters `w` so that the filtered signal `wᵀx` has **maximum
variance for one class and minimum for the other**. It solves the generalised
eigenvalue problem `Σ_A w = λ (Σ_A + Σ_B) w`.

* **Filters** are what you multiply the data by. They are hard to interpret
  because they also cancel noise.
* **Patterns** (the columns of the inverse of the filter matrix) show how each
  source projects to the scalp, so **plot patterns, not filters**.
* Log-variance of the filtered signals makes a compact feature vector for a
  linear classifier (LDA).
* CSP must be fitted **inside** cross-validation. Fitting it on all trials
  first lets test trials shape the filters and inflates accuracy.
""",
    "Chance level": """
With few trials, a classifier can reach 60 % or more by pure luck. The red
dashed line is the accuracy you need to beat chance at p < 0.05 for this
number of trials (binomial test, Combrisson & Jerbi, 2015). It is stricter
than the naive 50 %.
""",
}


def live_notes(params: dict, results: dict) -> list[str]:
    """Short sentences that react to what the user has currently selected."""
    notes = []
    ref = params["reference"]
    if ref == "As recorded":
        notes.append("You're looking at the raw reference. With the synthetic data, every "
                     "channel carries the same noisy-reference signal, so the topomap looks "
                     "washed out. Switch to *Common average* and watch the C3/C4 contrast jump.")
    elif ref == "Common average (CAR)":
        notes.append("CAR has removed everything that was identical on all channels. "
                     "In the top-right plot the grey (raw) trace carries extra shared noise "
                     "that the black trace no longer has.")
    elif ref == "Single electrode":
        notes.append(f"{params.get('ref_channel')} is now the reference, so it reads exactly "
                     "zero (look at it on the topomap). Activity under it appears with "
                     "opposite sign on every other channel.")
    elif ref == "Hjorth Laplacian":
        notes.append("The Hjorth Laplacian is a spatial high-pass filter: broad patterns "
                     "(like the occipital alpha that spreads everywhere) are suppressed, and "
                     "focal ones (mu over C3/C4) are kept.")
    else:
        notes.append("CSD estimates the current entering or leaving the scalp at each "
                     "electrode. It no longer depends on any reference. Units are V/m².")

    lo, hi = params["l_freq"], params["h_freq"]
    if hi < 8 or lo > 30:
        notes.append(f"Your band ({lo:g}-{hi:g} Hz) is outside mu/beta, where motor imagery "
                     "effects live. Expect the class difference to fade towards zero.")
    if params["tmin"] < 0.3:
        notes.append("Your time window starts before ERD has developed (it takes a few "
                     "hundred ms after the cue). Try starting at about 0.5 s.")

    acc = results.get("csp_acc")
    chance = results.get("chance")
    if acc is not None and chance is not None:
        if acc < chance:
            notes.append(f"CSP accuracy ({acc:.0%}) is **not** above the chance level "
                         f"({chance:.0%}) for {results.get('n_trials')} trials. Don't claim "
                         "the classifier works.")
        else:
            notes.append(f"CSP accuracy ({acc:.0%}) beats the chance level ({chance:.0%}).")
    chan = results.get("chan_acc")
    if chan is not None and acc is not None and abs(chan - acc) > 0.05:
        better = "CSP" if acc > chan else "the two-channel (C3+C4) feature"
        notes.append(f"Here {better} does better. CSP learns its own spatial filter, so "
                     "it hardly cares which reference you chose, while plain channel power does.")
    return notes


QUESTIONS = [
    dict(q="With the synthetic data and *As recorded*, which change makes the C3/C4 "
           "difference on the topomap much clearer?",
         options=["Switching to Common average (CAR)", "Narrowing the time window to 0-0.2 s",
                  "Changing the band to 1-4 Hz", "Lowering the number of CSP components"],
         answer=0,
         why="The simulated recording has a noisy reference that adds the same signal to every "
             "channel. CAR subtracts the mean across channels, which removes that shared part exactly."),
    dict(q="A participant imagines moving their RIGHT hand. Where do you expect mu-band ERD?",
         options=["Over C4 (right hemisphere)", "Over C3 (left hemisphere)",
                  "Over Oz", "Equally everywhere"],
         answer=1,
         why="Motor cortex is organised contralaterally, so right-hand imagery desynchronises "
             "the left motor cortex, around C3."),
    dict(q="After choosing *Single electrode* with Cz, what happens to Cz itself?",
         options=["It shows the strongest ERD", "It becomes exactly zero",
                  "It is replaced by the average of all channels", "Nothing changes"],
         answer=1,
         why="Every channel has Cz subtracted, including Cz, so Cz - Cz = 0. That is why a "
             "reference electrode should sit away from the activity you care about."),
    dict(q="Why does the app plot CSP *patterns* rather than CSP *filters*?",
         options=["Patterns are faster to compute",
                  "Filters also cancel noise, so their weights don't show where the source is",
                  "Filters only exist for two channels", "There is no difference"],
         answer=1,
         why="A filter can put large weights on noisy channels to cancel noise. The pattern "
             "(forward model) shows how the extracted source actually projects onto the scalp "
             "(Haufe et al., 2014)."),
    dict(q="Which evaluation is leak-free?",
         options=["Fit CSP on all trials, then cross-validate LDA on the CSP features",
                  "Put CSP and LDA in one pipeline and cross-validate the whole pipeline",
                  "Train and test on the same trials", "Pick the best fold and report it"],
         answer=1,
         why="CSP uses the labels. If it sees test trials during fitting, information "
             "leaks and the score is optimistic. The app always cross-validates CSP + LDA together."),
    dict(q="The Hjorth Laplacian behaves most like which kind of filter?",
         options=["Spatial low-pass", "Spatial high-pass", "Temporal notch filter",
                  "Temporal low-pass"],
         answer=1,
         why="Subtracting the neighbourhood average removes what is smooth across the scalp "
             "(broad, distant sources) and keeps what changes sharply from one electrode to "
             "the next (local sources)."),
]

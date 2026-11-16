"""Plotly figures: scalp topomaps, signal traces, ERD time courses, spectra
and CSP views. Topomaps are drawn by hand (interpolated heatmap + head
outline) so they stay interactive: clicking an electrode selects it."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RBFInterpolator
from scipy.signal import welch

CLASS_COLORS = ("#2563eb", "#dc2626")  # blue = class A, red = class B
RAW_COLOR = "#9ca3af"


# --------------------------------------------------------------------------
# Electrode projection
# --------------------------------------------------------------------------

def fit_sphere(xyz: np.ndarray):
    """Least-squares sphere through the electrode positions."""
    A = np.c_[2 * xyz, np.ones(len(xyz))]
    b = np.sum(xyz ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:3]
    radius = np.sqrt(sol[3] + center @ center)
    return center, radius


def project_2d(xyz: np.ndarray) -> np.ndarray:
    """Azimuthal equidistant projection, the same idea MNE uses: the vertex
    (Cz) maps to the centre, the head's equator (Fpz-T7-Oz-T8) to radius 1."""
    center, _ = fit_sphere(xyz)
    v = xyz - center
    r = np.linalg.norm(v, axis=1)
    theta = np.arccos(np.clip(v[:, 2] / r, -1, 1))  # angle from the vertex
    phi = np.arctan2(v[:, 1], v[:, 0])
    rho = theta / (np.pi / 2)
    return np.c_[rho * np.cos(phi), rho * np.sin(phi)]


# --------------------------------------------------------------------------
# Topomap
# --------------------------------------------------------------------------

def _interp_grid(xy, values, res=121):
    """Interpolate electrode values onto a disc covering every electrode."""
    lim = max(1.0, float(np.hypot(xy[:, 0], xy[:, 1]).max()) + 0.03)
    g = np.linspace(-lim, lim, res)
    gx, gy = np.meshgrid(g, g)
    pts = np.c_[gx.ravel(), gy.ravel()]
    rbf = RBFInterpolator(xy, values, kernel="thin_plate_spline", smoothing=1e-3)
    z = rbf(pts).reshape(res, res)
    z[np.hypot(gx, gy) > lim] = np.nan
    return g, z, lim


def _head_shapes(lim, xref="x", yref="y"):
    line = dict(color="#374151", width=2)
    r = 1.0
    shapes = [dict(type="circle", x0=-r, y0=-r, x1=r, y1=r, line=line,
                   xref=xref, yref=yref)]
    nose = dict(type="path", path=f"M -0.12 {r*0.99} L 0 {r*1.13} L 0.12 {r*0.99}",
                line=line, xref=xref, yref=yref)
    left_ear = dict(type="path",
                    path=f"M {-r} 0.12 Q {-r-0.1} 0.1 {-r-0.08} 0 Q {-r-0.1} -0.1 {-r} -0.12",
                    line=line, xref=xref, yref=yref)
    right_ear = dict(type="path",
                     path=f"M {r} 0.12 Q {r+0.1} 0.1 {r+0.08} 0 Q {r+0.1} -0.1 {r} -0.12",
                     line=line, xref=xref, yref=yref)
    return shapes + [nose, left_ear, right_ear]


def _symmetric_range(values):
    m = float(np.nanmax(np.abs(values))) or 1.0
    return -m, m


def add_topomap(fig, xy, values, names, row=None, col=None, selected=None,
                colorscale="RdBu_r", symmetric=True, unit="", show_names=True,
                showscale=True, colorbar_x=None):
    """Add interpolated map + electrode markers to ``fig`` (optionally a subplot)."""
    g, z, lim = _interp_grid(xy, values)
    zmin, zmax = _symmetric_range(values) if symmetric else (np.nanmin(values), np.nanmax(values))
    kw = dict(row=row, col=col) if row else {}
    cb = dict(title=dict(text=unit, side="right"), thickness=12, len=0.8)
    if colorbar_x is not None:
        cb["x"] = colorbar_x
    fig.add_trace(go.Heatmap(x=g, y=g, z=z, colorscale=colorscale, zmin=zmin, zmax=zmax,
                             zsmooth="best", hoverinfo="skip", showscale=showscale,
                             colorbar=cb, showlegend=False), **kw)
    fig.add_trace(go.Contour(x=g, y=g, z=z, showscale=False, hoverinfo="skip",
                             contours=dict(coloring="none"), ncontours=8,
                             line=dict(color="rgba(0,0,0,0.25)", width=1),
                             showlegend=False), **kw)
    sizes = [14 if n == selected else 7 for n in names]
    colors = ["#facc15" if n == selected else "#111827" for n in names]
    fig.add_trace(go.Scatter(
        x=xy[:, 0], y=xy[:, 1], mode="markers+text" if show_names else "markers",
        text=names if show_names else None, textposition="top center",
        textfont=dict(size=9, color="#111827"),
        marker=dict(size=sizes, color=colors, line=dict(color="white", width=1)),
        customdata=names,
        hovertemplate=[f"<b>{n}</b><br>{v:.2f} {unit}<extra></extra>"
                       for n, v in zip(names, values)],
        showlegend=False), **kw)
    return lim


def topomap_figure(xy, values, names, title, unit="", selected=None,
                   symmetric=True, height=470):
    fig = go.Figure()
    lim = add_topomap(fig, xy, values, names, selected=selected, unit=unit,
                      symmetric=symmetric)
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15)), height=height,
        margin=dict(l=10, r=10, t=45, b=10), shapes=_head_shapes(lim),
        xaxis=dict(visible=False, range=[-lim - 0.15, lim + 0.15], scaleanchor="y"),
        yaxis=dict(visible=False, range=[-lim - 0.1, lim + 0.2]),
        plot_bgcolor="white")
    return fig


def topomap_grid(xy, maps, names, titles, unit="a.u.", cols=4, height=260):
    """Several small topomaps side by side (used for CSP patterns)."""
    n = len(maps)
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.03, vertical_spacing=0.08)
    shapes = []
    lim = 1.0
    for i, m in enumerate(maps):
        r, c = i // cols + 1, i % cols + 1
        lim = add_topomap(fig, xy, m, names, row=r, col=c, unit=unit,
                          show_names=False, showscale=False)
        ax = i + 1
        xref = "x" if ax == 1 else f"x{ax}"
        yref = "y" if ax == 1 else f"y{ax}"
        shapes += _head_shapes(lim, xref, yref)
        fig.update_xaxes(visible=False, range=[-lim - 0.15, lim + 0.15], row=r, col=c,
                         scaleanchor=yref)
        fig.update_yaxes(visible=False, range=[-lim - 0.1, lim + 0.2], row=r, col=c)
    fig.update_layout(shapes=shapes, height=height * rows, plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    fig.update_annotations(font_size=12)
    return fig


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

def signal_figure(times, before, after, env_times, env_a, env_b, class_names,
                  channel, window, unit_after="µV", ref_label="", height=470):
    """Two linked rows for one channel:
    1) one trial before vs after the chosen reference (broadband),
    2) class-average band-power envelope (the ERD time course)."""
    top = (f"{channel}: one trial, as recorded" if ref_label == "As recorded"
           else f"{channel}: one trial, as recorded (grey) vs {ref_label} (black)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=(top,
                                        f"{channel}: band power over time (mean of trials)"))
    fig.add_trace(go.Scatter(x=times, y=before, name="As recorded",
                             line=dict(color=RAW_COLOR, width=1)), row=1, col=1)
    if ref_label != "As recorded":
        fig.add_trace(go.Scatter(x=times, y=after, name=ref_label,
                                 line=dict(color="#111827", width=1.2)), row=1, col=1)
    for env, name, color in zip((env_a, env_b), class_names, CLASS_COLORS):
        fig.add_trace(go.Scatter(x=env_times, y=env, name=name,
                                 line=dict(color=color, width=2)), row=2, col=1)
    for r in (1, 2):
        fig.add_vrect(x0=window[0], x1=window[1], fillcolor="#fde68a", opacity=0.35,
                      line_width=0, row=r, col=1)
        fig.add_vline(x=0, line=dict(color="#6b7280", dash="dot"), row=r, col=1)
    fig.update_yaxes(title_text=unit_after, row=1, col=1)
    fig.update_yaxes(title_text="% of baseline" if env_times[0] < 0 else "power", row=2, col=1)
    fig.update_xaxes(title_text="Time from cue (s)", row=2, col=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=45, b=10),
                      legend=dict(orientation="h", y=-0.18), plot_bgcolor="white")
    return fig


def spectrum_figure(X_a, X_b, sfreq, class_names, channel, band, height=300):
    """Power spectrum of one channel per class during the analysis window."""
    nper = min(X_a.shape[-1], int(sfreq))
    fig = go.Figure()
    for Xc, name, color in zip((X_a, X_b), class_names, CLASS_COLORS):
        f, p = welch(Xc, fs=sfreq, nperseg=nper, axis=-1)
        fig.add_trace(go.Scatter(x=f, y=10 * np.log10(p.mean(axis=0) + 1e-30),
                                 name=name, line=dict(color=color, width=2)))
    fig.add_vrect(x0=band[0], x1=band[1], fillcolor="#fde68a", opacity=0.4, line_width=0,
                  annotation_text="selected band", annotation_position="top left")
    fig.update_layout(title=dict(text=f"{channel}: spectrum inside the time window "
                                 "(after re-referencing, before band-pass)",
                                 font=dict(size=14)),
                      xaxis=dict(title="Frequency (Hz)", range=[1, min(45, sfreq / 2)]),
                      yaxis=dict(title="Power (dB)"), height=height,
                      margin=dict(l=10, r=10, t=45, b=10), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.12, x=0.6))
    return fig


def csp_scatter(feats, y, class_names, height=380):
    fig = go.Figure()
    for k, (name, color) in enumerate(zip(class_names, CLASS_COLORS)):
        m = y == k
        fig.add_trace(go.Scatter(x=feats[m, 0], y=feats[m, 1], mode="markers", name=name,
                                 marker=dict(color=color, size=8, opacity=0.75)))
    fig.update_layout(title=dict(text="Each dot is one trial: log-variance of CSP 1 vs CSP 2",
                                 font=dict(size=14)),
                      xaxis_title="CSP 1 (log variance)", yaxis_title="CSP 2 (log variance)",
                      height=height, margin=dict(l=10, r=10, t=45, b=10), plot_bgcolor="white")
    return fig


def accuracy_bar(labels, csp_acc, chan_acc, chance, height=360):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=chan_acc, name="C3 + C4 band power", marker_color="#9ca3af"))
    fig.add_trace(go.Bar(x=labels, y=csp_acc, name="CSP + LDA", marker_color="#2563eb"))
    fig.add_hline(y=chance, line=dict(color="#dc2626", dash="dash"),
                  annotation_text=f"chance (p<0.05) = {chance:.2f}")
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1], title="CV accuracy"),
                      height=height, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12), plot_bgcolor="white")
    return fig

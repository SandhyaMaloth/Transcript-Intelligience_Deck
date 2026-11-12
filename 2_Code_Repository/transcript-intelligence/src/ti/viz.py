"""Static matplotlib charts used by the notebook and the slide deck.

No backend is forced here, so plots render inline in Jupyter; scripts that
only save files (run_pipeline.py, build_notebook.py) select Agg themselves.

Colour follows the entity everywhere: external = blue, support = orange,
internal = green. Red is reserved for risk.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import LAUNCH_DAY, OUTAGE_DAY
from .taxonomy import CATEGORIES

TYPE_COLORS = {"external": "#2a78d6", "support": "#eb6834", "internal": "#1baf7a"}
TYPE_LABELS = {"external": "External (account mgmt)", "support": "Support", "internal": "Internal"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
RISK = "#e34948"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)


def _save(fig, out: Path | None):
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def categories_by_type(calls: pd.DataFrame, out=None):
    ct = pd.crosstab(calls.category, calls.call_type).reindex(columns=["external", "support", "internal"], fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    left = pd.Series(0, index=ct.index)
    for t in ct.columns:
        ax.barh(ct.index, ct[t], left=left, color=TYPE_COLORS[t], label=TYPE_LABELS[t],
                height=0.62, edgecolor=SURFACE, linewidth=2)
        left += ct[t]
    for i, v in enumerate(left):
        ax.text(v + 0.3, i, str(int(v)), va="center", color=INK2, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Calls (primary category)")
    ax.legend(loc="lower right")
    ax.set_title("What 100 calls are about — primary topic by call type")
    return _save(fig, out)


def sentiment_by_phase(calls: pd.DataFrame, out=None):
    p = calls.groupby(["phase", "call_type"])["sentiment_score"].mean().unstack()
    n = calls.groupby(["phase", "call_type"]).size().unstack()
    labels = [x.split(" ", 1)[1] for x in p.index]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for t in ["external", "internal", "support"]:
        ax.plot(labels, p[t], marker="o", markersize=8, lw=2, color=TYPE_COLORS[t], label=TYPE_LABELS[t],
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{p[t].iloc[-1]:.1f}", (len(labels) - 1, p[t].iloc[-1]), xytext=(10, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=10)
    ax.set_ylim(1.5, 4.7)
    ax.set_ylabel("Avg call sentiment (1–5)")
    ax.legend(loc="lower left", ncol=3)
    ax.set_title("Sentiment fell together, recovered apart — support never bounced back")
    counts = ", ".join(f"{lab}: {int(n.loc[i].sum())}" for i, lab in zip(p.index, labels, strict=True))
    ax.text(1.0, -0.16, "n per phase: " + counts, color=MUTED, fontsize=9, transform=ax.transAxes, ha="right")
    return _save(fig, out)


def weekly_trend(calls: pd.DataFrame, out=None):
    w = calls.groupby(["week", "call_type"])["sentiment_score"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for t in ["external", "internal", "support"]:
        s = w[t].dropna()
        ax.plot(s.index, s.values, marker="o", markersize=6, lw=2, color=TYPE_COLORS[t], label=TYPE_LABELS[t])
    for day, label in [(OUTAGE_DAY, "Detect outage"), (LAUNCH_DAY, "Comply v2 GA")]:
        day = day.tz_localize(None)
        ax.axvline(day, color=MUTED, lw=1, ls="--")
        ax.text(day, 4.95, " " + label, color=INK2, fontsize=9, va="top")
    ax.set_ylim(1, 5)
    ax.set_ylabel("Avg call sentiment (1–5)")
    ax.legend(loc="lower right", ncol=3)
    ax.set_title("Weekly sentiment by call type")
    return _save(fig, out)


def empathy_gap(calls: pd.DataFrame, out=None):
    g = calls[calls.call_type != "internal"].groupby("call_type")[["customer_net", "aegis_net"]].mean()
    fig, ax = plt.subplots(figsize=(8, 3))
    for i, (t, r) in enumerate(g.iterrows()):
        ax.plot([r.customer_net, r.aegis_net], [i, i], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter(r.customer_net, i, s=120, color=TYPE_COLORS[t], zorder=2, edgecolor=SURFACE, linewidth=2)
        ax.scatter(r.aegis_net, i, s=120, color=MUTED, zorder=2, edgecolor=SURFACE, linewidth=2)
        ax.text(r.customer_net, i + 0.22, f"customer {r.customer_net:+.2f}", ha="center", fontsize=9, color=INK2)
        ax.text(r.aegis_net, i + 0.22, f"Aegis {r.aegis_net:+.2f}", ha="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(g)), [TYPE_LABELS[t] for t in g.index])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-0.6, 0.7)
    ax.set_ylim(-0.5, len(g) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Net sentence sentiment (−1 negative … +1 positive)")
    ax.set_title("The customer's voice is far more negative than the call score suggests")
    return _save(fig, out)


def sentiment_by_category(calls: pd.DataFrame, out=None):
    s = calls.groupby("category")["sentiment_score"].agg(["mean", "size"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = [RISK if v < 3 else "#2a78d6" for v in s["mean"]]
    ax.barh(s.index, s["mean"], color=colors, height=0.6)
    for i, (v, n) in enumerate(zip(s["mean"], s["size"], strict=True)):
        ax.text(v + 0.05, i, f"{v:.1f}  (n={n})", va="center", fontsize=9, color=INK2)
    ax.axvline(3, color=MUTED, lw=1, ls="--")
    ax.set_xlim(1, 5.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Avg call sentiment (1–5); red = below neutral")
    ax.set_title("Sentiment by topic")
    return _save(fig, out)


def within_call_recovery(calls: pd.DataFrame, out=None):
    d = calls[calls.call_type != "internal"].groupby(["call_type", "primary"])[["open_net", "close_net"]].mean()
    d = d[calls[calls.call_type != "internal"].groupby(["call_type", "primary"]).size() >= 3]
    d = d.sort_values("close_net")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, ((t, _), r) in enumerate(d.iterrows()):
        ax.annotate("", xy=(r.close_net, i), xytext=(r.open_net, i),
                    arrowprops=dict(arrowstyle="-|>", color=TYPE_COLORS[t], lw=2))
        ax.scatter(r.open_net, i, s=40, color=MUTED, zorder=3)
    ax.set_yticks(range(len(d)), [f"{CATEGORIES[c]['name']} · {t}" for t, c in d.index], fontsize=9)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-1, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Customer net sentiment: first third → last third of call")
    ax.set_title("Calls de-escalate — but outage calls still end negative")
    return _save(fig, out)


def competitor_chart(cm: pd.DataFrame, out=None):
    g = cm.groupby(["competitor", "post_outage"]).meeting_id.nunique().unstack(fill_value=0)
    g = g.loc[g.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bar = dict(height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.barh(g.index, g.get(False, 0), color="#86b6ef", label="Before Mar 10", **bar)
    ax.barh(g.index, g.get(True, 0), left=g.get(False, 0), color="#184f95", label="From Mar 10", **bar)
    for i, v in enumerate(g.sum(axis=1)):
        ax.text(v + 0.2, i, str(v), va="center", color=INK2, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Calls mentioning competitor")
    ax.legend(loc="lower right")
    ax.set_title("SentinelShield: 3 calls before the outage, 19 after")
    return _save(fig, out)


def account_divergence(health: pd.DataFrame, out=None):
    h = health.dropna(subset=["avg_external", "avg_support"]).sort_values("ext_vs_support_gap")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, r in enumerate(h.itertuples()):
        ax.plot([r.avg_support, r.avg_external], [i, i], color=GRID, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter(r.avg_support, i, s=90, color=TYPE_COLORS["support"], zorder=2, edgecolor=SURFACE, linewidth=2)
        ax.scatter(r.avg_external, i, s=90, color=TYPE_COLORS["external"], zorder=2, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(len(h)), h.account, fontsize=9)
    ax.scatter([], [], color=TYPE_COLORS["external"], label="Avg on account-mgmt calls")
    ax.scatter([], [], color=TYPE_COLORS["support"], label="Avg on support calls")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(1, 5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Avg call sentiment (1–5)")
    ax.set_title("Same customer, two different stories")
    return _save(fig, out)

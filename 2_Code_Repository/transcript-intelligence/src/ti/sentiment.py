"""Stage 3 — sentiment.

The dataset ships two sentiment signals:
  * call level  : `sentimentScore` (1–5) + label, from the vendor summariser
  * sentence    : `sentimentType` (positive / neutral / negative) per utterance

Rather than replace them, we *decompose* them, because a single call score
hides the thing stakeholders actually care about:

  customer_net   mean sentence sentiment of the CUSTOMER's sentences (-1..+1)
  aegis_net      same for Aegis employees
  empathy_gap    aegis_net - customer_net (agents stay polite while the
                 customer is upset -> call-level score looks better than reality)
  open_net /     customer sentiment in the first vs last third of the call
  close_net      -> `recovery` = close - open  (did the call de-escalate?)

`validate_against_vendor` checks that the sentence aggregate agrees with the
vendor score (Spearman correlation) before the decomposition is trusted.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from .config import LAUNCH_DAY, OUTAGE_DAY, RECOVERY_START

PHASES = [
    ("1 Pre-outage", None, OUTAGE_DAY),
    ("2 Outage & fallout", OUTAGE_DAY, RECOVERY_START),
    ("3 Recovery", RECOVERY_START, LAUNCH_DAY),
    ("4 Post Comply v2", LAUNCH_DAY, None),
]


def phase_of(ts: pd.Timestamp) -> str:
    for name, lo, hi in PHASES:
        if (lo is None or ts >= lo) and (hi is None or ts < hi):
            return name
    return "?"


def call_level(meetings: pd.DataFrame, utts: pd.DataFrame) -> pd.DataFrame:
    u = utts.copy()
    n = u.groupby("meeting_id")["idx"].transform("max") + 1
    u["third"] = pd.cut(u["idx"] / n, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["open", "mid", "close"])

    side = u.pivot_table(index="meeting_id", columns="side", values="sent_val", aggfunc="mean")
    side = side.reindex(columns=["aegis", "customer"])
    side.columns = [f"{c}_net" for c in side.columns]

    cust = u[u.side == "customer"]
    traj = cust.pivot_table(index="meeting_id", columns="third", values="sent_val", aggfunc="mean", observed=False)
    traj = traj.rename(columns={"open": "open_net", "mid": "mid_net", "close": "close_net"})

    neg_share = (cust.sentiment == "negative").groupby(cust.meeting_id).mean().rename("customer_neg_share")
    words = u.pivot_table(index="meeting_id", columns="side", values="words", aggfunc="sum").fillna(0)
    parts = [side, traj, neg_share]
    if "customer" in words:
        parts.append((words["customer"] / words.sum(axis=1)).rename("customer_talk_share"))

    out = meetings.set_index("meeting_id").join(parts)
    out["all_net"] = u.groupby("meeting_id")["sent_val"].mean()
    out["empathy_gap"] = out["aegis_net"] - out["customer_net"]
    out["recovery"] = out["close_net"] - out["open_net"]
    out["phase"] = out["start"].apply(phase_of)
    return out.reset_index()


def validate_against_vendor(calls: pd.DataFrame) -> dict:
    rho, p = spearmanr(calls["all_net"], calls["sentiment_score"])
    ext = calls.dropna(subset=["customer_net"])
    rho_c, p_c = spearmanr(ext["customer_net"], ext["sentiment_score"])
    return {"rho_all_vs_vendor": rho, "p": p, "rho_customer_vs_vendor": rho_c, "p_customer": p_c}

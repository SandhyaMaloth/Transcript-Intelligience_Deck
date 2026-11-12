"""Stage 4 - insights beyond topic and sentiment.

Each function answers one stakeholder question and returns a table that can be
shown as-is:

  early_warning_signals    Eng leads   What risks were we discussing internally
                                       before customers felt them?
  say_do_gap               Eng + Sales What did we tell customers vs. what
                                       engineering said the same weeks?
  remediation_aftershocks  Eng leads   Did our fixes cause new customer issues?
  account_health           Sales / CS  Which accounts look fine to the AE but
                                       not to support (or vice versa)?
  competitor_mentions      PMM / Sales Who is coming for which module, and when?
  support_quality          Support     Is the support load our fault or the
                                       customer's, and how do cases end?
  feature_backlog          Product     What are customers asking for, by module?

The risk and assurance phrase lists are curated from this dataset's
transcripts; on new data they are a starting point, not a general detector.
"""
from __future__ import annotations

import re

import pandas as pd

from .config import OUTAGE_DAY, RECOVERY_START
from .taxonomy import COMPETITORS, tag_modules

# Detect-ingestion risk language. Kept narrow: broader terms ("bottleneck")
# also match unrelated Identity and Comply discussions.
RISK_PATTERN = re.compile(
    r"single point of failure|single processing node|no fallback|no redundancy|"
    r"redundan|failover|ingestion layer|memory ceiling",
    re.I,
)
RISK_CONTEXT = re.compile(r"ingestion|node|failover|single point", re.I)
CUSTOMER_ASSURANCE = re.compile(
    r"fix that went in|put redundancy measures in place|eliminating single points of failure|"
    r"redundant processing nodes|circuit breaker",
    re.I,
)
CHURN_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _with_call_info(utts: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    return utts.merge(calls[["meeting_id", "start", "title", "call_type", "account"]], on="meeting_id")


def early_warning_signals(utts, calls, before=OUTAGE_DAY):
    u = _with_call_info(utts, calls)
    hits = u[(u.call_type == "internal") & (u.start < before)
             & u.text.str.contains(RISK_PATTERN) & u.text.str.contains(RISK_CONTEXT)]
    hits = hits.assign(lead_days=(before - hits.start).dt.days)
    return hits[["start", "lead_days", "title", "speaker", "text"]].sort_values("start")


def say_do_gap(utts, calls, until=OUTAGE_DAY):
    """Before the outage: customer-facing resilience claims vs. internal status."""
    u = _with_call_info(utts, calls)
    said = u[(u.start < until) & (u.side == "aegis") & (u.call_type != "internal")
             & u.text.str.contains(CUSTOMER_ASSURANCE)].assign(kind="Told customer")
    internal = early_warning_signals(utts, calls, until).assign(kind="Said internally")
    cols = ["start", "kind", "title", "speaker", "text"]
    return pd.concat([said[cols], internal[cols]]).sort_values("start")


def remediation_aftershocks(calls, after=RECOVERY_START):
    """Calls after the outage window whose cause traces back to the outage fix."""
    c = calls[calls.remediation_side_effect & (calls.start >= after)]
    return c[["start", "title", "primary", "sentiment_score", "churn_risk", "rationale"]]


def account_health(calls):
    rows = []
    for acc, d in calls[calls.account.notna()].sort_values("start").groupby("account"):
        ext, sup = d[d.call_type == "external"], d[d.call_type == "support"]
        last = d.iloc[-1]
        rows.append(
            {
                "account": acc,
                "n_calls": len(d),
                "first_call": d.start.min().date(),
                "last_call": last.start.date(),
                "avg_external": ext.sentiment_score.mean(),
                "avg_support": sup.sentiment_score.mean(),
                "last_score": last.sentiment_score,
                "trend": last.sentiment_score - d.iloc[0].sentiment_score,
                "max_churn": d.churn_risk.map(CHURN_RANK).max(),
                "last_churn": last.churn_risk,
                "competitors": sorted({x for lst in d.competitors for x in lst}),
                "outage_touched": bool(d.outage_related.any()),
                "journey": " → ".join(f"{r.start:%b %d} {r.call_type[:3]} {r.sentiment_score}" for r in d.itertuples()),
            }
        )
    out = pd.DataFrame(rows)
    out["ext_vs_support_gap"] = out.avg_external - out.avg_support
    return out.sort_values(["max_churn", "last_score"], ascending=[False, True])


def competitor_mentions(utts, calls):
    u = _with_call_info(utts, calls)
    rows = []
    for comp in COMPETITORS:
        for mid, d in u[u.text.str.contains(comp, case=False)].groupby("meeting_id"):
            start = d.start.iloc[0]
            rows.append(
                {
                    "competitor": comp,
                    "meeting_id": mid,
                    "start": start,
                    "call_type": d.call_type.iloc[0],
                    "mentions": len(d),
                    "said_by_customer": int((d.side == "customer").sum()),
                    "modules": tag_modules(" ".join(d.text)),
                    "post_outage": start >= OUTAGE_DAY,
                }
            )
    cols = ["competitor", "meeting_id", "start", "call_type", "mentions", "said_by_customer", "modules", "post_outage"]
    return pd.DataFrame(rows, columns=cols)


def support_quality(calls):
    s = calls[calls.call_type == "support"]
    return {
        "n_support_calls": len(s),
        "aegis_root_cause_pct": (s.root_cause == "aegis").mean(),
        "customer_root_cause_pct": (s.root_cause == "customer").mean(),
        "resolved_on_call_pct": (s.resolution == "resolved_on_call").mean(),
        "escalated_pct": (s.resolution == "escalated").mean(),
        "comm_gap_calls": int(s.comm_gap.sum()),
        "support_delay_calls": int(s.support_delay.sum()),
        "known_issue_calls": int(s.known_issue.sum()),
        "avg_customer_net": s.customer_net.mean(),
        "avg_aegis_net": s.aegis_net.mean(),
    }


def feature_backlog(calls):
    f = calls[(calls.primary == "FEATURE") | calls.secondary.apply(lambda x: "FEATURE" in x)]
    return f[["start", "account", "call_type", "modules", "churn_risk", "rationale"]]

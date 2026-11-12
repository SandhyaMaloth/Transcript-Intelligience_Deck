"""Load raw meeting folders into two tidy tables.

Each meeting folder holds six JSON files. They are flattened into:
  * meetings   - one row per call (metadata, vendor summary, derived fields)
  * utterances - one row per transcript sentence, tagged with the speaker's side

Every speaker name is matched to an address in `allEmails`, which tags each
sentence as `aegis` (employee) or `customer`. That join is what makes
customer-voice sentiment and per-side analysis possible.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pandas as pd

from .config import INTERNAL_DOMAIN

SENTIMENT_VALUE = {"positive": 1, "neutral": 0, "negative": -1}


def _read(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def match_speaker_email(name: str, emails: list[str]) -> str | None:
    """Match a speaker name to an email address.

    Prefers addresses containing the last name plus the first name or first
    initial ('Kevin O'Brien' -> kevin.obrien@..., 'Owen Price' -> o.price@...),
    then falls back to the last name alone when exactly one address has it.
    """
    parts = [_norm(p) for p in name.split() if _norm(p)]
    if not parts:
        return None
    first, last = parts[0], parts[-1]
    locals_ = {e: _norm(e.split("@")[0]) for e in emails}
    with_last = [e for e, local in locals_.items() if last in local]
    full = [e for e in with_last if first in locals_[e] or locals_[e].startswith(first[0])]
    if len(full) == 1:
        return full[0]
    if len(with_last) == 1:
        return with_last[0]
    return None


def load_dataset(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    folders = sorted(p for p in raw_dir.iterdir() if p.is_dir() and (p / "meeting-info.json").exists())
    if not folders:
        raise FileNotFoundError(f"No meeting folders (with meeting-info.json) found in {raw_dir}")

    meetings, utts, unmatched = [], [], set()
    for folder in folders:
        info = _read(folder / "meeting-info.json")
        summ = _read(folder / "summary.json")
        trans = _read(folder / "transcript.json")["data"]
        emails = info["allEmails"]
        ext_domains = sorted({e.split("@")[1] for e in emails} - {INTERNAL_DOMAIN})

        meetings.append(
            {
                "meeting_id": info["meetingId"],
                "title": info["title"],
                "start": pd.Timestamp(info["startTime"]),
                "duration_min": info["duration"],
                "n_participants": len(emails),
                "external_domains": ext_domains,
                "account": ext_domains[0].split(".")[0] if ext_domains else None,
                "summary": summ["summary"],
                "vendor_topics": summ["topics"],
                "action_items": summ["actionItems"],
                "key_moments": summ["keyMoments"],
                "overall_sentiment": summ["overallSentiment"],
                "sentiment_score": summ["sentimentScore"],
            }
        )
        for u in trans:
            email = match_speaker_email(u["speaker_name"], emails)
            if email:
                side = "aegis" if email.endswith("@" + INTERNAL_DOMAIN) else "customer"
            else:
                # Unknown speaker: on an internal-only call everyone is Aegis.
                unmatched.add((info["meetingId"], u["speaker_name"]))
                side = "customer" if ext_domains else "aegis"
            utts.append(
                {
                    "meeting_id": info["meetingId"],
                    "idx": u["index"],
                    "speaker": u["speaker_name"],
                    "email": email,
                    "side": side,
                    "text": u["sentence"],
                    "sentiment": u["sentimentType"],
                    "t_start": u["time"],
                    "t_end": u["endTime"],
                    "confidence": u["averageConfidence"],
                }
            )
    if unmatched:
        warnings.warn(f"{len(unmatched)} speaker(s) could not be matched to an email: {sorted(unmatched)[:5]}",
                      stacklevel=2)

    m = pd.DataFrame(meetings).sort_values("start").reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Converting to PeriodArray")
        m["week"] = m["start"].dt.tz_convert(None).dt.to_period("W-SUN").dt.start_time
    u = pd.DataFrame(utts)
    u["sent_val"] = u["sentiment"].map(SENTIMENT_VALUE)
    u["words"] = u["text"].str.split().str.len()
    return m, u

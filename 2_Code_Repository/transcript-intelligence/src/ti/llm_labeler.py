"""Stage 2 - topic and judgement labels (LLM, cached, with a keyword fallback).

Why an LLM for topics:
  * Calls are multi-intent: a renewal call can be 70% outage fallout. The LLM
    labels the purpose (RENEWAL, secondary OUTAGE) and gives a one-line,
    auditable rationale.
  * Judgement fields (churn risk, known bug, communication gap, resolution)
    need reading comprehension that keywords cannot provide.

Labels are cached in `labels/llm_labels.json`, keyed by meeting id, so runs
are offline and reproducible, and only new calls ever need the API.

How a call gets its label (`load_labels`):
  1. cached label                 -> label_source = "llm"
  2. else, if use_api=True        -> labelled through the API, added to the cache
  3. else                         -> keyword baseline, label_source = "keyword";
                                     judgement fields are marked "unknown"/False

Provenance of the shipped cache: the 100 labels were produced by Claude in an
interactive session applying `prompts/classify_call.md` to each call, not by
the API path below. `python refresh_labels.py --all` regenerates them through
the API from the full transcripts.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

from .config import LABEL_CACHE, PROMPT_FILE
from .taxonomy import CATEGORIES

MODEL = os.environ.get("TI_MODEL", "claude-opus-5-5")

CODES = sorted(CATEGORIES)
ENUMS = {
    "primary": CODES,
    "churn_risk": ["none", "low", "medium", "high"],
    "root_cause": ["aegis", "customer", "third_party", "n/a"],
    "resolution": ["resolved_on_call", "workaround_provided", "escalated", "feature_request_logged",
                   "info_provided", "n/a"],
}
BOOL_FIELDS = ["outage_related", "comm_gap", "support_delay", "known_issue", "remediation_side_effect"]

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        **{f: {"type": "string", "enum": v} for f, v in ENUMS.items()},
        "secondary": {"type": "array", "items": {"type": "string", "enum": CODES}},
        **{b: {"type": "boolean"} for b in BOOL_FIELDS},
        "rationale": {"type": "string"},
    },
    "required": [*ENUMS, "secondary", *BOOL_FIELDS, "rationale"],
    "additionalProperties": False,
}


def validate_label(mid: str, lab: dict) -> None:
    """Raise ValueError if a label breaks the schema or the prompt's rules."""
    for field, allowed in ENUMS.items():
        if lab.get(field) not in allowed:
            raise ValueError(f"{mid}: {field}={lab.get(field)!r} not in {allowed}")
    sec = lab.get("secondary")
    if not isinstance(sec, list) or len(sec) > 2 or not set(sec) <= set(CODES) or lab["primary"] in sec:
        raise ValueError(f"{mid}: invalid secondary {sec!r}")
    for b in BOOL_FIELDS:
        if not isinstance(lab.get(b), bool):
            raise ValueError(f"{mid}: {b} must be true/false")
    if not lab.get("rationale"):
        raise ValueError(f"{mid}: missing rationale")


def _format_call(m, utts: pd.DataFrame) -> str:
    lines = [f"[{r.side}] {r.speaker}: {r.text}" for r in utts.itertuples()]
    return (
        f"TITLE: {m.title}\nCALL TYPE: {m.call_type}\nDATE: {m.start:%Y-%m-%d}\n"
        f"SUMMARY: {m.summary}\n\nTRANSCRIPT:\n" + "\n".join(lines)
    )


def label_with_claude(meetings: pd.DataFrame, utterances: pd.DataFrame) -> dict:
    """Label each meeting with one API call; output is constrained to LABEL_SCHEMA."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit("The 'anthropic' package is required for API labelling: pip install anthropic") from None

    client = anthropic.Anthropic(max_retries=4)
    system = PROMPT_FILE.read_text(encoding="utf-8")
    out = {}
    for i, m in enumerate(meetings.itertuples(), 1):
        u = utterances[utterances.meeting_id == m.meeting_id].sort_values("idx")
        try:
            response = client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                system=system,
                messages=[{"role": "user", "content": _format_call(m, u)}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
                # Re-run on a fallback model if the request is declined; sent as a raw body
                # field so it also works on SDK versions without a typed `fallbacks` argument.
                extra_body={"fallbacks": "default"},
            )
        except anthropic.AuthenticationError:
            raise SystemExit("Anthropic authentication failed: set ANTHROPIC_API_KEY "
                             "(or run `ant auth login`).") from None
        except anthropic.APIConnectionError:
            raise SystemExit("Could not reach the Anthropic API: check network access.") from None
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{m.meeting_id}: model declined to label the call")
        lab = json.loads(next(b.text for b in response.content if b.type == "text"))
        validate_label(m.meeting_id, lab)
        out[m.meeting_id] = lab
        print(f"  [{i}/{len(meetings)}] {m.meeting_id} -> {lab['primary']}")
    return out


def keyword_label(row) -> dict:
    """Offline fallback: keyword topic only; judgement fields are not assessed."""
    return {
        "primary": row.kw_primary,
        "secondary": [],
        "churn_risk": "unknown",
        **{b: False for b in BOOL_FIELDS},
        "root_cause": "unknown",
        "resolution": "unknown",
        "rationale": "Keyword fallback: no LLM label available for this call.",
    }


def read_cache() -> tuple[dict, dict]:
    if not LABEL_CACHE.exists():
        return {}, {}
    labels = json.loads(LABEL_CACHE.read_text(encoding="utf-8"))
    meta = labels.pop("_meta", {})
    for mid, lab in labels.items():
        validate_label(mid, lab)
    return labels, meta


def write_cache(labels: dict, meta: dict) -> None:
    LABEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LABEL_CACHE.write_text(json.dumps({"_meta": meta, **labels}, indent=1, ensure_ascii=False), encoding="utf-8")


def _api_meta(meta: dict, ids: list[str]) -> dict:
    added = meta.get("api_additions", [])
    stamp = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    added.append({"model": MODEL, "generated": stamp, "meeting_ids": ids})
    return {**meta, "api_additions": added}


def load_labels(meetings: pd.DataFrame, utterances: pd.DataFrame | None = None, use_api: bool = False) -> pd.DataFrame:
    """Attach labels to `meetings` (which must already have `call_type` and `kw_primary`)."""
    cached, meta = read_cache()
    missing = meetings[~meetings.meeting_id.isin(cached)]

    if len(missing) and use_api:
        if utterances is None:
            raise ValueError("utterances are required to label new calls through the API")
        print(f"Labelling {len(missing)} new call(s) with {MODEL} ...")
        new = label_with_claude(missing, utterances)
        cached.update(new)
        write_cache(cached, _api_meta(meta, list(new)))
        missing = missing.iloc[0:0]

    if len(missing):
        print(f"Note: {len(missing)} call(s) have no cached LLM label; using the keyword fallback "
              "(run with --use-api to label them with Claude).", file=sys.stderr)

    rows = []
    for m in meetings.itertuples():
        if m.meeting_id in cached:
            rows.append({"meeting_id": m.meeting_id, **cached[m.meeting_id], "label_source": "llm"})
        else:
            rows.append({"meeting_id": m.meeting_id, **keyword_label(m), "label_source": "keyword"})
    return meetings.merge(pd.DataFrame(rows), on="meeting_id", how="left")


def refresh(raw_dir, relabel_all: bool = False) -> None:
    """Label calls in `raw_dir` through the API: only uncached ones, or all with relabel_all."""
    from .call_type import add_call_type
    from .load import load_dataset

    m, u = load_dataset(raw_dir)
    m = add_call_type(m)
    cached, meta = read_cache()
    todo = m if relabel_all else m[~m.meeting_id.isin(cached)]
    if todo.empty:
        print("Every call already has a cached label. Use --all to regenerate them.")
        return

    if relabel_all and LABEL_CACHE.exists():
        backup = LABEL_CACHE.with_suffix(".previous.json")
        backup.write_text(LABEL_CACHE.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Previous cache kept at {backup.name}")

    print(f"Labelling {len(todo)} call(s) with {MODEL} ...")
    new = label_with_claude(todo, u)
    if relabel_all:
        stamp = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        meta = {"model": MODEL, "prompt": "prompts/classify_call.md",
                "source": "Anthropic API, full transcript per call", "generated": stamp}
        cached = {k: v for k, v in cached.items() if k not in set(m.meeting_id)}
    else:
        meta = _api_meta(meta, list(new))
    cached.update(new)
    write_cache(cached, meta)
    print(f"Wrote {len(new)} label(s) to {LABEL_CACHE}")

"""End-to-end pipeline: raw JSON -> enriched `calls` table + `utterances` table.

    load -> call type (rules) -> keyword baseline -> topic + judgement labels
         (LLM cache, optional API, keyword fallback) -> module / competitor tags
         -> sentiment decomposition
"""
from __future__ import annotations

from pathlib import Path

from .call_type import add_call_type
from .config import RAW_DIR
from .llm_labeler import load_labels
from .load import load_dataset
from .sentiment import call_level
from .taxonomy import CATEGORIES, find_competitors, keyword_classify, tag_modules


def run(raw_dir: str | Path = RAW_DIR, use_api: bool = False):
    meetings, utts = load_dataset(raw_dir)
    meetings = add_call_type(meetings)
    meetings["kw_primary"] = meetings.apply(lambda r: keyword_classify(r.title + " " + r.summary, r.call_type), axis=1)
    meetings = load_labels(meetings, utts, use_api=use_api)

    full_text = utts.groupby("meeting_id")["text"].apply(" ".join)
    meetings["modules"] = meetings.apply(lambda r: tag_modules(r.title + " " + r.summary), axis=1)
    meetings["competitors"] = meetings.meeting_id.map(full_text).apply(find_competitors)
    meetings["category"] = meetings.primary.map(lambda c: CATEGORIES[c]["name"])

    return call_level(meetings, utts), utts

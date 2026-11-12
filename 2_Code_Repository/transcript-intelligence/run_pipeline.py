"""Run the full pipeline and write enriched CSVs and charts.

    python run_pipeline.py                                    # the 100-call dataset -> outputs/
    python run_pipeline.py --data data/sample --out outputs/sample   # score other calls
    python run_pipeline.py --use-api                          # label uncached calls with Claude
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ti import discovery, insights, viz  # noqa: E402
from ti.config import OUTPUT_DIR, RAW_DIR  # noqa: E402
from ti.pipeline import run  # noqa: E402

MIN_CALLS_FOR_CORPUS_ANALYSIS = 20  # clustering and trend charts need a corpus

ENRICHED_COLUMNS = [
    "meeting_id", "start", "call_type", "account", "title", "category", "secondary", "label_source", "modules",
    "churn_risk", "competitors", "sentiment_score", "customer_net", "aegis_net", "recovery",
    "outage_related", "comm_gap", "known_issue", "remediation_side_effect", "root_cause", "resolution", "rationale",
]


def draw_charts(calls, health, cm, folder: Path) -> None:
    viz.categories_by_type(calls, folder / "01_categories.png")
    viz.sentiment_by_phase(calls, folder / "02_sentiment_phase.png")
    viz.weekly_trend(calls, folder / "03_weekly_trend.png")
    viz.empathy_gap(calls, folder / "04_empathy_gap.png")
    viz.sentiment_by_category(calls, folder / "05_sentiment_category.png")
    viz.within_call_recovery(calls, folder / "06_within_call.png")
    viz.competitor_chart(cm, folder / "07_competitors.png")
    viz.account_divergence(health, folder / "08_account_divergence.png")
    plt.close("all")


def parse_args():
    p = argparse.ArgumentParser(description="Transcript Intelligence pipeline")
    p.add_argument("--data", type=Path, default=RAW_DIR, help="folder of meeting folders (default: data/raw)")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output folder (default: outputs)")
    p.add_argument("--use-api", action="store_true",
                   help="label calls missing from the cache with Claude (needs `anthropic` and an API key)")
    return p.parse_args()


def main():
    args = parse_args()
    out, charts = args.out, args.out / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    try:
        calls, utts = run(args.data, use_api=args.use_api)
    except FileNotFoundError as e:
        raise SystemExit(f"Error: {e}") from None
    cm = insights.competitor_mentions(utts, calls)
    health = insights.account_health(calls)

    calls[ENRICHED_COLUMNS].to_csv(out / "calls_enriched.csv", index=False)
    health.to_csv(out / "account_health.csv", index=False)
    cm.to_csv(out / "competitor_mentions.csv", index=False)
    insights.early_warning_signals(utts, calls).to_csv(out / "early_warning.csv", index=False)
    insights.say_do_gap(utts, calls).to_csv(out / "say_do_gap.csv", index=False)
    insights.remediation_aftershocks(calls).to_csv(out / "aftershocks.csv", index=False)
    insights.feature_backlog(calls).to_csv(out / "feature_backlog.csv", index=False)

    llm = calls[calls.label_source == "llm"]
    print(f"calls={len(calls)} | LLM-labelled={len(llm)} | keyword fallback={len(calls) - len(llm)}")
    if len(llm):
        print(f"keyword-vs-LLM agreement={(llm.kw_primary == llm.primary).mean():.0%}")

    if len(calls) >= MIN_CALLS_FOR_CORPUS_ANALYSIS:
        draw_charts(calls, health, cm, charts)
        labels, k, _, _ = discovery.cluster(calls)
        print(f"KMeans k={k} | ARI vs LLM={discovery.agreement(labels, calls.primary):.2f}")
    else:
        print(f"{len(calls)} call(s): fewer than {MIN_CALLS_FOR_CORPUS_ANALYSIS}, "
              "so clustering and trend charts are skipped")

    if (llm.call_type == "support").any():
        print("support quality (LLM-labelled calls):")
        for key, val in insights.support_quality(llm).items():
            print(f"  {key:<26} {float(val):.2f}")
    n_csv, n_png = len(list(out.glob("*.csv"))), len(list(charts.glob("*.png")))
    print(f"wrote {n_csv} CSVs to {out} and {n_png} charts to {charts}")


if __name__ == "__main__":
    main()

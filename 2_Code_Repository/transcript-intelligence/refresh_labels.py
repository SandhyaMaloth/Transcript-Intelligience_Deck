"""Label calls through the Anthropic API and update labels/llm_labels.json.

    pip install anthropic
    export ANTHROPIC_API_KEY=...            # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
    python refresh_labels.py                # label only calls missing from the cache
    python refresh_labels.py --all          # regenerate every label (old cache kept as .previous.json)
    python refresh_labels.py --data data/sample

Needs network access; the model defaults to claude-opus-5-5 (override with TI_MODEL).
Not needed to run the pipeline or the notebook: the cached labels ship with the repo.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ti.config import RAW_DIR  # noqa: E402
from ti.llm_labeler import refresh  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Label calls with Claude and update the label cache")
    p.add_argument("--data", type=Path, default=RAW_DIR, help="folder of meeting folders (default: data/raw)")
    p.add_argument("--all", action="store_true", help="regenerate every label, not only missing ones")
    args = p.parse_args()
    refresh(args.data, relabel_all=args.all)

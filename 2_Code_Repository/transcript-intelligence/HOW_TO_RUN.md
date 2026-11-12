# How to run Transcript Intelligence

Everything below runs **offline, with no API key**, unless a step says otherwise.
Run all commands from the `transcript-intelligence` folder (the one containing `run_pipeline.py`).

## 1. Requirements
| | |
|---|---|
| Python | 3.10 or newer (tested on 3.13) |
| Packages | `requirements.txt`: pandas, numpy, scikit-learn, scipy, matplotlib |
| Optional | `jupyter` to open the notebook interactively; `anthropic` only for API labelling (step 6) |
| OS | Windows, macOS or Linux |

## 2. Set up
```bash
python -m venv .venv
# macOS / Linux:        source .venv/bin/activate
# Windows PowerShell:   .venv\Scripts\Activate.ps1
# Windows cmd:          .venv\Scripts\activate.bat
pip install -r requirements.txt
```

## 3. Run the pipeline on the 100-call dataset
```bash
python run_pipeline.py
```
It takes about 10 seconds. Expected output:
```
calls=100 | LLM-labelled=100 | keyword fallback=0
keyword-vs-LLM agreement=80%
KMeans k=11 | ARI vs LLM=0.38
support quality (LLM-labelled calls):
  n_support_calls            30.00
  aegis_root_cause_pct       0.70
  ...
wrote 7 CSVs to .../outputs and 8 charts to .../outputs/charts
```
Results land in `outputs/` (CSVs) and `outputs/charts/` (PNGs). The README's *Outputs* table describes each file.

## 4. Open the notebook
The notebook holds every result, plus the reasoning behind it, already executed.
```bash
pip install jupyter
jupyter notebook notebooks/transcript_intelligence.ipynb     # then Kernel > Restart & Run All
```
To re-execute it without Jupyter (this rewrites the `.ipynb` in place):
```bash
python notebooks/build_notebook.py
```

## 5. Score new transcripts
Put each call in its own folder holding the six JSON files (`meeting-info.json`, `transcript.json`,
`summary.json`, `speakers.json`, `speaker-meta.json`, `events.json`), then point the pipeline at the
parent folder. A ready-made example is in `data/sample/`:
```bash
python run_pipeline.py --data data/sample --out outputs/sample
```
Calls that are not in the label cache get the **keyword fallback**: correct call type, modules,
competitors and sentiment, plus a keyword topic, with `label_source = keyword` and the judgement
fields (churn risk, root cause, known issue ...) marked `unknown`. Clustering and the trend
charts need a corpus, so they are skipped for runs with fewer than 20 calls.

To score new calls *together with* the original 100, copy their folders into `data/raw/` and
run `python run_pipeline.py`.

## 6. Optional: label calls with Claude (needs network and an API key)
```bash
pip install anthropic
# macOS / Linux:        export ANTHROPIC_API_KEY=sk-ant-...
# Windows PowerShell:   $env:ANTHROPIC_API_KEY="sk-ant-..."

python run_pipeline.py --data data/sample --out outputs/sample --use-api   # label uncached calls, then run
python refresh_labels.py --data data/sample                                # label uncached calls only
python refresh_labels.py --all                                             # regenerate all 100 labels (~100 API calls)
```
* New labels are added to `labels/llm_labels.json`. `--all` first keeps the old cache as
  `labels/llm_labels.previous.json`.
* The model defaults to `claude-opus-5-5`. Set `TI_MODEL` to use another model.
* Each response is constrained to the label schema and validated again before it is cached.
* Regenerating all labels may shift the headline numbers slightly, because the shipped labels
  came from an interactive session (see the README's *Data notes*).

## 7. Run the tests
```bash
python -m unittest discover tests
```
10 tests, about 2 seconds, no network. They pin the headline numbers (80% agreement, ARI 0.38,
−0.37 vs +0.01, 32-day lead, 3 → 19 SentinelShield calls), check the new-call path on
`data/sample/`, and check the API request shape against a fake client.

Optional lint check: `pip install ruff && ruff check .`

## Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: pandas` (or another package) | Activate the virtual environment, then run `pip install -r requirements.txt` |
| `No meeting folders ... found in <dir>` | `--data` must point to the folder that *contains* the meeting folders, not to a meeting folder itself |
| `Note: N call(s) have no cached LLM label` | Expected for new calls; add `--use-api` to label them with Claude |
| `Anthropic authentication failed` | Set `ANTHROPIC_API_KEY` (or run `ant auth login`) before using `--use-api` / `refresh_labels.py` |
| Notebook cannot find `src/ti` | Start Jupyter from the repo folder or a subfolder of it |

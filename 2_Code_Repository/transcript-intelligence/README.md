# Transcript Intelligence

Turns 100 raw call recordings (support, account-management and internal calls at
*Aegis Cloud Security*) into categorised, sentiment-scored, cross-linked insight.

**To install and run it, see [HOW_TO_RUN.md](HOW_TO_RUN.md).** In short:

```bash
pip install -r requirements.txt
python run_pipeline.py                 # -> outputs/*.csv and outputs/charts/*.png, offline, ~10 s
```

## Pipeline

| Stage | What | Method | Why |
|---|---|---|---|
| 1 | Call type: support / external / internal | Rules on email domains and title patterns | Structural, not semantic; free, exact and auditable |
| 2 | Topic (1 primary + up to 2 secondary) and judgement fields | Claude with a fixed taxonomy, cached per call; keyword fallback for uncached calls | Calls are multi-intent; churn risk, root cause and "known bug?" need reading comprehension |
| 2b | Taxonomy discovery and label sanity check | TF-IDF + KMeans, keyword baseline | Shows where the categories came from and what the LLM adds |
| 2c | Product modules, competitors | Regex | Closed vocabularies; no model needed |
| 3 | Sentiment | Shipped sentence labels split by speaker side and call position | A call score hides *whose* sentiment and *when* it changes |
| 4 | Insights | Joins across calls, accounts and time | The value is in connecting calls, not scoring them one at a time |

### How a call gets its topic label
1. **Cached LLM label** in `labels/llm_labels.json` → `label_source = llm`.
2. **Not cached, `--use-api` given** → labelled by Claude (output constrained to the label schema) and added to the cache.
3. **Not cached, no API** → keyword baseline → `label_source = keyword`. The judgement fields are marked `unknown`, and corpus statistics such as support quality use LLM-labelled calls only.

So a new transcript never breaks the pipeline: drop its folder anywhere and run
`python run_pipeline.py --data <folder>`. `data/sample/` holds one such call.

## Repository layout
```
data/raw/                 100 meeting folders, 6 JSON files each (as supplied)
data/sample/              1 extra call that is not in the label cache (new-call demo and tests)
prompts/classify_call.md  LLM labelling prompt: taxonomy, field definitions, rules
labels/llm_labels.json    cached LLM labels, one object per meeting, schema-checked on load
src/ti/
  config.py        paths and the dataset's event dates (outage, recovery, launch)
  load.py          JSON -> meetings + utterances tables; tags each sentence aegis/customer
  call_type.py     stage 1: rule-based call type
  taxonomy.py      10-category taxonomy, keyword baseline, module and competitor tagging
  llm_labeler.py   stage 2: label cache, Claude API labelling, keyword fallback, schema
  discovery.py     TF-IDF + KMeans used to discover the taxonomy and sanity-check labels
  sentiment.py     stage 3: customer vs Aegis sentiment, within-call trajectory, phases
  insights.py      stage 4: early warning, say/do gap, aftershocks, account 360, competitors ...
  viz.py           charts
  pipeline.py      wires stages 1-3 together
run_pipeline.py           one-command run (--data, --out, --use-api)
refresh_labels.py         label uncached calls through the API (--all regenerates every label)
notebooks/transcript_intelligence.ipynb   executed notebook: every output plus the reasoning
notebooks/build_notebook.py               notebook source; re-executes and rewrites the .ipynb
tests/test_pipeline.py    regression tests on the headline numbers, new-call path, API request shape
outputs/                  enriched CSVs and charts from the 100-call run
```

## Outputs
| File | Contents |
|---|---|
| `calls_enriched.csv` | One row per call: type, category, secondary topics, label source, modules, competitors, churn risk, sentiment split, judgement fields, rationale |
| `account_health.csv` | Per account: external vs support sentiment, churn risk, competitors, call-by-call journey |
| `early_warning.csv` | Internal pre-outage quotes describing the failure that later happened |
| `say_do_gap.csv` | What customers were told vs what engineering said in the same weeks |
| `aftershocks.csv` | Post-outage cases caused by the outage remediation itself |
| `competitor_mentions.csv` | Competitor mentions per call, who raised them, which module |
| `feature_backlog.csv` | Feature requests by account and module |
| `charts/01-08_*.png` | The charts used in the notebook and the slide deck |

## Data notes
* The second zip supplied (6 meetings) is an exact, file-for-file duplicate of 6 meetings in
  the first, so the dataset is 100 unique calls.
* No synthetic data was added: all three call types are present (40 external / 30 support /
  30 internal) and 24 of 32 accounts recur across calls.
* **Label provenance:** the 100 cached labels were produced by Claude in an interactive session,
  applying the prompt's taxonomy and field definitions to each call's summary and transcript
  excerpts, not by the API code path. `python refresh_labels.py --all` regenerates them through
  the API from the full transcripts; numbers may shift slightly if you do.

## Key design decisions
| Decision | Choice | Reason |
|---|---|---|
| Call type | Rules on email domains + title patterns | Structural, not semantic; free and exact |
| Topic | LLM with a fixed, discovered taxonomy; primary + secondary labels | Calls are multi-intent; keyword rules agree 80% but miss the renewal calls about the outage |
| New calls | API on request, keyword fallback otherwise | The pipeline runs offline on any input, and every row says how it was labelled |
| Validation | Keyword baseline (80% agreement) + KMeans (ARI 0.38) | Shows what the LLM adds; doubles as a drift alarm |
| Sentiment | Shipped sentence labels split by speaker side and call position | Call scores hide that customers are far more negative than agents |
| Synthetic data | None added | Types are balanced and accounts recur; fake data would inflate trends |

## Headline findings
1. The Detect outage (10 Mar) drove every sentiment trend. External calls recovered (4.0); **support did not (3.0)**, because 5 April tickets trace back to the outage remediation itself.
2. The single point of failure was described in internal calls **32 days before** it failed, while customer-facing staff were telling customers the fix was already in.
3. Customer-side sentiment on support calls is **−0.37** vs **+0.01** for agents: the blended call score hides the customer's frustration.
4. 70% of support calls have an Aegis-side root cause; 4 were already-known bugs, and in 10 the customer complained that Aegis didn't tell them first.
5. SentinelShield mentions went from 3 calls before the outage to 19 after; Comply v2 (GA 7 Apr, 4.4–4.5 sentiment) is the main recovery lever.

## Known limitations
* 100 calls over 12 weeks with one dominant event, so trends are directional.
* Labels come from one model pass; churn risk is the most subjective field.
* Keyword-fallback labels carry a topic only; judgement fields need an LLM label.
* Sentence sentiment comes from the source data (validated against the call score, ρ = 0.93), not from human labels.
* The early-warning and say/do phrase lists are curated from this dataset; on new data they are a starting point, not a general detector.
* The transcripts disagree on some details (e.g. outage length 6–24 h depending on the call); the analysis reports the consensus (~6 h of full outage).

"""Build and execute `transcript_intelligence.ipynb` without Jupyter.

Code cells run in one shared namespace; stdout, the value of the last
expression (DataFrames as HTML) and matplotlib figures are captured as cell
outputs. Run: `python notebooks/build_notebook.py`. Opening the .ipynb in
Jupyter and choosing "Run All" gives the same result.
"""
import ast
import base64
import contextlib
import io
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.pop("ANTHROPIC_API_KEY", None)  # the notebook never calls the API
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
os.chdir(HERE)
CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


md("""
# Transcript Intelligence — technical notebook

**Dataset:** 100 call recordings from *Aegis Cloud Security* (fictional B2B security SaaS: Detect, Protect, Comply, Identity), 3 Feb – 28 Apr 2026.
Each call ships 6 JSON files: meeting info, transcript sentences (with speaker + sentence sentiment), speaker timeline, join/leave events and a vendor summary (summary, topics, call-level sentiment, key moments).

**Pipeline**

| Stage | What | Method | Why this method |
|---|---|---|---|
| 1 | Call type (support / external / internal) | Rules on metadata | Structural fact about who's on the call — rules are free, exact and auditable |
| 2 | Topic category + judgement fields | LLM (Claude) with a fixed taxonomy, cached; keyword fallback for uncached calls | Calls are multi-intent; churn risk, root cause, "known bug?" need reading comprehension |
| 2b | Taxonomy discovery + sanity check | TF-IDF + KMeans, keyword baseline | Shows where the categories came from and what the LLM adds over cheap methods |
| 2c | Entities: product modules, competitors | Regex | Closed vocabularies — no model needed |
| 3 | Sentiment | Decompose shipped sentence labels by speaker side & call position | A call score hides *whose* sentiment and *when* |
| 4 | Insights | Joins across calls, accounts and time | The value is in connecting calls, not scoring them one by one |

**Data decisions:** I did not generate synthetic data. All three call types are present and balanced (40 external / 30 support / 30 internal), and 24 of 32 customer accounts appear in more than one call, which is enough to show cross-call patterns. Adding synthetic calls would make the trends look stronger without making them more true. The second zip in the submission is an exact 6-call subset of the first (checked file by file), so it was deduplicated.
""")

code("""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore", message="Converting to PeriodArray")
# Repo root = the folder holding src/ti, wherever Jupyter was started from
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "ti").is_dir())
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 120); pd.set_option("display.width", 200)
from ti.load import load_dataset
from ti.call_type import add_call_type
from ti.taxonomy import CATEGORIES
from ti import discovery, insights, sentiment, viz
from ti.pipeline import run
""")

md("""
## 1 · Load & call-type classification

Every speaker name maps to an email in `allEmails` (311/311), so each sentence is tagged **aegis** or **customer**. That one join is what makes the customer-voice sentiment in section 3 possible.
""")
code("""
meetings, utts = load_dataset(ROOT / "data" / "raw")
print(f"{len(meetings)} calls, {len(utts):,} sentences, {meetings.duration_min.sum()/60:.1f} hours of audio")
print("speaker->email match rate:", utts.email.notna().mean())
meetings = add_call_type(meetings)
meetings.call_type.value_counts()
""")
code("""
# Rule check: one example title per call type
meetings.groupby("call_type").title.apply(lambda s: list(s.head(3)))
""")

md("""
## 2 · Topic categorisation

### 2a. Discovery — what groupings exist?
Before writing a taxonomy I clustered the calls (TF-IDF on title + summary + vendor topic tags, KMeans, k chosen by silhouette) and read each cluster's top terms.
""")
code("""
calls, utts = run()          # full pipeline (stages 1–3), cached LLM labels
labels, k, scores, top_terms = discovery.cluster(calls)
print("best k =", k, "| silhouette by k:", {kk: round(v, 3) for kk, v in scores.items()})
pd.DataFrame({"size": pd.Series(labels).value_counts().sort_index(),
              "top terms": pd.Series({c: ", ".join(t) for c, t in top_terms.items()})})
""")
md("""
**What the clusters say.** The groups are easy to recognise: outage, billing, Comply / multi-framework, Comply v2 launch, renewal + competitive, sprint rituals, Identity, onboarding. Those became the base of the taxonomy. But the **silhouette is very low (~0.04)**. Every call shares the same vocabulary (Detect, outage, compliance), so the clusters overlap heavily. Some clusters also form around a *customer name* ("nova retail", "summit trust", "yuki") rather than an intent. On its own, clustering is good for finding themes and poor at assigning calls to them.

### 2b. Taxonomy
Ten categories, each defined by **the decision it supports and who owns it**. One primary label per call, plus up to two secondary labels (multi-label).
""")
code("""
pd.DataFrame(CATEGORIES).T[["name", "owner", "definition"]]
""")
md("""
### 2c. LLM labelling
Prompt: `prompts/classify_call.md`. The model reads the summary and the full transcript and returns JSON: `primary`, `secondary`, `churn_risk`, `outage_related`, `comm_gap`, `support_delay`, `known_issue`, `remediation_side_effect`, `root_cause`, `resolution`, `rationale`.
Results are cached in `labels/llm_labels.json`, so this notebook runs offline and gives the same numbers every time.

> **Provenance, stated plainly:** the cached labels were produced by Claude in an interactive session, applying this prompt's taxonomy and field definitions to each call's summary and transcript excerpts. They were not produced by the API code path. `python refresh_labels.py --all` (needs `pip install anthropic` and an API key) regenerates all 100 labels through the API from the full transcripts, with the output constrained to the label schema. Every cached label is also schema-validated on load.
>
> **New calls:** a call with no cached label is never dropped. With `--use-api` it is labelled by Claude and added to the cache; otherwise it gets the keyword label and `label_source = "keyword"`, with the judgement fields marked `unknown` (see 2e).
""")
code("""
calls[["start", "call_type", "title", "category", "secondary", "churn_risk", "rationale"]].sample(8, random_state=1)
""")
code("""
fig = viz.categories_by_type(calls); plt.show()
""")
md("### Examples per category")
code("""
g = calls.sort_values("start").groupby("category")
ex = pd.DataFrame({"n": g.size(), "examples": g.title.agg(lambda t: " | ".join(t.head(3)))})
ex.sort_values("n", ascending=False)
""")
md("""
### 2d. How much does the LLM add? (agreement check)
A keyword scorer over title + summary (`taxonomy.keyword_classify`) agrees with the LLM on 80% of calls, and KMeans reaches an ARI of 0.38. The useful part is **where they disagree**:
""")
code("""
print(f"keyword vs LLM agreement: {(calls.kw_primary == calls.primary).mean():.0%}")
print(f"KMeans vs LLM ARI: {discovery.agreement(labels, calls.primary):.2f}")
dis = calls[calls.kw_primary != calls.primary][["title", "kw_primary", "primary", "secondary"]]
dis
""")
md("""
**Reading the disagreements.** 18 of the 20 come from the keyword scorer picking the loudest topic word, either *outage* (10) or *compliance* (8), over the purpose of the call. The biggest single group is **renewal calls labelled OUTAGE** (4). A sales manager filtering for renewals would miss exactly those at-risk accounts under keyword rules. That is the case for an LLM: it labels a call by its purpose, not by its most frequent words.

**Production design:** rules for call type → LLM for topic and judgement fields → keyword scorer kept as an offline fallback and a drift alarm (if agreement drops sharply, the taxonomy or prompt needs revisiting).

### 2e. Scoring a new call
`data/sample/` holds one call that is not in the label cache: a support case about an alert storm after a Detect agent upgrade, where the customer also mentions a June renewal and a SentinelShield quote. Without an API call it is scored with the keyword fallback:
""")
code("""
new_calls, new_utts = run(ROOT / "data" / "sample")
new_calls[["title", "call_type", "label_source", "category", "modules", "competitors",
           "sentiment_score", "customer_net", "aegis_net", "open_net", "close_net", "recovery"]].T
""")
md("""
Everything structural is right: support call, Detect module, SentinelShield mention, and the customer opening at −1.0 and closing at +1.0 once the baseline was restored live. The keyword topic is DEFECT, which is correct. What the fallback cannot give is the judgement layer: an LLM label (`python run_pipeline.py --data data/sample --use-api`) would add RENEWAL as secondary, churn risk *high*, a known issue and a communication gap. Those are the fields that make this call worth escalating to the account owner.
""")

md("""
## 3 · Sentiment

### 3a. Can we trust the shipped labels?
The vendor's call-level score (1–5) and our aggregate of sentence labels agree closely (Spearman ρ ≈ 0.93), so the rest of this section builds on the sentence labels.
""")
code("""
pd.Series(sentiment.validate_against_vendor(calls)).astype(float).round(4)
""")
code("""
calls.groupby("call_type")[["sentiment_score", "customer_net", "aegis_net", "empathy_gap", "recovery", "customer_talk_share"]].mean().round(2)
""")
md("### 3b. Trend over time — four phases set by the two events in the data")
code("""
fig = viz.weekly_trend(calls); plt.show()
fig = viz.sentiment_by_phase(calls); plt.show()
calls.pivot_table(index="phase", columns="call_type", values="sentiment_score", aggfunc=["mean", "size"]).round(2)
""")
md("""
**What it means**
* All three call types hit their low in the same two weeks (10–23 Mar). The outage drives sentiment everywhere: internal morale, customer calls and support.
* **External calls recovered to 4.0, close to their pre-outage 4.25. Support did not (3.0 now vs 3.1 before; the post-launch support average sits below the recovery-phase level).** Of the 8 support calls since Comply v2 launched, 7 have an Aegis-side root cause and 5 trace back to the outage remediation itself (section 4c). Leadership looking at renewal calls would conclude "we're through it"; the support queue says otherwise.
""")
code("""
fig = viz.sentiment_by_category(calls); plt.show()
""")
md("### 3c. Whose sentiment? The empathy gap")
code("""
fig = viz.empathy_gap(calls); plt.show()
""")
md("""
On support calls, customers' sentences net out to **−0.37** while Aegis agents sit at **+0.01**. A call-level score blends the two, so an angry customer talking to a calm agent reads as "mixed". **Recommendation:** report customer-side sentiment as the headline metric for support and CS; keep the blended score for internal calls only.
""")
md("### 3d. Within-call trajectory — do calls de-escalate?")
code("""
fig = viz.within_call_recovery(calls); plt.show()
calls[calls.call_type != "internal"].groupby("call_type").recovery.apply(lambda s: f"{(s > 0).mean():.0%} of calls end better than they start")
""")
md("""
About 80% of customer calls end more positive than they start, so agents are good at de-escalating *inside* a call. The exception is outage calls on the support line, which open at −0.82 and close at −0.68. Agents cannot talk their way out of a platform failure, which points to a product and communications fix rather than a support training fix.
""")

md("""
## 4 · Beyond topic & sentiment

### 4a. Early warning: the outage was visible in internal calls 32 days in advance
""")
code("""
ew = insights.early_warning_signals(utts, calls)
ew.groupby(["start", "title"]).agg(lead_days=("lead_days", "first"), quotes=("text", "size"), example=("text", "first"))
""")
md("""
The single point of failure in Detect ingestion was described exactly ("all running through a single processing node with no fallback… we lose event visibility entirely") on **5 Feb**. It came up again on 16 Feb, at the 28 Feb all-hands, and on 2 Mar, and stayed scoped as a "design spike". It failed on **10 Mar**.
**Product idea:** a *risk-signal tracker* that pulls risk language ("single point of failure", "no fallback", "we're deferring…") from internal calls, links it to the product area, and raises an alert when the same risk shows up in ≥3 calls without a committed ticket.
""")
md("### 4b. Say/do gap: what customers were told vs what engineering said that week")
code("""
sd = insights.say_do_gap(utts, calls)
sd[["start", "kind", "title", "speaker", "text"]]
""")
md("""
On 11 Feb an AE told a prospect that "the fix that went in was adding redundant processing nodes and implementing a circuit breaker". On 9 Mar, **one day before the outage**, a CSM told a renewal customer "we've put redundancy measures in place". Engineering's calls in the same weeks show the redundancy was still a plan (2 Mar: "design spike on redundancy"). It shipped on 28–29 Mar.
Only a tool that reads **internal and external calls together** can catch this. It matters because customers who heard "it's fixed" were hit hardest by the loss of trust afterwards (e.g. Brightpath: was told 11 Feb → evaluating SentinelShield on 22 Apr).
""")
md("### 4c. Remediation aftershocks: the fix created new tickets")
code("""
insights.remediation_aftershocks(calls)
""")
md("""
Five support cases between 10 and 27 Apr **trace back to the outage remediation itself** (three confirmed on the call, two given as the support engineer’s likely cause): rerouted network paths slowed backups, an overloaded post-remediation cluster delayed alerts and Identity sync, the new circuit breaker throttled a SIEM integration, and replay traffic triggered overage charges. Three different modules were affected. **Recommendation:** treat a large remediation like a release: change review, a canary, tenant-level monitoring, and proactive notice to the tenants whose infrastructure moved.
""")
md("### 4d. Account 360: the AE and support see different customers")
code("""
health = insights.account_health(calls)
fig = viz.account_divergence(health); plt.show()
health[["account", "n_calls", "avg_external", "avg_support", "last_churn", "competitors", "journey"]].head(12)
""")
md("""
For accounts with both kinds of calls, support calls average **~1.1 points lower** (20 accounts) than account-management calls. Several swings happen within days. Clearwater: 4.8 at kickoff → 2.4 three days later on a billing dispute. Crestline: 4.8 on Identity expansion → 2.4 three days later on a sync delay with a 5-day support wait. **Product idea:** an account health view that merges support and CS signals, and alerts the account owner when a support call goes negative near a renewal.
""")
md("### 4e. Competitive intelligence")
code("""
cm = insights.competitor_mentions(utts, calls)
fig = viz.competitor_chart(cm); plt.show()
cm.groupby("competitor").agg(calls=("meeting_id", "nunique"), post_outage=("post_outage", "sum"),
                             raised_by_customer=("said_by_customer", lambda s: (s > 0).sum()))
""")
md("""
SentinelShield appears in 3 calls before the outage and **19 after**, and in 13 of the 22 calls the customer is the one who brings it up. It goes after Detect on reliability and SLA (offering a 99.9% SLA). VaultEdge comes up on Comply/Protect, CyberNova on Identity pricing. The internal win/loss call puts 5 lost deals down to the outage and $2.1M ACV lost in Q1 overall.
""")
md("### 4f. Support quality — whose fault, and how it ends")
code("""
pd.Series(insights.support_quality(calls)).astype(float).round(2)
""")
code("""
sup = calls[calls.call_type == "support"]
pd.crosstab(sup.primary, sup.resolution)
""")
md("""
* **70% of support calls have an Aegis-side root cause** and only 13% a customer-side one. Support here is mostly absorbing product and operations failures, not configuration questions.
* **4 cases were bugs Aegis already knew about** (backup agent 3.4.1, LogVault connector 3.4.1, Identity 3.4.1, Detect patch 4.2.1), and in **10 support calls** the customer complained that Aegis did not tell them first. A known-issue → affected-tenant notification would prevent a whole class of tickets.
* Only 17% resolve on the call and 43% escalate.
""")
md("### 4g. Product backlog from the voice of the customer")
code("""
insights.feature_backlog(calls)
""")
md("""
## 5 · Limitations & next steps
* **n = 100, 12 weeks, one dominant event.** The trends describe this period and should not be read as general effects. Phase averages rest on 3–15 calls each.
* **LLM labels were produced once by one model.** Before production: double-label a 20% sample, measure agreement per field (churn risk is the most subjective), and version the prompt.
* Sentence sentiment comes from the source vendor and was validated against the call score, not against human labels.
* **Next:** embeddings for semantic search ("show me every call where a customer threatened to leave over SLA"), commitment tracking (did we send the post-mortem we promised?), and a revenue join (ARR per account) to rank risk in dollars.
""")


def _exec_cell(src, ns):
    outputs = []
    tree = ast.parse(src)
    last_expr = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, "<cell>", "exec"), ns)
        result = eval(compile(ast.Expression(last_expr.value), "<cell>", "eval"), ns) if last_expr else None
    if buf.getvalue():
        outputs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
    for num in plt.get_fignums():
        fig = plt.figure(num)
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}})
    plt.close("all")
    if result is not None and not isinstance(result, matplotlib.figure.Figure):
        data = {"text/plain": repr(result)}
        if isinstance(result, (pd.DataFrame, pd.Series)):
            obj = result.to_frame() if isinstance(result, pd.Series) else result
            data["text/html"] = obj.to_html(max_rows=40)
        outputs.append({"output_type": "execute_result", "execution_count": None, "metadata": {}, "data": data})
    return outputs


def build():
    ns = {}
    plt.show = lambda *a, **k: None
    nb_cells, n = [], 0
    for kind, src in CELLS:
        lines = [line + "\n" for line in src.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            n += 1
            outs = _exec_cell(src, ns)
            for o in outs:
                if o["output_type"] == "execute_result":
                    o["execution_count"] = n
            nb_cells.append({"cell_type": "code", "execution_count": n, "metadata": {}, "outputs": outs, "source": lines})
    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (HERE / "transcript_intelligence.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote notebook with {len(nb_cells)} cells")


if __name__ == "__main__":
    build()

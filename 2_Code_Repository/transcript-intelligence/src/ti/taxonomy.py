"""Topic taxonomy, keyword baseline and deterministic entity tagging.

The taxonomy was discovered, not assumed:
  1. TF-IDF + KMeans over summaries and vendor topic tags (discovery.py)
     surfaced the natural groupings (outage, billing, compliance, renewal ...).
  2. Clusters that meant the same thing to a stakeholder were merged, and
     clusters that mixed intents were split (e.g. renewal vs. outage fallout).
  3. Each category is defined by the decision it informs, so a support
     leader, PM or sales manager can filter straight to what they own.

Each call gets one primary category plus up to two secondary categories, so a
renewal call dominated by outage fallout is RENEWAL primary + OUTAGE secondary.
"""
import re

CATEGORIES = {
    "OUTAGE": {
        "name": "Outage & Incident Response",
        "owner": "Eng leads, Support leaders",
        "definition": "Service outage handling: war rooms, customer impact, RCA, post-mortems, reliability follow-up.",
    },
    "DEFECT": {
        "name": "Product Defects & Tech Issues",
        "owner": "Support leaders, Eng leads",
        "definition": "A specific bug, regression, misconfiguration or performance problem being troubleshot.",
    },
    "BILLING": {
        "name": "Billing & Usage Disputes",
        "owner": "Finance, Account managers",
        "definition": "Invoices, overages, seat counts, proration, credits.",
    },
    "FEATURE": {
        "name": "Feature Requests & Product Gaps",
        "owner": "Product managers",
        "definition": "Customer asks for capability that does not exist or is insufficient.",
    },
    "RENEWAL": {
        "name": "Renewals & Account Health",
        "owner": "Sales managers, CSMs",
        "definition": "Renewal, QBR, contract, pricing, expansion or retention conversation.",
    },
    "ONBOARD": {
        "name": "Onboarding & Deployment",
        "owner": "CS / Professional services",
        "definition": "Kickoffs, deployment planning, tuning after go-live.",
    },
    "COMPLY": {
        "name": "Compliance & Audit Readiness",
        "owner": "Product (Comply), Sales",
        "definition": "Audit prep (SOC 2, HIPAA, PCI, ISO), Comply demos, internal certification work.",
    },
    "PLANNING": {
        "name": "Eng Planning & Roadmap",
        "owner": "Eng leads, Product",
        "definition": "Sprint planning, retros, standups, roadmap and quarterly planning, all-hands.",
    },
    "LAUNCH": {
        "name": "Product Launch (Comply v2)",
        "owner": "Product, Eng, Marketing",
        "definition": "Design review, launch readiness, GA plan, adoption review for a launch.",
    },
    "GTM": {
        "name": "Competitive & GTM Strategy",
        "owner": "Sales leadership, Product marketing",
        "definition": "Competitive landscape, win/loss, competitive response.",
    },
}

INTERNAL_ONLY = {"PLANNING", "LAUNCH", "GTM"}
CUSTOMER_ONLY = {"BILLING", "ONBOARD", "FEATURE", "RENEWAL", "DEFECT"}
DEFAULT_BY_CALL_TYPE = {"internal": "PLANNING", "support": "DEFECT", "external": "RENEWAL"}

# Keyword baseline over title + summary: the offline fallback for calls with
# no LLM label, and the yardstick for how much the LLM adds.
KEYWORDS = {
    "OUTAGE": ["outage", "war room", "pipeline failure", "post-incident", "post-mortem", "root cause analysis",
               "incident", "blackout", "loss of threat visibility", "reliability"],
    "DEFECT": ["bug", "error", "failure", "failing", "regression", "timeout", "latency", "slow", "sync",
               "false positive", "troubleshoot", "workaround", "patch", "known issue", "alert storm", "baseline"],
    "BILLING": ["invoice", "billing", "overage", "proration", "charge", "credit", "seat"],
    "FEATURE": ["feature request", "feature gap", "pain point", "not currently support", "roadmap", "feedback"],
    "RENEWAL": ["renewal", "contract", "pricing", "business review", "account review", "qbr", "expansion"],
    "ONBOARD": ["onboarding", "kickoff", "deployment", "setup", "implementation", "tuning", "workshop"],
    "COMPLY": ["audit", "soc 2", "hipaa", "pci", "iso 27001", "compliance", "evidence"],
    "PLANNING": ["sprint", "standup", "retro", "roadmap", "quarterly planning", "all hands", "all-hands", "capacity"],
    "LAUNCH": ["launch", "ga ", "general availability", "design review", "adoption"],
    "GTM": ["competitive", "win/loss", "win-loss", "competitor", "landscape"],
}

MODULES = {
    "Detect": r"\bdetect\b",
    "Protect": r"\bprotect\b|\bbackup",
    "Comply": r"\bcomply\b",
    "Identity": r"\bidentity\b|\bsso\b|\bmfa\b|\bsaml\b|\bscim\b|\bldap\b",
}

COMPETITORS = ["SentinelShield", "VaultEdge", "CyberNova", "Fortiguard Pro"]


def keyword_classify(text: str, call_type: str) -> str:
    """Highest keyword count wins; ties go to the earlier category."""
    t = text.lower()
    excluded = CUSTOMER_ONLY if call_type == "internal" else INTERNAL_ONLY
    scores = {c: sum(t.count(k) for k in kws) for c, kws in KEYWORDS.items() if c not in excluded}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else DEFAULT_BY_CALL_TYPE[call_type]


def tag_modules(text: str) -> list[str]:
    t = text.lower()
    return [m for m, pat in MODULES.items() if re.search(pat, t)]


def find_competitors(text: str) -> list[str]:
    t = text.lower()
    return [c for c in COMPETITORS if c.lower() in t]

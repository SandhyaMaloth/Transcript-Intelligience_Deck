"""Unsupervised discovery — how the taxonomy was found, and a sanity check on it.

TF-IDF over (title + summary + vendor topic tags) -> KMeans.
We pick k by silhouette over a small range, print the top terms per cluster,
then compare clusters with the final LLM labels (Adjusted Rand Index +
contingency table). High-but-imperfect agreement is the expected result:
clusters find the obvious lexical groups; the LLM separates intents that share
vocabulary (e.g. an outage support call vs a renewal call about the outage).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

STOP_EXTRA = [
    "aegis", "cloud", "security", "team", "call", "discussed", "customer", "meeting",
    "including", "agreed", "steps", "concluded", "held", "met", "expressed", "would",
]


def build_corpus(meetings: pd.DataFrame) -> pd.Series:
    return (
        meetings["title"] + ". " + meetings["summary"] + " " +
        meetings["vendor_topics"].apply(lambda t: " ".join(t))
    )


K_RANGE = range(6, 13)


def cluster(meetings: pd.DataFrame, k_range=K_RANGE, seed: int = 7):
    """KMeans over TF-IDF for each k in k_range; returns labels, best k, silhouettes, top terms."""
    # joblib probes physical cores via `wmic`, which is missing on recent Windows
    # and prints a traceback; a user CPU limit skips the probe.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))
    vec = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS.union(STOP_EXTRA)),
        ngram_range=(1, 2), min_df=2, max_df=0.6, sublinear_tf=True,
    )
    X = vec.fit_transform(build_corpus(meetings))
    scores, models = {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        scores[k] = silhouette_score(X, km.labels_, metric="cosine")
        models[k] = km
    best_k = max(scores, key=scores.get)
    km = models[best_k]
    terms = np.array(vec.get_feature_names_out())
    top_terms = {
        c: list(terms[np.argsort(-km.cluster_centers_[c])[:8]]) for c in range(best_k)
    }
    return km.labels_, best_k, scores, top_terms


def agreement(cluster_labels, llm_labels) -> float:
    return adjusted_rand_score(llm_labels, cluster_labels)

"""Analytics derived only from the latest recommendation run (no persistent history)."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

COMPONENT_COLUMNS = ["topic", "difficulty", "format", "performance", "quality"]


def results_to_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    component_labels = {
        "topic": "Topic Score",
        "difficulty": "Difficulty Match Score",
        "format": "Format Score",
        "performance": "Performance Score",
        "quality": "Quality Score",
    }
    for r in items:
        rows.append({
            "Rank": r["rank"],
            "Title": r["title"],
            "Type": r["type"],
            "Source": r["source"],
            "Origin": r["origin"],
            "Score": r["score"],
            "Difficulty": r["metadata"].get("difficulty") or "Unknown",
            "Relevant": r["relevant"],
            "URL": r["url"],
            **{component_labels[c]: r["components"][c] for c in COMPONENT_COLUMNS},
        })
    cols = ["Rank", "Title", "Type", "Source", "Origin", "Score", "Difficulty", "Relevant", "URL"] + \
           [component_labels[c] for c in COMPONENT_COLUMNS]
    return pd.DataFrame(rows, columns=cols)


def summary_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    recs = result.get("recommendations", [])
    scores = [r["score"] for r in recs]
    return {
        "recommended": len(recs),
        "evaluated": result.get("candidates_evaluated", 0),
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "top_score": max(scores) if scores else 0.0,
        "live": sum(1 for r in recs if r["origin"] == "live"),
        "local": sum(1 for r in recs if r["origin"] == "local"),
        "partner": sum(1 for r in recs if r["origin"] == "partner"),
    }


def type_distribution(result: Dict[str, Any]) -> pd.Series:
    df = results_to_df(result.get("recommendations", []))
    return df["Type"].value_counts() if not df.empty else pd.Series(dtype=int)


def mean_components(result: Dict[str, Any]) -> pd.Series:
    """Return mean component scores using the display labels created by results_to_df."""
    df = results_to_df(result.get("recommendations", []))
    component_labels = [
        "Topic Score",
        "Difficulty Match Score",
        "Format Score",
        "Performance Score",
        "Quality Score",
    ]
    return df[component_labels].mean().round(2) if not df.empty else pd.Series(dtype=float)

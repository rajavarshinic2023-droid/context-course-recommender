"""Deterministic, explainable weighted-scoring engine.

Final = 0.30*Topic + 0.20*Difficulty + 0.20*Format + 0.20*Performance + 0.10*Quality
Each component is normalised to 0-100. Personalisation context (LLM or rule-based) only adds
signals; it never replaces this scoring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .learning_logic import DIFFICULTIES, StudentContext, score_band, tokenize
from .resources import Resource, dedupe_resources

WEIGHTS = {"topic": 0.30, "difficulty": 0.20, "format": 0.20, "performance": 0.20, "quality": 0.10}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

MIN_TOPIC_SCORE = 20.0      # below this a resource is treated as not relevant to the topic
EXTRA_TERM_INFLUENCE = 0.30  # how much personalisation keywords can fill the topic gap
LLM_DIFFICULTY_DISCOUNT = 0.90
SUGGESTED_FORMAT_SCORE = 55.0
OTHER_FORMAT_SCORE = 20.0
UNKNOWN_DIFFICULTY_SCORE = 50.0
QUALITY_TIER_SCORE = {"official": 85.0, "established": 75.0, "community": 60.0}
NO_EVIDENCE_QUALITY = 40.0
ENGAGEMENT_SATURATION = 300  # reactions + 2*comments at which live quality saturates at 100

# Performance fit tables per score band: how suitable a difficulty / format is for that band.
PERF_DIFFICULTY = {
    "foundational": {"Beginner": 100, "Intermediate": 50, "Advanced": 10},
    "reinforcement": {"Beginner": 60, "Intermediate": 100, "Advanced": 50},
    "advanced": {"Beginner": 20, "Intermediate": 70, "Advanced": 100},
}
PERF_FORMAT = {
    "foundational": {"Articles": 100, "Video Tutorials": 90, "Online Courses": 90,
                     "Practice Questions": 70, "Documentation": 40},
    "reinforcement": {"Practice Questions": 100, "Articles": 90, "Documentation": 70,
                      "Video Tutorials": 70, "Online Courses": 70},
    "advanced": {"Practice Questions": 100, "Documentation": 100, "Articles": 80,
                 "Online Courses": 70, "Video Tutorials": 50},
}


@dataclass
class ScoredResource:
    resource: Resource
    components: Dict[str, float]
    final_score: float
    explanation: str
    relevant: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"resource": self.resource.to_dict(), "components": self.components,
                "score": self.final_score, "explanation": self.explanation, "relevant": self.relevant}


def _clamp(x: float) -> float:
    return max(0.0, min(100.0, x))


def difficulty_fit(a: str, b: str) -> float:
    return {0: 100.0, 1: 60.0, 2: 20.0}[abs(DIFFICULTIES.index(a) - DIFFICULTIES.index(b))]


def extract_hints(personalization: Optional[Dict[str, Any]]):
    """Pull validated hints from the personalisation dict: (extra_terms, difficulty, formats)."""
    if not personalization:
        return [], None, []
    extra = tokenize(f"{personalization.get('interpreted_topic', '')} "
                     f"{' '.join(personalization.get('search_keywords', []))}")
    diff = personalization.get("suggested_difficulty")
    return extra, diff if diff in DIFFICULTIES else None, list(personalization.get("recommended_formats", []))


def topic_score(res: Resource, primary: List[str], extra: List[str]) -> float:
    title, tags, desc = set(tokenize(res.title)), set(tokenize(" ".join(res.tags))), set(tokenize(res.description))

    def credit(tok: str) -> float:
        return 1.0 if tok in title else 0.7 if tok in tags else 0.4 if tok in desc else 0.0

    if not primary:
        return 0.0
    p = sum(credit(t) for t in primary) / len(primary)
    extras = [t for t in extra if t not in primary]
    e = sum(credit(t) for t in extras) / len(extras) if extras else 0.0
    score = 100 * (p + EXTRA_TERM_INFLUENCE * (1 - p) * e)
    if all(t in title for t in primary):
        score += 10  # whole topic appears in the title
    return _clamp(score)


def difficulty_score(res_diff: Optional[str], requested: str, suggested: Optional[str]) -> float:
    if res_diff is None:
        return UNKNOWN_DIFFICULTY_SCORE
    s = difficulty_fit(res_diff, requested)
    if suggested and suggested != requested:
        s = max(s, LLM_DIFFICULTY_DISCOUNT * difficulty_fit(res_diff, suggested))
    return _clamp(s)


def format_score(res_type: str, preferred, suggested_formats) -> float:
    if res_type in preferred:
        return 100.0
    return SUGGESTED_FORMAT_SCORE if res_type in suggested_formats else OTHER_FORMAT_SCORE


def performance_score(res: Resource, band: str) -> float:
    d = PERF_DIFFICULTY[band].get(res.difficulty, UNKNOWN_DIFFICULTY_SCORE) if res.difficulty else UNKNOWN_DIFFICULTY_SCORE
    return _clamp(0.6 * d + 0.4 * PERF_FORMAT[band].get(res.type, 50))


def quality_score(res: Resource) -> float:
    if res.quality_tier:
        return QUALITY_TIER_SCORE.get(res.quality_tier, QUALITY_TIER_SCORE["community"])
    if res.reactions is not None:
        eng = res.reactions + 2 * (res.comments or 0)
        return _clamp(35 + 65 * min(1.0, math.log10(1 + eng) / math.log10(1 + ENGAGEMENT_SATURATION)))
    return NO_EVIDENCE_QUALITY


def final_score(components: Dict[str, float]) -> float:
    return round(_clamp(sum(WEIGHTS[k] * components[k] for k in WEIGHTS)), 2)


def explain(res: Resource, c: Dict[str, float], ctx: StudentContext, band: str,
            suggested_diff: Optional[str]) -> str:
    parts = []
    t = c["topic"]
    parts.append(("Strong topic match" if t >= 70 else "Partial topic match" if t >= 40 else "Weak topic match")
                 + f" for '{ctx.topic}'")
    if res.difficulty is None:
        parts.append("difficulty unknown (neutral score)")
    elif res.difficulty == ctx.difficulty:
        parts.append(f"{res.difficulty} level matches your request")
    elif suggested_diff and res.difficulty == suggested_diff:
        parts.append(f"{res.difficulty} level matches the personalised suggestion")
    else:
        parts.append(f"{res.difficulty} level differs from your {ctx.difficulty} request")
    if res.type in ctx.preferred_formats:
        parts.append(f"{res.type} is one of your preferred formats")
    elif c["format"] > OTHER_FORMAT_SCORE:
        parts.append(f"{res.type} was suggested by personalisation")
    else:
        parts.append(f"{res.type} is not a preferred format")
    parts.append(f"suits the {band} band for a previous score of {ctx.previous_score:g}")
    if res.quality_tier:
        parts.append(f"{res.quality_tier} source (curated tier)")
    elif res.reactions is not None:
        parts.append(f"{res.reactions} reactions on DEV")
    return "; ".join(parts) + "."


def score_resource(res: Resource, ctx: StudentContext, personalization: Optional[Dict[str, Any]] = None) -> ScoredResource:
    extra, sdiff, sfmts = extract_hints(personalization)
    band = score_band(ctx.previous_score)
    comps = {
        "topic": topic_score(res, tokenize(ctx.topic), extra),
        "difficulty": difficulty_score(res.difficulty, ctx.difficulty, sdiff),
        "format": format_score(res.type, ctx.preferred_formats, sfmts),
        "performance": performance_score(res, band),
        "quality": quality_score(res),
    }
    comps = {k: round(v, 2) for k, v in comps.items()}
    return ScoredResource(res, comps, final_score(comps), explain(res, comps, ctx, band, sdiff),
                          comps["topic"] >= MIN_TOPIC_SCORE)


def rank_resources(resources: List[Resource], ctx: StudentContext,
                   personalization: Optional[Dict[str, Any]] = None) -> List[ScoredResource]:
    """Score all candidates and sort best-first (relevant ones always before irrelevant ones)."""
    if not isinstance(ctx, StudentContext):
        raise TypeError("ctx must be a validated StudentContext (use validate_context).")
    scored = [score_resource(r, ctx, personalization) for r in dedupe_resources(list(resources or []))]
    scored.sort(key=lambda s: (not s.relevant, -s.final_score, s.resource.title.lower()))
    return scored

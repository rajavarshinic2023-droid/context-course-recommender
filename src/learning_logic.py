"""Student-context validation, score bands, tokenisation and the rule-based
(non-LLM) personalisation used as a fallback."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

DIFFICULTIES = ("Beginner", "Intermediate", "Advanced")
FORMATS = ("Video Tutorials", "Articles", "Documentation", "Practice Questions", "Online Courses")
GOALS = (
    "Understand the fundamentals",
    "Improve conceptual understanding",
    "Practise problem-solving",
    "Prepare for an examination",
    "Learn advanced concepts",
)
MAX_TOPIC_LEN = 100
MAX_CONTEXT_LEN = 500

# Score bands (documented in README): <40 foundational, 40-69 reinforcement, >=70 advanced.
BAND_LOW = 40
BAND_HIGH = 70


class ValidationError(ValueError):
    """Raised when student input is invalid. `errors` holds user-friendly messages."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class StudentContext:
    topic: str
    difficulty: str
    preferred_formats: tuple
    previous_score: float
    learning_goal: str
    additional_context: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["preferred_formats"] = list(self.preferred_formats)
        return d


_CTRL = re.compile(r"[\x00-\x1f\x7f]+")


def clean_text(value: Any, max_len: int) -> str:
    if not isinstance(value, str):
        return ""
    return _CTRL.sub(" ", value).strip()[:max_len].strip()


def _canonical(value: Any, allowed):
    if not isinstance(value, str):
        return None
    for item in allowed:
        if item.lower() == value.strip().lower():
            return item
    return None


def validate_context(raw: Dict[str, Any]) -> StudentContext:
    """Validate raw input and return a StudentContext, or raise ValidationError."""
    errors: List[str] = []
    if not isinstance(raw, dict):
        raise ValidationError(["Input must be a set of fields."])

    topic = clean_text(raw.get("topic"), MAX_TOPIC_LEN + 1)
    if not topic:
        errors.append("Please enter a subject or topic.")
    elif len(topic) > MAX_TOPIC_LEN:
        errors.append(f"Topic must be at most {MAX_TOPIC_LEN} characters.")
    elif not re.search(r"[A-Za-z0-9]", topic):
        errors.append("Topic must contain letters or numbers.")

    difficulty = _canonical(raw.get("difficulty"), DIFFICULTIES)
    if difficulty is None:
        errors.append("Difficulty must be Beginner, Intermediate or Advanced.")

    fmts_raw = raw.get("preferred_formats")
    if isinstance(fmts_raw, str):
        fmts_raw = [fmts_raw]
    formats: List[str] = []
    if not isinstance(fmts_raw, (list, tuple)) or not fmts_raw:
        errors.append("Select at least one preferred learning format.")
    else:
        for f in fmts_raw:
            canon = _canonical(f, FORMATS)
            if canon is None:
                errors.append(f"Unknown learning format: {clean_text(str(f), 40)!r}.")
            elif canon not in formats:
                formats.append(canon)

    score = raw.get("previous_score")
    score_val = None
    if isinstance(score, bool) or score is None or score == "":
        errors.append("Previous score must be a number between 0 and 100.")
    else:
        try:
            score_val = float(score)
            if math.isnan(score_val) or math.isinf(score_val) or not 0 <= score_val <= 100:
                raise ValueError
        except (TypeError, ValueError):
            score_val = None
            errors.append("Previous score must be a number between 0 and 100.")

    goal = _canonical(raw.get("learning_goal"), GOALS)
    if goal is None:
        errors.append("Please choose a learning goal from the list.")

    extra_raw = raw.get("additional_context", "")
    if extra_raw is not None and not isinstance(extra_raw, str):
        errors.append("Additional context must be text.")
        extra_raw = ""
    extra = clean_text(extra_raw or "", MAX_CONTEXT_LEN + 1)
    if len(extra) > MAX_CONTEXT_LEN:
        errors.append(f"Additional context must be at most {MAX_CONTEXT_LEN} characters.")

    if errors:
        raise ValidationError(errors)
    return StudentContext(topic, difficulty, tuple(formats), score_val, goal, extra)


def score_band(score: float) -> str:
    if score < BAND_LOW:
        return "foundational"
    if score < BAND_HIGH:
        return "reinforcement"
    return "advanced"


# ---------------------------------------------------------------- tokenisation
STOPWORDS = {
    "a", "an", "and", "the", "of", "to", "in", "on", "for", "with", "how", "what", "is",
    "are", "i", "me", "my", "about", "using", "use", "learn", "basic",
}
SYNONYMS = {
    "ml": ["machine", "learning"],
    "dsa": ["data", "structures", "algorithms"],
    "os": ["operating", "systems"],
    "dbms": ["database"],
    "cn": ["computer", "networks"],
    "oop": ["object", "oriented"],
    "ai": ["artificial", "intelligence"],
    "networking": ["networks"],
    "databases": ["database"],
}


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> List[str]:
    """Lower-case, split, expand abbreviations, drop stop-words, light stemming. Unique, ordered."""
    out: List[str] = []
    for raw in re.findall(r"[a-z0-9]+", (text or "").lower()):
        for tok in SYNONYMS.get(raw, [raw]):
            tok = _stem(tok)
            if tok not in STOPWORDS and tok not in out:
                out.append(tok)
    return out


# ------------------------------------------------------ rule-based personalisation
_BAND_FORMATS = {
    "foundational": ["Articles", "Video Tutorials"],
    "reinforcement": ["Practice Questions", "Articles"],
    "advanced": ["Practice Questions", "Documentation"],
}
_BAND_STRATEGY = {
    "foundational": "Start with clear foundational explanations, then try short guided exercises.",
    "reinforcement": "Reinforce core concepts, then apply them with practice problems.",
    "advanced": "Skim the basics, then work on challenging problems and in-depth references.",
}
_BAND_NEEDS = {
    "foundational": ["Understand the core concepts", "Work through simple guided examples"],
    "reinforcement": ["Consolidate core concepts", "Practise applying them to varied problems"],
    "advanced": ["Deepen understanding of edge cases", "Solve harder problems and read in-depth references"],
}


def rule_based_context(ctx: StudentContext) -> Dict[str, Any]:
    """Deterministic personalisation with the same schema the LLM must return."""
    band = score_band(ctx.previous_score)
    idx = DIFFICULTIES.index(ctx.difficulty)
    if band == "foundational" and idx == 2:
        idx = 1  # an Advanced request with a low score is eased to Intermediate
    elif band == "advanced" and idx == 0:
        idx = 1  # a Beginner request with a high score is raised to Intermediate
    formats = list(ctx.preferred_formats)
    for f in _BAND_FORMATS[band]:
        if f not in formats:
            formats.append(f)
    return {
        "interpreted_topic": ctx.topic,
        "estimated_learning_needs": list(_BAND_NEEDS[band]),
        "suggested_difficulty": DIFFICULTIES[idx],
        "recommended_formats": formats,
        "learning_strategy": _BAND_STRATEGY[band],
        "reasoning_summary": (
            f"Rule-based: a previous score of {ctx.previous_score:g} falls in the {band} band, "
            f"so resources for the goal '{ctx.learning_goal}' are tuned accordingly."
        ),
        "search_keywords": tokenize(ctx.topic)[:5],
        "source": "rule_based",
    }

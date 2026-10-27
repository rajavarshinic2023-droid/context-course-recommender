"""Optional REST service. Reuses src.pipeline (no duplicated recommendation logic).

Run:  uvicorn api:app --reload
"""
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.learning_logic import ValidationError
from src.pipeline import run_recommendation

log = logging.getLogger("api")
app = FastAPI(title="Context-Aware Study Resource Recommender API", version="1.0.0")


class RecommendRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=100, examples=["Python functions"])
    difficulty: str = Field(..., examples=["Beginner"])
    preferred_formats: List[str] = Field(..., min_length=1, examples=[["Articles", "Practice Questions"]])
    previous_score: float = Field(..., ge=0, le=100, examples=[45])
    learning_goal: str = Field(..., examples=["Understand the fundamentals"])
    additional_context: str = Field("", max_length=500)
    top_n: int = Field(8, ge=1, le=20)


class PartnerRecommendRequest(BaseModel):
    """Request schema for the EduPulse partner integration."""
    student_id: Optional[int] = Field(None, ge=1)
    course: str = Field(..., min_length=1, max_length=120)
    topic: str = Field(..., min_length=1, max_length=100)
    difficulty: str = Field(..., examples=["Intermediate"])
    assessment_score: float = Field(..., ge=0, le=100)
    completed_lessons: int = Field(0, ge=0, le=10000)
    learning_progress: float = Field(0, ge=0, le=100)
    preferred_resource_type: str = Field("Mixed", examples=["Video"])
    top_n: int = Field(8, ge=1, le=20)


def _partner_formats(value: str) -> List[str]:
    """Translate partner format labels into the formats supported by StudyCompass."""
    aliases = {
        "video": ["Video Tutorials"],
        "article": ["Articles"],
        "articles": ["Articles"],
        "practice": ["Practice Questions"],
        "practice questions": ["Practice Questions"],
        "documentation": ["Documentation"],
        "course": ["Online Courses"],
        "courses": ["Online Courses"],
        "online course": ["Online Courses"],
        "online courses": ["Online Courses"],
        "mixed": ["Video Tutorials", "Articles", "Practice Questions", "Documentation", "Online Courses"],
    }
    result = aliases.get((value or "Mixed").strip().lower())
    if result is None:
        raise HTTPException(status_code=422, detail="preferred_resource_type must be Video, Article, Practice, Documentation, Course, or Mixed.")
    return result


def _partner_difficulty(value: str) -> str:
    """Normalize EduPulse's Easy/Medium/Hard labels to StudyCompass levels."""
    aliases = {
        "easy": "Beginner",
        "beginner": "Beginner",
        "medium": "Intermediate",
        "intermediate": "Intermediate",
        "hard": "Advanced",
        "advanced": "Advanced",
    }
    normalized = aliases.get((value or "").strip().lower())
    if normalized is None:
        raise HTTPException(
            status_code=422,
            detail="difficulty must be Easy, Medium, Hard, Beginner, Intermediate, or Advanced.",
        )
    return normalized


def _partner_learning_goal(score: float) -> str:
    if score < 40:
        return "Understand the fundamentals"
    if score < 70:
        return "Improve conceptual understanding"
    return "Learn advanced concepts"


def _partner_response(result: dict) -> dict:
    """Expose a compact stable response schema for EduPulse clients."""
    recommendations = []
    for item in result.get("recommendations", []):
        metadata = item.get("metadata") or {}
        recommendations.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "resource_type": item.get("type", "Article"),
            "difficulty": metadata.get("difficulty") or "Unknown",
            "score": item.get("score", 0),
            "explanation": item.get("explanation", ""),
        })
    return {
        "recommendations": recommendations,
        "status": result.get("ranking", {}).get("status", "unknown"),
        "message": result.get("message") or result.get("ranking", {}).get("message", ""),
        "count": len(recommendations),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    payload = req.model_dump(exclude={"top_n"})
    try:
        result = run_recommendation(payload, top_n=req.top_n)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors)
    except Exception as exc:  # never leak internals
        log.error("Recommendation failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Internal error while generating recommendations.")
    result.pop("all_scored", None)
    return result


@app.post("/api/recommendations")
def partner_recommend(req: PartnerRecommendRequest):
    """Partner-facing endpoint for EduPulse. Existing /recommend remains unchanged."""
    formats = _partner_formats(req.preferred_resource_type)
    context_parts = [f"Course: {req.course}", f"Completed lessons: {req.completed_lessons}",
                     f"Learning progress: {req.learning_progress:g}%"]
    if req.student_id is not None:
        context_parts.append(f"Partner student ID: {req.student_id}")
    payload = {
        "topic": req.topic,
        "difficulty": _partner_difficulty(req.difficulty),
        "preferred_formats": formats,
        "previous_score": req.assessment_score,
        "learning_goal": _partner_learning_goal(req.assessment_score),
        "additional_context": "; ".join(context_parts),
    }
    try:
        result = run_recommendation(payload, top_n=req.top_n)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors)
    except Exception as exc:
        log.error("Partner recommendation failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Internal error while generating recommendations.")
    # Never imply success if Gemini ranking is unavailable.
    response = _partner_response(result)
    response["requested_count"] = req.top_n
    response["preferred_resource_type"] = req.preferred_resource_type
    response["preferred_format_count"] = sum(
        1 for item in response["recommendations"]
        if item["resource_type"] in formats
    )
    retrieval = result.get("retrieval", {})
    response["format_fallback_used"] = bool(retrieval.get("format_fallback_used", False))
    response["preferred_format_candidates_available"] = int(
        retrieval.get("preferred_format_candidate_count", 0)
    )
    if response["format_fallback_used"] and response["status"] == "llm_used":
        response["message"] = result.get("message") or (
            "Not enough relevant resources were available in the requested format; relevant alternatives were used."
        )
    elif response["status"] == "llm_used":
        response["message"] = "All returned recommendations were selected from the requested format(s)."
    if response["status"] == "llm_unavailable":
        raise HTTPException(status_code=503, detail={"message": response["message"], "status": response["status"], "recommendations": []})
    return response

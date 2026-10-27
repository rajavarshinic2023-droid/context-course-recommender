from src.learning_logic import validate_context
from src.resources import Resource


def make_ctx(**kw):
    base = dict(topic="Python functions", difficulty="Beginner",
                preferred_formats=["Articles", "Practice Questions"], previous_score=45,
                learning_goal="Understand the fundamentals", additional_context="")
    base.update(kw)
    return validate_context(base)


def make_res(title="Python functions guide", type="Articles", difficulty="Beginner", tags=("python", "functions"),
             description="", origin="local", tier="established", url=None, **kw):
    return Resource(title=title, url=url or f"https://example.com/{abs(hash((title, type, difficulty)))}",
                    type=type, source="Test", origin=origin, description=description, tags=list(tags),
                    difficulty=difficulty, difficulty_source="provided" if difficulty else "unknown",
                    quality_tier=tier, **kw)

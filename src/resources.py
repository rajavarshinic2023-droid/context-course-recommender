"""Common resource model, URL validation, de-duplication and the local resource library."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .learning_logic import DIFFICULTIES, FORMATS

log = logging.getLogger(__name__)
LOCAL_LIBRARY_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_resources.json"


@dataclass
class Resource:
    title: str
    url: str
    type: str                      # one of FORMATS
    source: str
    origin: str                    # "live" (DEV API), "local" (library) or "partner"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    difficulty: Optional[str] = None
    difficulty_source: str = "unknown"   # provided | inferred | unknown
    published_at: Optional[str] = None
    reactions: Optional[int] = None
    comments: Optional[int] = None
    reading_time_min: Optional[int] = None
    quality_tier: Optional[str] = None   # local library only: official | established | community

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def is_valid_url(url: Any) -> bool:
    if not isinstance(url, str) or not url.strip() or len(url) > 2048 or any(c.isspace() for c in url.strip()):
        return False
    p = urlparse(url.strip())
    return p.scheme in ("http", "https") and bool(p.netloc) and "." in p.netloc


def canonical_key(url: str) -> str:
    p = urlparse(url.strip())
    q = f"?{p.query}" if p.query else ""
    return f"{p.netloc.lower()}{p.path.rstrip('/')}{q}"


def dedupe_resources(resources: List[Resource]) -> List[Resource]:
    seen, out = set(), []
    for r in resources:
        k = canonical_key(r.url)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def load_local_resources(path=None) -> List[Resource]:
    """Load the curated local library. Invalid entries are skipped, never fabricated."""
    path = Path(path) if path else LOCAL_LIBRARY_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Local resource library unavailable: %s", type(exc).__name__)
        return []
    out: List[Resource] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in FORMATS or not is_valid_url(item.get("url")) or not item.get("title"):
            log.warning("Skipping invalid local resource: %r", item.get("title"))
            continue
        diff = item.get("difficulty")
        out.append(Resource(
            title=str(item["title"]), url=item["url"].strip(), type=item["type"],
            source=str(item.get("source", "Local library")), origin="local",
            description=str(item.get("description", "")),
            tags=[str(t) for t in item.get("tags", [])],
            difficulty=diff if diff in DIFFICULTIES else None,
            difficulty_source="provided" if diff in DIFFICULTIES else "unknown",
            quality_tier=item.get("quality_tier"),
        ))
    return dedupe_resources(out)

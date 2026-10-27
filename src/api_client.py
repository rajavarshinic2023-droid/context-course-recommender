"""DEV Community (Forem) API client. Retrieves real articles and normalises them to Resource."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

from .learning_logic import STOPWORDS, StudentContext, tokenize
from .resources import Resource, dedupe_resources, is_valid_url

log = logging.getLogger(__name__)
DEV_ARTICLES_URL = "https://dev.to/api/articles"
HEADERS = {"Accept": "application/vnd.forem.api-v1+json", "User-Agent": "study-recommender/1.0"}
MAX_TAG_REQUESTS = 3
PER_PAGE = 30

BEGINNER_WORDS = ("beginner", "introduction", "intro to", "getting started", "basics", "101",
                  "fundamentals", "crash course", "for dummies", "first steps")
ADVANCED_WORDS = ("advanced", "deep dive", "under the hood", "internals", "expert", "in depth",
                  "in-depth", "architecture", "optimization", "optimisation")


@dataclass
class FetchResult:
    resources: List[Resource] = field(default_factory=list)
    status: str = "empty"          # live | empty | error
    message: str = ""
    tags_queried: List[str] = field(default_factory=list)


def derive_tags(ctx: StudentContext, extra_keywords: Optional[List[str]] = None) -> List[str]:
    """DEV's /articles endpoint filters by tag (single word, no spaces), so derive tag candidates."""
    words = [w for w in re.findall(r"[a-z0-9]+", ctx.topic.lower()) if w not in STOPWORDS and len(w) > 1]
    cands: List[str] = []
    if len(words) > 1 and len("".join(words)) <= 30:
        cands.append("".join(words))
    cands.extend(words)
    cands.extend(extra_keywords or [])
    out: List[str] = []
    for c in cands:
        if c and c not in out and re.fullmatch(r"[a-z0-9]+", c):
            out.append(c)
    return out[:MAX_TAG_REQUESTS]


def infer_difficulty(title: str, tags: List[str]) -> Optional[str]:
    text = f"{title} {' '.join(tags)}".lower()
    if any(w in text for w in ADVANCED_WORDS):
        return "Advanced"
    if any(w in text for w in BEGINNER_WORDS):
        return "Beginner"
    return None


def _int_or_none(v: Any) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def normalize_article(item: Any) -> Optional[Resource]:
    """Convert one DEV article object to a Resource, or None if unusable."""
    if not isinstance(item, dict):
        return None
    title = item.get("title")
    url = item.get("url") or item.get("canonical_url")
    if not isinstance(title, str) or not title.strip() or not is_valid_url(url):
        return None
    tags = item.get("tag_list")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    if not isinstance(tags, list):
        tags = item.get("tags") if isinstance(item.get("tags"), list) else \
            [t.strip() for t in str(item.get("tags") or "").split(",")]
    tags = [str(t).strip() for t in tags if str(t).strip()]
    user = item.get("user") if isinstance(item.get("user"), dict) else {}
    author = user.get("name") or user.get("username")
    diff = infer_difficulty(title, tags)
    desc = item.get("description")
    return Resource(
        title=title.strip(), url=url.strip(), type="Articles",
        source=f"DEV Community{f' - {author}' if author else ''}", origin="live",
        description=desc.strip() if isinstance(desc, str) else "", tags=tags,
        difficulty=diff, difficulty_source="inferred" if diff else "unknown",
        published_at=item.get("published_at") if isinstance(item.get("published_at"), str) else None,
        reactions=_int_or_none(item.get("public_reactions_count")),
        comments=_int_or_none(item.get("comments_count")),
        reading_time_min=_int_or_none(item.get("reading_time_minutes")),
    )


def fetch_dev_articles(tags: List[str], http_get: Callable = requests.get,
                       timeout: float = 8.0) -> FetchResult:
    """Query DEV once per tag. Never raises; reports status/message instead."""
    result = FetchResult(tags_queried=list(tags))
    if not tags:
        result.message = "No search tags could be derived from the topic."
        return result
    collected: List[Resource] = []
    ok = failed = 0
    for tag in tags:
        try:
            resp = http_get(DEV_ARTICLES_URL, params={"tag": tag, "per_page": PER_PAGE, "top": 365},
                            headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError("unexpected response shape")
        except requests.Timeout:
            failed += 1
            log.warning("DEV API timeout for tag %s", tag)
            continue
        except (requests.RequestException, ValueError) as exc:
            failed += 1
            log.warning("DEV API failure for tag %s: %s", tag, type(exc).__name__)
            continue
        ok += 1
        collected.extend(r for r in map(normalize_article, data) if r)
    collected = dedupe_resources(collected)
    if collected:
        result.resources, result.status = collected, "live"
        result.message = f"Retrieved {len(collected)} live article(s) from the DEV Community API."
        if failed:
            result.message += f" ({failed} of {len(tags)} tag request(s) failed.)"
    elif ok:
        result.status, result.message = "empty", "The DEV Community API returned no usable articles for this topic."
    else:
        result.status, result.message = "error", "The DEV Community API could not be reached or returned invalid data."
    return result


def fetch_wikipedia_articles(topic: str, http_get: Callable = requests.get,
                             timeout: float = 6.0, limit: int = 8) -> FetchResult:
    """Search Wikipedia for broad subject coverage when DEV has no matching articles.

    This is a live search, not a guarantee that every result is a course/tutorial.
    Results are labelled as reference articles so students can explore unfamiliar topics.
    """
    result = FetchResult()
    query = (topic or "").strip()
    if not query:
        result.message = "No topic supplied for Wikipedia search."
        return result
    try:
        response = http_get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "generator": "search", "gsrsearch": query,
                    "gsrnamespace": 0, "gsrlimit": max(1, min(int(limit), 10)),
                    "prop": "extracts|info", "exintro": 1, "explaintext": 1,
                    "inprop": "url", "format": "json"},
            headers={"User-Agent": "StudyCompass/1.0 (educational resource recommender)"},
            timeout=timeout)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {}) if isinstance(data, dict) else {}
        resources = []
        for page in pages.values() if isinstance(pages, dict) else []:
            title = page.get("title")
            url = page.get("fullurl") or ("https://en.wikipedia.org/?curid=" + str(page.get("pageid")))
            if not isinstance(title, str) or not title.strip() or not is_valid_url(url):
                continue
            extract = page.get("extract") if isinstance(page.get("extract"), str) else ""
            tags = [w for w in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(w) > 2][:10]
            resources.append(Resource(title=title.strip(), url=url, type="Articles",
                source="Wikipedia reference", origin="live", description=extract[:700],
                tags=tags + [w.lower() for w in title.split() if w.isalpha()],
                difficulty=None, difficulty_source="unknown", quality_tier="established"))
        result.resources = dedupe_resources(resources)
        result.status = "live" if result.resources else "empty"
        result.message = (f"Found {len(result.resources)} reference article(s) for this subject."
                          if result.resources else "Wikipedia returned no matching reference articles.")
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        log.warning("Wikipedia search failed: %s", type(exc).__name__)
        result.status = "error"
        result.message = "Wikipedia search is currently unavailable."
    return result

"""Single orchestration path used by BOTH the Streamlit app and the FastAPI service."""
from __future__ import annotations

import logging
import os
import re
from urllib.parse import quote_plus

import requests
from typing import Any, Callable, Dict, List, Optional
from dataclasses import replace

from . import api_client, llm_service
from .learning_logic import validate_context
from .partner_client import PartnerClient, PartnerResult
from .recommender import WEIGHTS, rank_resources, score_resource
from .resources import Resource, dedupe_resources, load_local_resources

log = logging.getLogger(__name__)


def _rec_dict(rank: int, s) -> Dict[str, Any]:
    r = s.resource
    return {
        "rank": rank, "title": r.title, "url": r.url, "type": r.type, "source": r.source,
        "origin": r.origin, "score": s.final_score, "components": s.components,
        "explanation": s.explanation, "relevant": s.relevant,
        "metadata": {"description": r.description, "tags": r.tags, "difficulty": r.difficulty,
                     "difficulty_source": r.difficulty_source, "published_at": r.published_at,
                     "reactions": r.reactions, "comments": r.comments,
                     "reading_time_min": r.reading_time_min, "quality_tier": r.quality_tier},
    }


def _estimate_difficulty(title: str, description: str, requested: str) -> tuple[str, str]:
    """Estimate resource difficulty from its public metadata; never leave it Unknown.

    This is a transparent heuristic, not an authoritative level supplied by YouTube.
    Strong title/description cues win; otherwise the learner's requested level is used
    as a neutral estimate so the API has a usable difficulty value.
    """
    text = f"{title} {description}".lower()
    beginner_cues = ("for beginners", "beginner", "absolute beginner", "getting started",
                     "introduction to", "basics", "fundamentals", "from scratch", "for dummies")
    advanced_cues = ("advanced", "expert", "deep dive", "internals", "optimization",
                     "metaprogramming", "decorator", "decorators", "lambda", "closures",
                     "recursion", "λ-function", "higher-order")
    intermediate_cues = ("intermediate", "practical examples", "exercises", "problem solving",
                         "practice problems", "arguments", "parameters", "return values", "scope")
    if any(cue in text for cue in beginner_cues):
        return "Beginner", "inferred_from_metadata"
    if any(cue in text for cue in advanced_cues):
        return "Advanced", "inferred_from_metadata"
    if any(cue in text for cue in intermediate_cues):
        return "Intermediate", "inferred_from_metadata"
    normal = {"easy": "Beginner", "medium": "Intermediate", "hard": "Advanced"}
    requested = normal.get(str(requested).strip().lower(), str(requested).strip().title())
    if requested not in {"Beginner", "Intermediate", "Advanced"}:
        requested = "Intermediate"
    return requested, "estimated_from_learner_level"


def _fill_missing_difficulty(resources: List[Resource], requested: str) -> List[Resource]:
    """Fill missing levels consistently while preserving explicit source-provided levels."""
    out = []
    for resource in resources:
        if resource.difficulty and str(resource.difficulty).lower() not in {"unknown", "none", "n/a"}:
            out.append(resource)
            continue
        level, source = _estimate_difficulty(resource.title, resource.description, requested)
        out.append(replace(resource, difficulty=level, difficulty_source=source))
    return out


def _topic_search_resources(topic: str, preferred_formats: tuple, additional_context: str = "", requested_difficulty: str = "Intermediate") -> List[Resource]:
    """Retrieve real YouTube videos when an API key is configured.

    Without a YouTube Data API key, add only one clearly labelled search page as a
    transparent fallback. Never manufacture several search URLs and count them as
    individual video recommendations.
    """
    if "Video Tutorials" not in preferred_formats:
        return []
    topic_clean = " ".join(topic.split()).strip()
    if not topic_clean:
        return []

    course_match = re.search(r"Course:\s*([^;]+)", additional_context, flags=re.IGNORECASE)
    course = course_match.group(1).strip() if course_match else ""
    query = " ".join(part for part in (course, topic_clean, "tutorial examples") if part)
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if api_key and api_key.lower() not in {"your-youtube-api-key-here", "changeme"}:
        try:
            response = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={"part": "snippet", "type": "video", "maxResults": 25,
                        "q": query, "key": api_key, "safeSearch": "moderate"},
                timeout=8,
            )
            response.raise_for_status()
            payload = response.json()
            out: List[Resource] = []
            seen_ids = set()
            topic_tokens = {w.lower() for w in re.findall(r"[a-z0-9]+", f"{course} {topic_clean}") if len(w) > 2}
            for item in payload.get("items", []):
                ident = (item.get("id") or {}).get("videoId")
                snippet = item.get("snippet") or {}
                title = str(snippet.get("title") or "").strip()
                description = str(snippet.get("description") or "").strip()
                if not ident or ident in seen_ids or not title:
                    continue
                seen_ids.add(ident)
                searchable = f"{title} {description}".lower()
                # Require both the requested topic and the course context when the
                # course contains a meaningful subject term (e.g. Python in Python Programming).
                topic_words = {w.lower() for w in re.findall(r"[a-z0-9]+", topic_clean) if len(w) > 2}
                generic_course_words = {"programming", "course", "subject", "basics", "fundamentals", "introduction", "intro", "and", "the", "of", "for"}
                course_words = {w.lower() for w in re.findall(r"[a-z0-9]+", course) if len(w) > 2} - generic_course_words
                if topic_words and not any(w in searchable for w in topic_words):
                    continue
                if course_words and not any(w in searchable for w in course_words):
                    continue
                out.append(Resource(
                    title=title, url=f"https://www.youtube.com/watch?v={ident}",
                    type="Video Tutorials", source="YouTube Data API", origin="live",
                    description=description or f"YouTube video about {query}.",
                    tags=list(topic_tokens) + ["video", "tutorial"],
                    difficulty=_estimate_difficulty(title, description, requested_difficulty)[0],
                    difficulty_source=_estimate_difficulty(title, description, requested_difficulty)[1], quality_tier="community",
                ))
            if out:
                return out
            log.info("YouTube API returned no topic-matching videos for %r", query)
        except Exception as exc:
            log.warning("YouTube video retrieval unavailable: %s", type(exc).__name__)

    # Honest fallback: a single discovery page, not multiple fake individual videos.
    return [Resource(
        title=f"YouTube search: {query}",
        url="https://www.youtube.com/results?search_query=" + quote_plus(query),
        type="Video Tutorials", source="YouTube (search results)", origin="local",
        description=(f"Topic-specific YouTube search results for {query}. "
                     "This opens a search page, not a verified individual video."),
        tags=topic_clean.lower().split() + course.lower().split() + ["video", "tutorial"],
        difficulty=_estimate_difficulty(f"YouTube search: {query}", query, requested_difficulty)[0],
        difficulty_source=_estimate_difficulty(f"YouTube search: {query}", query, requested_difficulty)[1], quality_tier="community",
    )]


def run_recommendation(raw_input: Dict[str, Any], *, top_n: int = 8,
                       personalizer: Optional[Callable] = None,
                       fetcher: Optional[Callable] = None,
                       partner_client: Optional[PartnerClient] = None,
                       local_loader: Optional[Callable] = None) -> Dict[str, Any]:
    """validate -> personalise -> retrieve (DEV + Wikipedia + local + partner) -> Gemini-rank -> explain.

    Raises ValidationError for bad input; every external failure is converted to a status."""
    ctx = validate_context(raw_input)
    personalization = (personalizer or llm_service.personalize)(ctx)
    p_ctx = personalization.context

    fetch = (fetcher or api_client.fetch_dev_articles)
    tags = api_client.derive_tags(ctx, p_ctx.get("search_keywords"))
    live = fetch(tags)
    # Broaden coverage beyond developer-focused topics. Wikipedia is a general-purpose
    # reference source and can return articles for science, humanities, business, etc.
    wiki = api_client.fetch_wikipedia_articles(ctx.topic)
    local = (local_loader or load_local_resources)()
    partner = (partner_client or PartnerClient()).exchange(ctx)

    live_resources = dedupe_resources(list(live.resources) + list(wiki.resources))
    generated_searches = _topic_search_resources(ctx.topic, ctx.preferred_formats, ctx.additional_context, ctx.difficulty)
    # If the primary live article source fails, make the fallback pool genuinely local.
    # Do not accidentally label Wikipedia/partner/live resources as local fallback results.
    if live.status != "live":
        fallback_pool = list(local) + list(generated_searches)
        candidates = _fill_missing_difficulty(dedupe_resources(fallback_pool), ctx.difficulty)
    else:
        candidates = _fill_missing_difficulty(
            dedupe_resources(live_resources + list(local) + list(partner.resources) + generated_searches),
            ctx.difficulty,
        )
    # Course relevance is a separate guard from topic relevance. For a request such as
    # course="Python Programming", topic="Functions", a generic JavaScript or math
    # functions resource should not pass merely because it mentions "functions".
    course_match = re.search(r"Course:\s*([^;]+)", ctx.additional_context, flags=re.IGNORECASE)
    course_name = course_match.group(1).strip() if course_match else ""
    generic_course_words = {"programming", "course", "subject", "basics", "fundamentals", "introduction", "intro", "and", "the", "of", "for"}
    course_terms = {w.lower() for w in re.findall(r"[a-z0-9]+", course_name) if len(w) > 2} - generic_course_words
    if course_terms:
        course_relevant = []
        for candidate in candidates:
            searchable = f"{candidate.title} {candidate.description} {' '.join(candidate.tags)}".lower()
            # For live YouTube results, don't trust the tags added by our own query;
            # the actual title/description must demonstrate course relevance.
            if candidate.source == "YouTube Data API":
                searchable = f"{candidate.title} {candidate.description}".lower()
            if any(term in searchable for term in course_terms):
                course_relevant.append(candidate)
        candidates = course_relevant
    ranking_status = "llm_used"
    ranking_message = "Recommendations ranked by Gemini using the learner profile and retrieved resource details."
    ranked = []
    ranking_candidates = []
    preferred_candidates = []
    format_fallback_used = False
    try:
        # Course relevance, topic relevance, and requested format are hard constraints
        # for the selected pool. Difficulty is the only preference we relax. If sources
        # cannot supply enough matching resources, return fewer rather than inventing filler.
        relevance_floor = 42.0
        scored_candidates = [
            (i, candidate, score_resource(candidate, ctx, p_ctx).components["topic"])
            for i, candidate in enumerate(candidates)
        ]
        format_priority = {fmt: i for i, fmt in enumerate(ctx.preferred_formats)}
        difficulty_priority = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
        requested_diff_index = difficulty_priority[ctx.difficulty]
        def candidate_key(item):
            i, candidate, topic_match = item
            return (
                format_priority.get(candidate.type, len(format_priority)),
                -topic_match,
                abs(difficulty_priority.get(candidate.difficulty, requested_diff_index) - requested_diff_index),
                i,
            )

        indexed_candidates = sorted(scored_candidates, key=candidate_key)
        relevant_candidates = [item for item in indexed_candidates if item[2] >= relevance_floor]
        preferred_candidates = [item for item in relevant_candidates
                                if item[1].type in ctx.preferred_formats]
        # Keep the requested format strict. If there are too few resources, return
        # fewer rather than silently switching formats or including unrelated content.
        format_fallback_used = len(preferred_candidates) < top_n
        topic_fallback_used = False
        selected_candidates = preferred_candidates if ctx.preferred_formats else relevant_candidates
        selected_candidates = sorted(selected_candidates, key=candidate_key)
        # When live retrieval succeeds, include a relevant local-library result near the
        # front when one exists, so the recommender can blend live and curated material.
        # The format/topic filters above still apply.
        if live.status == "live" and selected_candidates:
            local_match = next((item for item in selected_candidates if item[1].origin == "local"), None)
            if local_match is not None and local_match not in selected_candidates[:3]:
                selected_candidates = [item for item in selected_candidates if item is not local_match]
                selected_candidates.insert(min(2, len(selected_candidates)), local_match)
        ranking_candidates = [item[1] for item in selected_candidates[:40]]
        # Keep the stable public function as the orchestration seam. This also lets
        # tests and integrations mock ranking without depending on provider internals.
        llm_rows = llm_service.rank_candidates(
            ctx, p_ctx,
            [{"title": r.title, "url": r.url, "type": r.type, "source": r.source,
              "description": r.description, "tags": r.tags, "difficulty": r.difficulty}
             for r in ranking_candidates],
            top_n=top_n,
        )
        ranking_provider = getattr(llm_service, "LAST_RANKING_PROVIDER", "unknown")
        if ranking_provider == "unknown":
            # Compatibility for tests/integrations that mock rank_candidates directly.
            ranking_provider = "Configured"
        ranking_status = "llm_used"
        ranking_message = (
            f"Recommendations ranked successfully by {ranking_provider} using the learner profile "
            "and retrieved resource details."
        )
        for row in llm_rows:
            candidate = ranking_candidates[row["candidate_index"]]
            diagnostics = score_resource(candidate, ctx, p_ctx)
            # The selected LLM supplies the ranking score and learner-specific explanation.
            ranked.append(replace(diagnostics, final_score=row["score"],
                                  explanation=row["explanation"],
                                  relevant=diagnostics.components["topic"] >= relevance_floor))
    except Exception as exc:
        log.warning("LLM recommendation ranking failed (%s).", type(exc).__name__)

        has_llm_key = bool(
            llm_service.get_setting("GEMINI_API_KEY")
            or llm_service.get_setting("GROQ_API_KEY")
        )

        if has_llm_key:
            # An LLM was configured, but both configured providers failed.
            ranking_status = "python_fallback"
            ranking_message = (
                "Gemini and Groq ranking failed, so the transparent "
                "local relevance scorer was used as a last-resort fallback."
            )

            fallback_scored = [
                score_resource(candidate, ctx, p_ctx)
                for candidate in ranking_candidates
            ]

            ranked.extend(
                sorted(
                    fallback_scored,
                    key=lambda item: item.final_score,
                    reverse=True,
                )[:max(1, top_n)]
            )
        else:
            # Neither LLM provider is configured.
            ranking_status = "llm_unavailable"
            ranking_message = (
                "No ranking LLM is configured. "
                "No rule-based recommendations were substituted. "
                "Set GEMINI_API_KEY or GROQ_API_KEY to enable LLM ranking."
            )
    relevant = ranked
    top = relevant[:max(1, top_n)]
    # The local library is part of the candidate pool; flag fallback whenever the primary
    # live developer-article retrieval failed, even if Wikipedia still returned results.
    fallback_used = live.status != "live"
    live_count = len(live_resources)
    if live.status != "live":
        retrieval_msg = (
            "Live developer-article search failed. The local resource library is used as a fallback, "
            "along with any available Wikipedia or connected partner resources."
        )
    elif live_count:
        retrieval_msg = (f"Retrieved {len(live.resources)} developer article(s) and "
                         f"{len(wiki.resources)} general reference article(s) when available. "
                         "The local library supplements these results with curated resources.")
    else:
        retrieval_msg = ("Live searches did not return usable resources. The candidate pool contains "
                         "the local resource library and any connected partner source.")
    message = "" if top else (
        ranking_message if ranking_status != "llm_used" else
        "Gemini did not identify sufficiently relevant resources among the retrieved candidates. Try a broader topic."
    )
    if ranking_status in {"llm_used", "python_fallback"} and top and format_fallback_used:
        available_preferred = sum(1 for item in top if item.resource.type in ctx.preferred_formats)
        message = (
            f"Found {len(preferred_candidates)} topic-relevant resource(s) in the requested format(s); "
            f"returned {available_preferred} preferred-format resource(s). The requested format and topic "
            "were kept strict; difficulty was treated as a preference. The result count may be below the "
            "request when the configured sources cannot provide enough matching resources."
        )
    elif ranking_status in {"llm_used", "python_fallback"} and top and len(top) < top_n:
        message = (
            f"Returned {len(top)} topic-relevant resource(s) out of {top_n} requested. "
            "The available sources did not provide enough resources matching both topic and preferred format; "
            "unrelated resources were excluded. Difficulty was treated as a preference."
        )

    # Resource Explorer needs the full evaluated candidate pool, not only the
    # small top-N list returned by the LLM. Score every candidate for display,
    # then preserve any LLM scores/explanations for resources that were ranked.
    all_scored_items = [score_resource(candidate, ctx, p_ctx) for candidate in candidates]
    ranked_by_url = {item.resource.url: item for item in ranked}
    all_scored_items = [
        ranked_by_url.get(item.resource.url, item) for item in all_scored_items
    ]
    all_scored_items.sort(key=lambda item: item.final_score, reverse=True)
    all_scored_dicts = [
        _rec_dict(index, item) for index, item in enumerate(all_scored_items, 1)
    ]

    return {
        "input": ctx.as_dict(),
        "personalization": personalization.to_dict(),
        "ranking": {"status": ranking_status, "message": ranking_message, "engine": ((ranking_provider + " LLM") if ranking_status == "llm_used" else ("Python relevance scorer" if ranking_status == "python_fallback" else "LLM unavailable"))},
        "retrieval": {"live_status": live.status, "live_count": live_count,
                      "dev_count": len(live.resources), "wikipedia_status": wiki.status,
                      "wikipedia_count": len(wiki.resources), "local_count": len(local),
                      "fallback_used": fallback_used, "format_fallback_used": format_fallback_used,
                      "topic_fallback_used": locals().get("topic_fallback_used", False),
                      "preferred_format_candidate_count": len(preferred_candidates), "message": retrieval_msg,
                      "tags_queried": live.tags_queried},
        "partner": partner.to_dict(),
        "weights": dict(WEIGHTS),
        "candidates_evaluated": len(all_scored_dicts),
        "recommendations": [_rec_dict(i, s) for i, s in enumerate(top, 1)],
        "all_scored": all_scored_dicts,
        "message": message,
    }

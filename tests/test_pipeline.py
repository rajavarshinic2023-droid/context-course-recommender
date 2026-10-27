import json
import unittest

from src import llm_service as L
from src.api_client import FetchResult
from src.learning_logic import ValidationError
from src.partner_client import PartnerClient
from src.pipeline import run_recommendation
from src.resources import Resource
from tests.helpers import make_ctx  # noqa: F401

RAW = {"topic": "Python functions", "difficulty": "Beginner", "preferred_formats": ["Articles", "Practice Questions"],
       "previous_score": 45, "learning_goal": "Understand the fundamentals"}
LLM_JSON = json.dumps({"interpreted_topic": "Python functions", "estimated_learning_needs": ["Define functions"],
                       "suggested_difficulty": "Beginner", "recommended_formats": ["Articles"],
                       "learning_strategy": "Basics first", "reasoning_summary": "Foundations.", "search_keywords": ["python"]})
LIVE = Resource("Python Functions Explained", "https://dev.to/x/functions", "Articles", "DEV Community", "live",
                tags=["python", "functions"], difficulty="Beginner", difficulty_source="inferred", reactions=50)


def live_ok(tags):
    return FetchResult([LIVE], "live", "Retrieved 1 live article(s) from the DEV Community API.", tags)


def live_fail(tags):
    return FetchResult([], "error", "down", tags)


def pz_llm(ctx):
    return L.personalize(ctx, call_llm=lambda p: LLM_JSON)


def pz_off(ctx):
    return L.personalize(ctx, call_llm=lambda p: (_ for _ in ()).throw(TimeoutError()))


NO_PARTNER = PartnerClient(base_url="")


class PipelineTests(unittest.TestCase):
    def run_pipe(self, fetcher, personalizer, raw=RAW):
        def mock_llm_rank(ctx, personalization, candidates, top_n=8):
            if "Medieval basket weaving" in ctx.topic:
                return []
            return [{"candidate_index": i, "score": 95 - i * 5,
                     "explanation": "Mock Gemini ranking for test coverage."}
                    for i in range(min(3, len(candidates), top_n))]
        from unittest.mock import patch
        with patch("src.llm_service.rank_candidates", side_effect=mock_llm_rank):
            return run_recommendation(raw, fetcher=fetcher, personalizer=personalizer, partner_client=NO_PARTNER)

    def test_live_and_llm_flow(self):
        r = self.run_pipe(live_ok, pz_llm)
        self.assertTrue(r["personalization"]["used_llm"])
        self.assertFalse(r["retrieval"]["fallback_used"])
        origins = {x["origin"] for x in r["recommendations"]}
        self.assertIn("live", origins)
        self.assertIn("local", origins)
        self.assertEqual([x["rank"] for x in r["recommendations"]], list(range(1, len(r["recommendations"]) + 1)))
        scores = [x["score"] for x in r["recommendations"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_api_failure_triggers_local_fallback(self):
        r = self.run_pipe(live_fail, pz_llm)
        self.assertTrue(r["retrieval"]["fallback_used"])
        self.assertIn("local resource library", r["retrieval"]["message"])
        self.assertTrue(r["recommendations"])
        self.assertTrue(all(x["origin"] == "local" for x in r["recommendations"]))

    def test_works_without_llm(self):
        r = self.run_pipe(live_ok, pz_off)
        self.assertFalse(r["personalization"]["used_llm"])
        self.assertEqual(r["personalization"]["source"], "rule_based")
        self.assertTrue(r["recommendations"])

    def test_invalid_input_rejected_before_any_call(self):
        called = []
        with self.assertRaises(ValidationError):
            run_recommendation(dict(RAW, topic=""), fetcher=lambda t: called.append(1), personalizer=pz_off)
        self.assertEqual(called, [])

    def test_unknown_topic_gives_message_not_junk(self):
        r = self.run_pipe(live_fail, pz_off, dict(RAW, topic="Medieval basket weaving"))
        self.assertEqual(r["recommendations"], [])
        self.assertTrue(r["message"])

    def test_json_serialisable(self):
        json.dumps(self.run_pipe(live_ok, pz_llm))


    def test_unrelated_resources_are_excluded_even_when_count_is_short(self):
        from unittest.mock import patch
        unrelated = Resource(
            "Operating Systems Full Course", "https://example.org/os",
            "Video Tutorials", "YouTube", "local", tags=["operating systems"],
            description="Operating systems course", difficulty="Beginner"
        )
        relevant = Resource(
            "Python Functions Tutorial", "https://example.org/python-functions",
            "Video Tutorials", "YouTube", "local", tags=["python", "functions", "arguments", "return"],
            description="Explains Python functions, parameters, and return values.", difficulty="Intermediate"
        )
        captured = {}
        def fetch(tags):
            return FetchResult([], "error", "offline", tags)
        def rank(ctx, personalization, candidates, top_n=8):
            captured["candidates"] = candidates
            return [{"candidate_index": i, "score": 90, "explanation": "Relevant resource."}
                    for i in range(len(candidates))]
        raw = dict(RAW, preferred_formats=["Video Tutorials"])
        with patch("src.llm_service.rank_candidates", side_effect=rank):
            result = run_recommendation(raw, fetcher=fetch, personalizer=pz_llm,
                                        partner_client=NO_PARTNER, local_loader=lambda: [unrelated, relevant])
        self.assertTrue(captured["candidates"])
        self.assertFalse(any("operating systems" in c["title"].lower() for c in captured["candidates"]))
        self.assertTrue(all(r["relevant"] for r in result["recommendations"]))

    def test_gemini_cannot_add_unrelated_resources_when_topic_pool_is_small(self):
        from unittest.mock import patch
        relevant = Resource(
            "Python Functions Tutorial", "https://example.org/python-functions",
            "Video Tutorials", "YouTube", "local", tags=["python", "functions"],
            description="Python functions", difficulty="Intermediate"
        )
        unrelated = Resource(
            "Machine Learning Tutorial", "https://example.org/ml",
            "Video Tutorials", "YouTube", "local", tags=["machine learning"],
            description="Machine learning basics", difficulty="Beginner"
        )
        def fetch(tags):
            return FetchResult([], "error", "offline", tags)
        def rank(ctx, personalization, candidates, top_n=8):
            return [{"candidate_index": i, "score": 99-i, "explanation": "Model-selected."}
                    for i in range(len(candidates))]
        raw = dict(RAW, preferred_formats=["Video Tutorials"])
        with patch("src.llm_service.rank_candidates", side_effect=rank):
            result = run_recommendation(raw, fetcher=fetch, personalizer=pz_llm,
                                        partner_client=NO_PARTNER, local_loader=lambda: [relevant, unrelated])
        self.assertTrue(result["recommendations"])
        self.assertFalse(any("machine learning" in r["title"].lower() for r in result["recommendations"]))
        self.assertTrue(all(r["relevant"] for r in result["recommendations"]))

    def test_requested_format_is_strict_when_enough_candidates_exist(self):
        from unittest.mock import patch
        videos = [
            Resource(f"Python Functions Video {i}", f"https://example.org/video-{i}",
                     "Video Tutorials", "YouTube", "local",
                     tags=["python", "functions", "parameters"],
                     description="Python functions and parameters tutorial", difficulty="Intermediate")
            for i in range(2)
        ]
        article = Resource("Python Functions Article", "https://example.org/article",
                           "Articles", "DEV Community", "local",
                           tags=["python", "functions"], description="Python functions guide",
                           difficulty="Intermediate")
        captured = {}
        def fetch(tags):
            return FetchResult([], "error", "offline", tags)
        def rank(ctx, personalization, candidates, top_n=8):
            captured["types"] = [c["type"] for c in candidates]
            return [{"candidate_index": i, "score": 90-i, "explanation": "Relevant resource."}
                    for i in range(min(top_n, len(candidates)))]
        raw = dict(RAW, preferred_formats=["Video Tutorials"])
        with patch("src.llm_service.rank_candidates", side_effect=rank):
            result = run_recommendation(raw, top_n=2, fetcher=fetch, personalizer=pz_llm,
                                        partner_client=NO_PARTNER, local_loader=lambda: videos + [article])
        self.assertGreaterEqual(len(captured["types"]), 2)
        self.assertTrue(all(t == "Video Tutorials" for t in captured["types"]))
        self.assertFalse(result["retrieval"]["format_fallback_used"])
        self.assertTrue(all(r["type"] == "Video Tutorials" for r in result["recommendations"]))

    def test_requested_format_can_use_different_difficulty_to_preserve_count(self):
        from unittest.mock import patch
        videos = [
            Resource(f"Python Functions Video {i}", f"https://example.org/mixed-{i}",
                     "Video Tutorials", "YouTube", "local",
                     tags=["python", "functions"], description="Python functions tutorial",
                     difficulty="Beginner" if i == 0 else "Advanced")
            for i in range(3)
        ]
        def fetch(tags):
            return FetchResult([], "error", "offline", tags)
        def rank(ctx, personalization, candidates, top_n=8):
            return [{"candidate_index": i, "score": 90-i, "explanation": "Covers Python functions."}
                    for i in range(min(top_n, len(candidates)))]
        raw = dict(RAW, difficulty="Intermediate", preferred_formats=["Video Tutorials"])
        with patch("src.llm_service.rank_candidates", side_effect=rank):
            result = run_recommendation(raw, top_n=3, fetcher=fetch, personalizer=pz_llm,
                                        partner_client=NO_PARTNER, local_loader=lambda: videos)
        self.assertEqual(len(result["recommendations"]), 3)
        self.assertTrue(all(r["type"] == "Video Tutorials" for r in result["recommendations"]))
        self.assertFalse(result["retrieval"]["format_fallback_used"])

    def test_analytics_from_real_result(self):
        from src import analytics
        r = self.run_pipe(live_ok, pz_llm)
        m = analytics.summary_metrics(r)
        self.assertEqual(m["recommended"], len(r["recommendations"]))
        self.assertEqual(m["live"] + m["local"] + m["partner"], m["recommended"])
        self.assertFalse(analytics.type_distribution(r).empty)


if __name__ == "__main__":
    unittest.main()

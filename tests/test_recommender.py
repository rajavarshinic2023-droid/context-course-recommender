import unittest

from src.learning_logic import ValidationError, validate_context, score_band, tokenize
from src.recommender import (WEIGHTS, final_score, rank_resources, score_resource, quality_score)
from src.resources import Resource
from tests.helpers import make_ctx, make_res


class ValidationTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(make_ctx().topic, "Python functions")

    def test_empty_topic_rejected(self):
        with self.assertRaises(ValidationError):
            make_ctx(topic="   ")

    def test_score_bounds_and_types(self):
        for bad in (-1, 101, "abc", None, True, float("nan")):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                make_ctx(previous_score=bad)
        self.assertEqual(make_ctx(previous_score="72.5").previous_score, 72.5)

    def test_bad_enums_rejected(self):
        for kw in ({"difficulty": "Expert"}, {"preferred_formats": []}, {"preferred_formats": ["Podcasts"]},
                   {"learning_goal": "Win"}):
            with self.assertRaises(ValidationError):
                make_ctx(**kw)

    def test_bands(self):
        self.assertEqual([score_band(s) for s in (0, 39.9, 40, 69.9, 70, 100)],
                         ["foundational", "foundational", "reinforcement", "reinforcement", "advanced", "advanced"])

    def test_tokenize_synonyms(self):
        self.assertIn("machine", tokenize("ML"))
        self.assertEqual(tokenize("Python Functions"), ["python", "function"])


class ScoringTests(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)

    def test_final_score_is_weighted_sum(self):
        comps = {"topic": 100, "difficulty": 50, "format": 100, "performance": 0, "quality": 80}
        self.assertEqual(final_score(comps), round(30 + 10 + 20 + 0 + 8, 2))

    def test_scores_in_range(self):
        ctx = make_ctx()
        for r in (make_res(), make_res(title="Unrelated cooking", tags=("food",), difficulty=None, tier=None)):
            s = score_resource(r, ctx)
            self.assertTrue(0 <= s.final_score <= 100)
            for v in s.components.values():
                self.assertTrue(0 <= v <= 100)

    def test_relevant_above_unrelated(self):
        ctx = make_ctx()
        ranked = rank_resources([make_res(title="Gardening basics", tags=("garden",)), make_res()], ctx)
        self.assertEqual(ranked[0].resource.title, "Python functions guide")
        self.assertFalse(ranked[-1].relevant)

    def test_format_affects_ranking(self):
        a = make_res(title="Python functions A", type="Articles", url="https://e.com/a")
        v = make_res(title="Python functions V", type="Video Tutorials", url="https://e.com/v")
        top = rank_resources([a, v], make_ctx(preferred_formats=["Video Tutorials"]))[0]
        self.assertEqual(top.resource.type, "Video Tutorials")
        top = rank_resources([a, v], make_ctx(preferred_formats=["Articles"]))[0]
        self.assertEqual(top.resource.type, "Articles")

    def test_difficulty_affects_ranking(self):
        b = make_res(title="Python functions B", difficulty="Beginner", url="https://e.com/b")
        a = make_res(title="Python functions A", difficulty="Advanced", url="https://e.com/ad")
        self.assertEqual(rank_resources([a, b], make_ctx(difficulty="Beginner"))[0].resource.difficulty, "Beginner")
        self.assertEqual(rank_resources([a, b], make_ctx(difficulty="Advanced", previous_score=45))[0].resource.difficulty, "Advanced")

    def test_performance_affects_ranking(self):
        b = make_res(title="Python functions B", difficulty="Beginner", url="https://e.com/b")
        a = make_res(title="Python functions A", difficulty="Advanced", url="https://e.com/ad")
        kw = dict(difficulty="Intermediate")  # equidistant, so only performance separates them
        low = rank_resources([a, b], make_ctx(previous_score=10, **kw))[0].resource.difficulty
        high = rank_resources([a, b], make_ctx(previous_score=95, **kw))[0].resource.difficulty
        self.assertEqual((low, high), ("Beginner", "Advanced"))

    def test_personalization_hint_changes_score(self):
        ctx = make_ctx(preferred_formats=["Articles"])
        doc = make_res(title="Python functions doc", type="Documentation", url="https://e.com/d")
        plain = score_resource(doc, ctx, None).components["format"]
        hinted = score_resource(doc, ctx, {"recommended_formats": ["Documentation"], "suggested_difficulty": "Beginner"}).components["format"]
        self.assertGreater(hinted, plain)

    def test_quality_uses_real_engagement_only(self):
        none = Resource("t", "https://e.com/x", "Articles", "s", "live")
        low = Resource("t", "https://e.com/y", "Articles", "s", "live", reactions=2)
        high = Resource("t", "https://e.com/z", "Articles", "s", "live", reactions=500, comments=20)
        self.assertLess(quality_score(none), quality_score(low))
        self.assertLess(quality_score(low), quality_score(high))
        self.assertLessEqual(quality_score(high), 100)

    def test_empty_and_invalid(self):
        self.assertEqual(rank_resources([], make_ctx()), [])
        with self.assertRaises(TypeError):
            rank_resources([make_res()], {"topic": "x"})

    def test_duplicates_removed(self):
        r = make_res(url="https://e.com/same")
        self.assertEqual(len(rank_resources([r, make_res(url="https://e.com/same/")], make_ctx())), 1)


if __name__ == "__main__":
    unittest.main()

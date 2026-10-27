import unittest
from unittest import mock

try:
    from fastapi.testclient import TestClient
    import api
    HAVE_FASTAPI = True
except Exception:  # fastapi/httpx not installed in this environment
    HAVE_FASTAPI = False

from src.api_client import FetchResult
from src.partner_client import PartnerClient

VALID = {"topic": "Python functions", "difficulty": "Beginner", "preferred_formats": ["Articles", "Practice Questions"],
         "previous_score": 45, "learning_goal": "Understand the fundamentals"}


@unittest.skipUnless(HAVE_FASTAPI, "fastapi / httpx not installed")
class FastApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.p1 = mock.patch("src.api_client.fetch_dev_articles",
                             side_effect=lambda tags, **k: FetchResult([], "error", "down", tags))
        self.p2 = mock.patch.dict("os.environ", {"GEMINI_API_KEY": "", "GROQ_API_KEY": "", "PARTNER_API_URL": ""})
        self.p1.start(); self.p2.start()

    def tearDown(self):
        self.p1.stop(); self.p2.stop()

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

    def test_valid_request_structure(self):
        r = self.client.post("/recommend", json=VALID)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for key in ("input", "personalization", "retrieval", "partner", "recommendations", "weights"):
            self.assertIn(key, body)
        self.assertTrue(body["retrieval"]["fallback_used"])
        self.assertNotIn("all_scored", body)
        self.assertEqual(body["ranking"]["status"], "llm_unavailable")
        self.assertEqual(body["recommendations"], [])
        self.assertIn("No rule-based recommendations", body["message"])

    def test_invalid_requests(self):
        self.assertEqual(self.client.post("/recommend", json=dict(VALID, previous_score=150)).status_code, 422)
        self.assertEqual(self.client.post("/recommend", json=dict(VALID, topic="   ")).status_code, 422)
        self.assertEqual(self.client.post("/recommend", json=dict(VALID, difficulty="Expert")).status_code, 422)
        self.assertEqual(self.client.post("/recommend", json={}).status_code, 422)

    def test_partner_endpoint_maps_request_and_returns_contract(self):
        partner_payload = {
            "student_id": 101, "course": "Python Programming", "topic": "Functions",
            "difficulty": "Intermediate", "assessment_score": 65,
            "completed_lessons": 5, "learning_progress": 40,
            "preferred_resource_type": "Video"
        }
        mocked_result = {
            "recommendations": [{"title": "Functions Guide", "url": "https://example.org/functions",
                                 "type": "Video Tutorials", "score": 0.91,
                                 "explanation": "Matches the topic", "metadata": {"difficulty": "Beginner"}}],
            "ranking": {"status": "llm_used", "message": "ranked by Gemini"},
            "message": ""
        }
        with mock.patch("api.run_recommendation", return_value=mocked_result) as run:
            r = self.client.post("/api/recommendations", json=partner_payload)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["requested_count"], 8)
        self.assertEqual(body["preferred_resource_type"], "Video")
        self.assertEqual(body["preferred_format_count"], 1)
        self.assertEqual(body["recommendations"][0]["resource_type"], "Video Tutorials")
        self.assertEqual(body["recommendations"][0]["difficulty"], "Beginner")
        raw = run.call_args.args[0]
        self.assertEqual(raw["topic"], "Functions")
        self.assertEqual(raw["preferred_formats"], ["Video Tutorials"])
        self.assertEqual(raw["previous_score"], 65)
        self.assertIn("Learning progress: 40%", raw["additional_context"])

    def test_partner_difficulty_easy_medium_hard_mapping(self):
        base = {
            "course": "Python Programming", "topic": "Functions",
            "assessment_score": 65, "completed_lessons": 5,
            "learning_progress": 40, "preferred_resource_type": "Video"
        }
        mocked_result = {"recommendations": [], "ranking": {"status": "llm_used", "message": "ranked"}, "message": ""}
        for partner_level, expected in (("Easy", "Beginner"), ("Medium", "Intermediate"), ("Hard", "Advanced")):
            with self.subTest(partner_level=partner_level), mock.patch("api.run_recommendation", return_value=mocked_result) as run:
                response = self.client.post("/api/recommendations", json=dict(base, difficulty=partner_level))
                self.assertEqual(response.status_code, 200)
                self.assertEqual(run.call_args.args[0]["difficulty"], expected)

    def test_partner_endpoint_rejects_unknown_difficulty(self):
        payload = {"course": "Python", "topic": "Functions", "difficulty": "Expert",
                   "assessment_score": 60, "preferred_resource_type": "Video"}
        self.assertEqual(self.client.post("/api/recommendations", json=payload).status_code, 422)

    def test_partner_endpoint_rejects_bad_score(self):
        payload = {"course": "Python", "topic": "Functions", "difficulty": "Beginner",
                   "assessment_score": 101, "completed_lessons": 1,
                   "learning_progress": 10, "preferred_resource_type": "Video"}
        self.assertEqual(self.client.post("/api/recommendations", json=payload).status_code, 422)

    def test_partner_endpoint_does_not_fake_llm_success(self):
        payload = {"course": "Python", "topic": "Functions", "difficulty": "Beginner",
                   "assessment_score": 60, "completed_lessons": 1,
                   "learning_progress": 10, "preferred_resource_type": "Video"}
        mocked_result = {"recommendations": [], "ranking": {"status": "llm_unavailable", "message": "quota exhausted"}, "message": "quota exhausted"}
        with mock.patch("api.run_recommendation", return_value=mocked_result):
            r = self.client.post("/api/recommendations", json=payload)
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["detail"]["status"], "llm_unavailable")

    def test_internal_error_is_safe(self):
        with mock.patch("api.run_recommendation", side_effect=RuntimeError("secret detail")):
            r = self.client.post("/recommend", json=VALID)
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("secret", r.text)


if __name__ == "__main__":
    unittest.main()

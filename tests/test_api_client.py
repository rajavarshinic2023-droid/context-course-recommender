import unittest
from unittest import mock

import requests

from src.api_client import derive_tags, fetch_dev_articles, normalize_article
from tests.helpers import make_ctx


def resp(json_data=None, status=200, bad_json=False):
    r = mock.Mock()
    r.status_code = status
    r.raise_for_status.side_effect = requests.HTTPError("err") if status >= 400 else None
    if bad_json:
        r.json.side_effect = ValueError("bad json")
    else:
        r.json.return_value = json_data
    return r


ART = {"id": 1, "title": "Python Functions for Beginners", "url": "https://dev.to/a/py-1",
       "description": "Intro", "tag_list": ["python", "beginners"], "published_at": "2025-01-02T00:00:00Z",
       "public_reactions_count": 120, "comments_count": 4, "reading_time_minutes": 5,
       "user": {"name": "Ada", "username": "ada"}}


class DevApiTests(unittest.TestCase):
    def test_success_parsing(self):
        out = fetch_dev_articles(["python"], http_get=lambda *a, **k: resp([ART]))
        self.assertEqual(out.status, "live")
        r = out.resources[0]
        self.assertEqual((r.type, r.origin, r.reactions, r.difficulty, r.difficulty_source),
                         ("Articles", "live", 120, "Beginner", "inferred"))
        self.assertIn("Ada", r.source)

    def test_request_parameters(self):
        get = mock.Mock(return_value=resp([ART]))
        fetch_dev_articles(["python"], http_get=get, timeout=5)
        _, kwargs = get.call_args
        self.assertEqual(kwargs["params"]["tag"], "python")
        self.assertEqual(kwargs["timeout"], 5)

    def test_empty_response(self):
        self.assertEqual(fetch_dev_articles(["x"], http_get=lambda *a, **k: resp([])).status, "empty")

    def test_timeout_http_error_invalid_json_wrong_shape(self):
        def timeout(*a, **k):
            raise requests.Timeout()
        self.assertEqual(fetch_dev_articles(["x"], http_get=timeout).status, "error")
        self.assertEqual(fetch_dev_articles(["x"], http_get=lambda *a, **k: resp(status=500)).status, "error")
        self.assertEqual(fetch_dev_articles(["x"], http_get=lambda *a, **k: resp(bad_json=True)).status, "error")
        self.assertEqual(fetch_dev_articles(["x"], http_get=lambda *a, **k: resp({"a": 1})).status, "error")

    def test_missing_metadata_and_invalid_links(self):
        items = [{"title": "No url"}, {"url": "https://dev.to/x"}, {"title": "Bad", "url": "javascript:alert(1)"},
                 {"title": "Minimal", "url": "https://dev.to/min"}, "garbage", None]
        out = fetch_dev_articles(["x"], http_get=lambda *a, **k: resp(items))
        self.assertEqual([r.title for r in out.resources], ["Minimal"])
        self.assertIsNone(out.resources[0].reactions)
        self.assertEqual(out.resources[0].difficulty_source, "unknown")

    def test_dedupe_across_tags_and_partial_failure(self):
        calls = iter([resp([ART]), resp([ART]), resp(status=503)])
        out = fetch_dev_articles(["a", "b", "c"], http_get=lambda *a, **k: next(calls))
        self.assertEqual(len(out.resources), 1)
        self.assertIn("failed", out.message)

    def test_derive_tags(self):
        self.assertEqual(derive_tags(make_ctx(topic="Machine Learning"))[:2], ["machinelearning", "machine"])
        self.assertEqual(len(derive_tags(make_ctx(topic="alpha beta gamma delta"), ["x", "y"])), 3)

    def test_normalize_non_dict(self):
        self.assertIsNone(normalize_article(5))


if __name__ == "__main__":
    unittest.main()

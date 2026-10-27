import unittest
from unittest import mock

import requests

from src.partner_client import PartnerClient
from tests.helpers import make_ctx


def ok_resp(data):
    r = mock.Mock()
    r.raise_for_status.return_value = None
    r.json.return_value = data
    return r


class PartnerTests(unittest.TestCase):
    def test_not_configured(self):
        with mock.patch.dict("os.environ", {"PARTNER_API_URL": ""}):
            res = PartnerClient().exchange(make_ctx())
        self.assertEqual(res.status, "not_configured")

    def test_success_and_filtering(self):
        data = {"recommended_resources": [
            {"title": "A", "url": "https://p.example.com/a", "type": "Articles", "difficulty": "Beginner"},
            {"title": "B", "url": "not a url", "type": "Articles"},
            {"title": "C", "url": "https://p.example.com/c", "type": "Podcast"}], "guidance": "Practise daily"}
        post = mock.Mock(return_value=ok_resp(data))
        res = PartnerClient("https://p.example.com", "/rec", "k", http_post=post).exchange(make_ctx())
        self.assertEqual((res.status, len(res.resources), res.resources[0].origin), ("ok", 1, "partner"))
        _, kw = post.call_args
        self.assertEqual(kw["json"]["topic"], "Python functions")
        self.assertEqual(kw["headers"]["Authorization"], "Bearer k")

    def test_failures(self):
        def timeout(*a, **k):
            raise requests.Timeout()
        self.assertEqual(PartnerClient("https://p.example.com", http_post=timeout).exchange(make_ctx()).status, "error")
        self.assertEqual(PartnerClient("https://p.example.com", http_post=lambda *a, **k: ok_resp([1])).exchange(make_ctx()).status,
                         "invalid_response")


if __name__ == "__main__":
    unittest.main()

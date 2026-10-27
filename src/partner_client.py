"""Optional partner-application client, kept separate from the core engine.

NOTHING here is verified against a real partner: the endpoint, path and schema are
configurable placeholders until the partner supplies theirs. A successful exchange is only
reported when an HTTP response was actually received and validated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

from .config import get_setting
from .learning_logic import DIFFICULTIES, FORMATS, StudentContext
from .resources import Resource, is_valid_url

log = logging.getLogger(__name__)


@dataclass
class PartnerResult:
    status: str = "not_configured"   # not_configured | ok | error | invalid_response
    message: str = "Partner API is not configured (set PARTNER_API_URL to enable it)."
    resources: List[Resource] = field(default_factory=list)
    guidance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "message": self.message, "count": len(self.resources),
                "guidance": self.guidance}


class PartnerClient:
    def __init__(self, base_url: Optional[str] = None, path: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 8.0,
                 http_post: Callable = requests.post):
        self.base_url = (base_url if base_url is not None else get_setting("PARTNER_API_URL") or "").rstrip("/")
        self.path = path if path is not None else get_setting("PARTNER_API_PATH", "/recommend")
        self.api_key = api_key if api_key is not None else get_setting("PARTNER_API_KEY")
        self.timeout, self.http_post = timeout, http_post

    @property
    def configured(self) -> bool:
        return is_valid_url(self.base_url)

    def build_payload(self, ctx: StudentContext) -> Dict[str, Any]:
        return {"topic": ctx.topic, "difficulty": ctx.difficulty,
                "preferred_formats": list(ctx.preferred_formats),
                "previous_score": ctx.previous_score, "learning_goal": ctx.learning_goal}

    def exchange(self, ctx: StudentContext) -> PartnerResult:
        if not self.configured:
            return PartnerResult()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = self.http_post(self.base_url + self.path, json=self.build_payload(ctx),
                                  headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            return PartnerResult("error", "The partner API timed out.")
        except (requests.RequestException, ValueError) as exc:
            log.warning("Partner API failure: %s", type(exc).__name__)
            return PartnerResult("error", "The partner API could not be reached or returned invalid data.")
        return self.parse_response(data)

    @staticmethod
    def parse_response(data: Any) -> PartnerResult:
        if not isinstance(data, dict):
            return PartnerResult("invalid_response", "The partner API reply was not a JSON object.")
        items = data.get("recommended_resources", [])
        if not isinstance(items, list):
            return PartnerResult("invalid_response", "The partner API reply had an invalid resource list.")
        resources: List[Resource] = []
        for it in items:
            if not isinstance(it, dict) or not it.get("title") or not is_valid_url(it.get("url")):
                continue
            if it.get("type") not in FORMATS:       # never guess/mislabel a resource type
                continue
            diff = it.get("difficulty") if it.get("difficulty") in DIFFICULTIES else None
            resources.append(Resource(
                title=str(it["title"]).strip(), url=it["url"].strip(), type=it["type"],
                source=str(it.get("source") or "Partner application"), origin="partner",
                description=str(it.get("description") or ""),
                tags=[str(t) for t in it.get("tags", []) if isinstance(t, str)],
                difficulty=diff, difficulty_source="provided" if diff else "unknown"))
        guidance = data.get("guidance")
        return PartnerResult("ok", f"Partner API replied with {len(resources)} valid resource(s).",
                             resources, guidance[:500] if isinstance(guidance, str) else "")

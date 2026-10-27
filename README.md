# Context-Aware Study Resource Recommender Using Learning Patterns

Fourth-year Integrated M.Tech CSE · Advanced Python Programming project.

## 1. Abstract
A Streamlit web application that recommends study resources from a student's learning context (topic, level, preferred formats, previous score, goal). A Gemini interprets the context and ranks retrieved resource candidates first; Groq is a backup ranking provider. The interface reports which ranking provider succeeded. Candidate URLs come from live sources, the curated local library, and an optional partner API.

## 2. Problem statement
Search engines return unfiltered link lists that ignore a learner's level, preferred format and recent performance.

## 3. Objectives
Collect a learning context; interpret it with Gemini; retrieve candidate resources; have Gemini rank candidates and explain learner-specific relevance; report service failures clearly.

## 4. Limitations of the existing approach and proposed solution
Plain keyword search ignores context. Here: Gemini interprets the learner profile and performs the primary candidate-ranking decision. Python validates inputs, retrieves and validates URLs, and provides diagnostic signals. If Gemini ranking is unavailable, the app does not silently substitute rule-based recommendations.

## 5. Technology stack
Python 3.10+, Streamlit, requests, pandas, python-dotenv, google-genai (Gemini), Groq API backup, FastAPI + Uvicorn (optional REST), pytest/unittest.

## 6. Architecture
```
+---------+   +-----------+   +------------+   +--------------------+
| Student |-->| Streamlit |-->| Validation |-->| Personalization    |
|         |   |  app.py   |   | learning_  |   | llm_service.py     |
+---------+   +-----------+   | logic.py   |   | (Gemini / rules)   |
                  ^           +------------+   +---------+----------+
                  |                                      |
          +-------+--------+   +-------------------+     v
          | Results, cards |<--| recommender.py    |<--+--------------------+
          | analytics      |   | weighted scoring  |   | Retrieval          |
          +----------------+   +-------------------+   | api_client.py DEV  |
                                        ^              | resources.py local |
   +----------------+  uses same         |              | partner_client.py  |
   | api.py FastAPI |-- pipeline.py -----+              +--------------------+
   +----------------+
```
`src/pipeline.py::run_recommendation` is the **single** orchestration path used by both `app.py` and `api.py`.

## 7. Directory structure
```
app.py                Streamlit UI (5 tabs)
api.py                FastAPI service: GET /health, POST /recommend, POST /api/recommendations
src/config.py         Environment settings (no secrets in code)
src/learning_logic.py Validation, score bands, tokenizer, rule-based personalization
src/llm_service.py    Gemini personalization + primary LLM candidate ranking
src/api_client.py     DEV Community API client + normalisation
src/resources.py      Resource model, URL validation, de-dup, local library loader
src/recommender.py    Diagnostic factor scores (not the ranking decision)
src/partner_client.py Optional, configurable partner exchange
src/pipeline.py       Orchestration (validate -> personalise -> retrieve -> rank)
src/analytics.py      DataFrames/metrics for the analytics tab
data/sample_resources.json  34 curated local resources
tests/                48 tests (unittest-style; run by pytest too)
```

## 8. Workflow
Form → validate learner context → Gemini personalization → retrieve live DEV/Wikipedia resources + local library (+ optional partner resources) → de-duplicate → Gemini ranks supplied candidates and writes explanations → display top recommendations. Python diagnostic scores do not decide recommendation order.

## 9. LLM integration
* Real call via the official `google-genai` SDK (`client.models.generate_content`, JSON mime type) only when `GEMINI_API_KEY` is set.
* Prompt carries the student profile as JSON *data* and tells the model to ignore instructions inside it.
* Reply is parsed and **whitelisted**: invalid difficulty → reject; unknown formats dropped; keywords reduced to `[a-z0-9]`; lengths capped; unknown keys ignored. Nothing is executed.
* Personalization can still show a transparent rule-based context if its Gemini call fails, but final recommendations require a successful Gemini ranking call. If ranking cannot run, the app returns no recommendations and displays a clear status; it does not fall back to weighted rule-based ranking. Errors are generic; keys are never logged.
* Results are cached in memory per identical input to avoid repeat calls.
* Primary recommendation decision: Gemini receives the learner context and a bounded list of retrieved candidates (title, URL, source, description, tags, format and difficulty). It returns candidate indices, scores and explanations. The response is validated so it can only select supplied URLs; invented links are not accepted.

## 10. Diagnostic scoring (not used to rank recommendations)
`Final = 0.30·Topic + 0.20·Difficulty + 0.20·Format + 0.20·Performance + 0.10·Quality` (each 0–100, final clamped to 0–100, rounded to 2 dp).

| Component | Calculation |
|---|---|
| Topic | Each topic token earns 1.0 if in title, 0.7 in tags, 0.4 in description; averaged ×100. Personalization terms fill at most 30 % of the remaining gap. +10 if every topic token is in the title. Resources scoring < 20 are marked not relevant and are not recommended. |
| Difficulty | Exact 100, adjacent level 60, two levels 20; unknown 50. A personalised suggested level counts at 90 % of its fit (max of the two). |
| Format | Preferred 100; suggested by personalization 55; other 20. |
| Performance | Score band: < 40 foundational, 40–69 reinforcement, ≥ 70 advanced. `0.6·difficulty-fit + 0.4·format-fit` from tables in `recommender.py` (e.g. low score favours Beginner articles/videos; high score favours Advanced practice/documentation). |
| Quality | Live DEV: `35 + 65·min(1, log10(1+reactions+2·comments)/log10(301))` from real reaction/comment counts; 40 if none supplied. Local: documented editorial tier (official 85, established 75, community 60). No ratings are invented. |

These diagnostic components are retained for transparency/analytics only. They do not determine the order of recommendations. Gemini ranking is still an LLM-based decision and should be evaluated for relevance and consistency.

## 11. External API
`GET https://dev.to/api/articles?tag=<tag>&per_page=30&top=365` (up to 3 tag requests, 8 s timeout). Handles timeout, HTTP errors, invalid JSON, wrong shape, missing fields; filters invalid URLs; de-duplicates. Article difficulty is **inferred** from title/tag words (e.g. "beginner", "advanced") and labelled `inferred`; otherwise unknown. DEV articles are always type "Articles".
> The DEV `tag` filter and parameter names follow my knowledge of the Forem API; they could not be verified against the live service from the offline build environment — run once with internet and check the API Status tab.

## 12. Fallback
If the live call fails or is empty, the UI shows: *"Live resource retrieval is currently unavailable. Showing relevant resources from the local resource library."* Results carry an `origin` (`live` / `local` / `partner`) shown on every card. The local library is also always merged in for formats DEV lacks (video, courses, practice, docs).

## 13. Partner API (optional, NOT verified)
`src/partner_client.py` POSTs `{topic, difficulty, preferred_formats, previous_score, learning_goal}` to `PARTNER_API_URL + PARTNER_API_PATH` (Bearer `PARTNER_API_KEY` optional) and accepts `{"recommended_resources":[{title,url,type,...}], "guidance":"..."}`. No partner specification was supplied, so this schema is a placeholder: **no successful exchange has been tested with a real partner.** To finish: obtain the partner's base URL, path, auth and response schema and adjust `build_payload` / `parse_response`. Unset `PARTNER_API_URL` disables it.

## 14. REST API
`uvicorn api:app --reload --host 0.0.0.0 --port 8000` exposes `GET /health`, the original `POST /recommend`, and the EduPulse-facing `POST /api/recommendations`; interactive docs are at `/docs`.

The partner endpoint accepts `difficulty` values `Easy`, `Medium`, and `Hard` (also accepts `Beginner`, `Intermediate`, and `Advanced`) and maps them internally to `Beginner`, `Intermediate`, and `Advanced`. `preferred_resource_type` accepts `Video`, `Article`, `Practice`, `Documentation`, `Course`, or `Mixed`. `top_n` controls the requested number of results (1–20; default 8). Resources matching the preferred format are placed earlier in the LLM candidate pool and the prompt explicitly prioritizes them. The response includes `count`, `requested_count`, `preferred_resource_type`, and `preferred_format_count`. It may return fewer than requested if the LLM identifies fewer relevant candidates or suitable resources are unavailable; it does not invent URLs or pad the list with fake recommendations. Example body:
```json
{"topic":"Python functions","difficulty":"Beginner","preferred_formats":["Articles","Practice Questions"],"previous_score":45,"learning_goal":"Understand the fundamentals"}
```

EduPulse partner request example:
```json
{"student_id":101,"course":"Python Programming","topic":"Python Functions","difficulty":"Medium","assessment_score":65,"completed_lessons":5,"learning_progress":40,"preferred_resource_type":"Video","top_n":5}
```
A `200 OK` response with `status: "llm_used"` confirms LLM ranking was used. A `503` with `status: "llm_unavailable"` means Gemini could not rank the candidates (for example, due to quota/key/model/network issues); the API deliberately does not present rule-based results as LLM output.

## 15. Installation and run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then edit GEMINI_API_KEY (Windows: copy)
streamlit run app.py
uvicorn api:app --reload         # optional, separate terminal
pytest -q                        # or: python -m unittest discover -s tests -t .
```
Environment variables: `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_TIMEOUT_SECONDS`, `PARTNER_API_URL`, `PARTNER_API_PATH`, `PARTNER_API_KEY`, and optional `YOUTUBE_API_KEY`. Internet is needed for live retrieval and Gemini; the app still loads its local library offline.

**Troubleshooting:** "LLM NOT used / no_api_key" → set the key in `.env`; `llm_error` → check the key, quota, model name, network; `ModuleNotFoundError` → activate the venv and reinstall; port busy → `streamlit run app.py --server.port 8502`.

## 16. Streamlit Community Cloud
Push the repo (without `.env`), create the app with entry point `app.py`, and add secrets under *App settings → Secrets* as `GEMINI_API_KEY = "..."` (Streamlit exposes top-level secrets as environment variables). **The app has not been deployed**; do not claim a hosted URL until you have one.

## 17. Test status
See the final summary for the actual run. The test suite covers validation, scoring, ranking effects, LLM validation/fallback, DEV client (mocked), partner client, pipeline integration, Easy/Medium/Hard mapping, partner response shape and FastAPI (skipped automatically if `fastapi`/`httpx` are missing). Tests never call live services.

## 18. Expected output
Home form → Recommendations tab: LLM status banner, live/fallback banner, guidance box, ranked cards (score, explanation, breakdown, link) → Explorer (filter/sort), Analytics (metrics and charts from the current run only), API Status (configuration + last-run outcomes).

## 19. Limitations
* Scores come from hand-set weights/tables, not a trained model; there is no ground-truth evaluation.
* Keyword matching (not semantic); topics outside the 34-item local library rely on live DEV articles only.
* Local library links and the `top`/`tag` DEV parameters were not link-checked from the offline build environment. YouTube search pages are explicitly labelled as search pages, not individual videos.
* No persistent student history; analytics reflect one run.
* Streamlit UI was syntax-checked but not launched in the build environment.

## 20. Future work
Semantic embeddings for topic match, user feedback loop to tune weights, more live sources (video/docs APIs), persistent profiles, offline evaluation (precision@k).


### Broader subject coverage
The recommender accepts free-text topics across subjects. It tries DEV Community for developer-oriented articles and Wikipedia's public search API for general reference articles, then combines these with the curated local library and any configured partner resources. Live sources require internet access; the local library remains finite, and the app does not promise every subject or resource format will always have results. Wikipedia results are reference material, not necessarily structured lessons or videos.


## Recommendation quality safeguards (latest update)

- The pipeline now applies a topic-relevance gate before sending candidates to Gemini and checks relevance again after ranking. This helps prevent unrelated resources from being used to fill the list.
- Candidate ordering prioritizes the requested resource format, then closer difficulty levels, then stronger topic matches.
- Gemini's ranking prompt now explicitly prioritizes direct topic relevance and the requested difficulty. The original resource difficulty is not rewritten to match the learner.
- The local Python video entry uses a more specific search query for functions, parameters, arguments, and return values. It is clearly labeled as a YouTube search page, not a verified individual video.
- The API returns fewer than the requested count if fewer suitable resources are available. It does not duplicate or invent resources to reach the target.

A YouTube search-results URL is not the same as a verified individual video. This build can retrieve individual video URLs through YouTube Data API v3 when `YOUTUBE_API_KEY` is configured. Add `YOUTUBE_API_KEY=your-key` to the root `.env` file and enable YouTube Data API v3 for that key. Keep the key private and never commit `.env`. Without the key, the system adds at most one clearly labelled topic-specific YouTube search page rather than fabricating several search pages and counting them as separate videos. If fewer relevant videos are available, it returns fewer recommendations and explains the shortage rather than including unrelated topics or silently switching formats.

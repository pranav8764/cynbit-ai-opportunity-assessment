# Technical Approach Report - Revio Cross-Review Issue Miner

Companion to `docs/AI_OPPORTUNITY_REPORT.md`. Code: `poc/review_miner.py`.

## 1. Problem
Given raw customer reviews from several branches, produce for each branch a ranked list of recurring complaint categories with supporting verbatim quotes.

## 2. Architecture
```
reviews (JSON) -> prompt builder -> LLM (OpenRouter chat completions) -> strict parser/validator
              -> aggregator (plain Python) -> per-branch summary (JSON + console) 
```
Two steps are kept separate on purpose:
1. **Text to structured aspects (LLM).** One call per review. Output contract: `{"aspects": [{"category", "sentiment", "severity", "evidence"}]}`.
2. **Structured aspects to priorities (no ML).** Per branch and category: negative-mention count, average severity, and priority = sum of raw negative severities. Top 3 quotes are attached.

## 3. Model choice
- **API prompt-based classifier, no training.** No dataset was provided and training is not required. For a short-text, fixed-label task, a prompted LLM is the fastest route and avoids building a labelled set first.
- **Default model:** `nvidia/nemotron-3.5-super-vl:free` through OpenRouter, a $0 model at the time of the run, with an API key that has a $0 credit limit. Any OpenRouter slug can be set with `OPENROUTER_MODEL` or `--model`.
- **Alternative (not used for the reported run):** `typesafe/jev-router`. It routes each call to varying models and cost is not fixed, so it makes results harder to explain.
- No claim is made that this model is better than any other; no comparison was run.

## 4. Data pipeline
- Input: `data/reviews_synthetic.json`, 15 synthetic reviews across 3 branches (Jaipur C-Scheme, Malviya Nagar, Vaishali Nagar), written for this PoC. No real customer data.
- Each review is sent with branch, date, star rating and text.
- Output files: `out/classifications.json` and `out/summary.json` (ignored by git).

## 5. Prompt engineering
- System prompt fixes the allowed categories: `service_speed`, `staff_behavior`, `product_quality`, `pricing_billing`, `cleanliness_ambience`, `other`.
- One review can produce several aspects.
- `evidence` must be a verbatim quote; `severity` is 1 to 5 and rates the aspect, not the review.
- Reply JSON only, in an exact shape; the request also sets `response_format: json_object`.

## 6. Validation and error handling
- The parser accepts plain JSON and JSON wrapped in prose or code fences.
- Unknown category, sentiment or severity values fall back to defined defaults.
- Non-JSON responses raise an error instead of producing a fake result.
- Ranking uses raw severity sums, so rounding of the displayed average cannot change the order (covered by a regression test).

## 7. Testing: what was run
- 8 offline pytest tests (`tests/test_review_miner.py`): JSON parsing, wrapped JSON, invalid-field fallback, non-JSON error, ranking math (a category with severity 4+2 ranks above a single severity 5; positive mentions are excluded), full pipeline on canned responses with no network, fixed category set, and the rounding regression.
- Mock mode (`--mock`) replays canned fixtures. It proves the pipeline runs, not that any model produces those outputs.
- One live run on 2026-10-29 with the free model: 15/15 reviews classified, 35 aspects. Every evidence quote was checked to appear verbatim in its source review. Results: `docs/live-run-2026-10-29.md`.

## 8. What was not tested
- Accuracy against hand labels (no labelled set, no baseline comparison yet).
- Real reviews, other languages (Hinglish, Hindi) or long reviews.
- Prompt injection inside review text, and PII handling.
- Repeatability across runs, rate-limit behaviour at volume, and cost at scale.

## 9. Execution plan
| Step | What | Done? |
|---|---|---|
| 1 | Fixed-category prompt, parser, aggregator, offline tests | Yes |
| 2 | Live run on synthetic reviews with a free model | Yes (one run) |
| 3 | Hand-label reviews, keyword baseline, compare precision/recall | Planned (`docs/evaluation-plan.md`) |
| 4 | Add PII redaction, injection tests, Hinglish reviews | Planned |
| 5 | Small web view or Revio dashboard integration | Planned |

## 10. How to run
See README. Offline: `python3 -m poc.review_miner --mock`. Live: set `OPENROUTER_API_KEY` in your own environment and run `python3 -m poc.review_miner`. Keys are never committed.

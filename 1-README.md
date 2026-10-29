# Cynbit AI/ML Engineer Intern - Technical Assessment

## Status
- **Assigned:** Thu, Oct 29, 2026 (email from opportunities@cynbit.in)
- **Deadline:** Sat, Oct 31, 2026, 4:45 PM (timezone not stated in the email or portal; very likely IST - the deadline is exactly 48h after the assignment email, sent 4:45 PM IST)
- **Submission:** candidate portal (https://cynbit.in/candidate/cmt8cbikl000004l40fphlaj2) - 1 primary link + up to 4 additional links + optional notes
- **Visibility rule:** the deliverable must be **publicly accessible** at submission time. Keep this repo private while working; flip to public only when ready to submit.

## Problem Statement (verbatim from the assignment)
Analyze Cynbit's website, services, and publicly available information. Identify opportunities where AI/ML could improve Cynbit's business, internal processes, customer experience, or future products. Choose **one** opportunity and demonstrate how you would approach it.

## Required Deliverables
1. **AI Opportunity Report** - industry analysis detailing high-value AI/ML opportunities for Cynbit.
2. **AI/ML Proof of Concept** - functional demo or code prototype (Colab notebook, Python script, Streamlit app, or API).
3. **Technical Approach Report** - document covering model choices, data pipeline, prompt engineering/architecture, and execution plan.

No dataset is provided and there is no custom-training requirement; model choice is the applicant's. No AI-assistance or collaboration restrictions are stated in the assignment email.

## Chosen opportunity (proposed, not a company-confirmed gap)
**Cross-review issue mining for Revio** (Cynbit's AI review-management product).
Revio's site (https://www.cynbit.in/products) already claims per-review sentiment, AI responses, crisis detection and a multi-location dashboard. The proposed extension aggregates *across* reviews: group complaints into recurring issue themes, rank each branch's top issues by frequency x severity, and attach the verbatim review quotes as evidence. That cross-review, evidence-linked prioritization is not among the advertised features.

## PoC: `poc/review_miner.py`
Two steps, deliberately separated:

1. **Text -> structured aspects (LLM API).** Each free-text review goes to an LLM via OpenRouter with a fixed-category prompt and a JSON structured-output contract. Categories: `service_speed`, `staff_behavior`, `product_quality`, `pricing_billing`, `cleanliness_ambience`, `other`. Each aspect carries sentiment, severity (1-5) and a verbatim evidence quote.
2. **Structured aspects -> priorities (plain analytics).** Per branch: negative-mention counts, average severity, priority score = count x avg severity, top-3 evidence quotes. No ML here, by design.

## Setup
```bash
git clone <this repo>
cd cynbit-ai-opportunity-assessment
python3 -m pip install -r requirements.txt   # only needed for the tests (pytest)
```
Runtime uses the Python standard library only (3.10+). No other dependencies.

## Run - offline demo (no API key, no network)
```bash
python3 -m poc.review_miner --mock
```
Replays canned model responses from `data/mock_classifications.json` against the 15 synthetic reviews in `data/reviews_synthetic.json` and writes `out/classifications.json` + `out/summary.json`.

**Note:** mock outputs are canned fixtures. They prove the pipeline runs, not that any model produces them.

## Run - live (OpenRouter)
```bash
export OPENROUTER_API_KEY=sk-or-...            # never commit this
export OPENROUTER_MODEL=nvidia/nemotron-3.5-super-vl:free   # default: verified $0 on OpenRouter (2026-10-29); any slug works
# Jev Router alternative: OPENROUTER_MODEL=typesafe/jev-router + JEV_COST_TIER=low (preference, NOT a price cap, cost not fixed)
python3 -m poc.review_miner
```
`JEV_COST_TIER` is only sent when the model is `typesafe/jev-router`. The run prints which model actually served each response (the router chooses) and writes the same two output files. Live results are pending until an API key is available; nothing in `out/` should be presented as model output unless produced by a live run.

## Tests
```bash
python3 -m pytest tests/ -q
```
Offline only: parser edge cases (code fences, junk output, unknown categories), ranking math, and the mock pipeline end-to-end.

## Repo layout
```
data/reviews_synthetic.json       15 synthetic reviews, 3 branches (no real customer data)
data/mock_classifications.json    canned model outputs for --mock mode
poc/review_miner.py               classifier + aggregator + CLI
tests/test_review_miner.py        offline tests
docs/evaluation-plan.md           baseline-vs-model evaluation plan (no results yet)
```

## Research base (company)
- Cynbit Technologies, Jaipur. Services: AI agents/automation, custom software, cloud, web/mobile, technical SEO (https://www.cynbit.in/services)
- Products: Cynbit Studio (career/coding), TenantEase (property), Revio (review/reputation AI) (https://www.cynbit.in/products)
- Case studies: https://www.cynbit.in/case-studies

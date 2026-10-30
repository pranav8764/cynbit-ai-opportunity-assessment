# AI Opportunity Report - Cynbit Technologies

Author: Pranav Singh Rajoria. Written for the Cynbit AI/ML Engineer Intern assessment (Oct 2026).

## 1. Scope and sources
This report is based only on Cynbit's public website, read in Oct 2026:
- Services: https://www.cynbit.in/services
- Products (Cynbit Studio, TenantEase, Revio): https://www.cynbit.in/products
- Case studies: https://www.cynbit.in/case-studies

I did not have access to Cynbit's internal data, customers or revenue. The opportunities below are proposals inferred from public pages, not confirmed gaps. I have not included third-party market statistics, because I did not verify them against primary sources for this submission.

## 2. What Cynbit does (from the public site)
- **Services:** autonomous AI agents and cognitive automation, custom software, cloud, web/mobile, and technical SEO. Stack listed: Python, TypeScript, FastAPI, Next.js, OpenAI, LangChain, PyTorch, Pinecone, AWS, Docker, Kubernetes.
- **Products:** Cynbit Studio (adaptive coding paths, AI code evaluation, AI mock interviews, AI resume engine), TenantEase (property management) and Revio (AI-first reputation engine: cross-platform review sync, AI response engine, sentiment intelligence, crisis detection, multi-location control).
- **Case studies:** AI diagnostic triage, IoT patient monitoring, adaptive learning, campaign optimisation, AR try-on, ERP modernisation, payments and others across health, retail, education and fintech.

## 3. Candidate AI opportunities
Scoring is my judgement (High/Medium/Low) from public information only.

| # | Opportunity | Where it fits | Value | Feasibility in weeks | Risk |
|---|---|---|---|---|---|
| 1 | **Cross-review issue mining for Revio**: group complaints into recurring themes per branch, rank by frequency x severity, attach verbatim quotes | Revio (multi-location control, sentiment intelligence) | High: turns reviews into an action list for a multi-branch owner | High: a single LLM call per review plus ordinary analytics | LLM mislabels; needs evaluation on real reviews |
| 2 | Reusable AI-agent evaluation harness for client projects (test prompts, regression checks, cost tracking) | Services (AI agents and automation) | Medium-High: lowers delivery risk across projects | Medium | Needs real client workloads |
| 3 | Code-review feedback quality checks for Cynbit Studio (rubric-based scoring, consistency audits) | Cynbit Studio AI code evaluation | Medium | Medium | Ground truth is subjective |
| 4 | Tenant-message triage and late-rent risk signals for TenantEase | TenantEase | Medium | Medium | Fairness and privacy concerns with tenant data |
| 5 | Internal proposal/case-study assistant (retrieve past case studies to draft client proposals) | Sales and services | Medium | High | Hallucinated claims in client documents |

## 4. Chosen opportunity and why
I chose **#1, cross-review issue mining for Revio**.
- Revio's public page already advertises per-review sentiment, AI replies, crisis detection and multi-location control. What I could not find advertised is aggregation across reviews into ranked, evidence-linked issues per branch. That is the proposed extension.
- It needs no custom training and no private data to demonstrate. Reviews can be synthetic for a demo.
- The model's job is narrow (text to a fixed category/sentiment/severity contract). Counting and ranking stay in plain code, which is easy to test.

**What it is not:** it is not a claim that Revio lacks this feature internally, and not evidence of real-world accuracy.

## 5. Expected value and success measures (targets, not results)
- Owner can see each branch's top 3 issues with quotes in one screen.
- Classification agreement with hand labels (category, sentiment) on a labelled set. Target to be set after the first labelled run; no number has been measured yet.
- Cost per 1,000 reviews and latency, recorded per model.

## 6. Risks and limits
- The PoC was run only on 15 synthetic reviews. It shows the pipeline works, not that the model is accurate on real reviews.
- Free-tier models have rate limits and can change; the model slug is configurable.
- Reviews can contain personal data and prompt-injection text. The PoC has no PII redaction or injection defence; both are required before production use.
- Severity scores come from the model and are not calibrated.

## 7. Recommended next steps
1. Hand-label a few hundred real, consented reviews and compare the LLM against a keyword baseline (plan in `docs/evaluation-plan.md`).
2. Add PII redaction and prompt-injection tests.
3. Pilot with one multi-branch Revio customer and measure whether owners act on the ranked issues.

"""Hosted demo: cross-review issue mining (Cynbit AI/ML Engineer Intern assessment PoC).

Wraps the pipeline from poc/review_miner.py in a small web UI:
- analyze a single pasted review with the LLM (OpenRouter free model), or
- run the 15-review synthetic sample batch, live or replaying canned fixtures.

The OpenRouter key lives only in the host's secret store and the model slug is a
hardcoded :free model, so public use can never bill anything. Data files are
fetched from this public repo at runtime.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict

try:
    import streamlit as st
except ImportError:  # allows importing the logic without streamlit installed
    st = None

CATEGORIES = [
    "service_speed",
    "staff_behavior",
    "product_quality",
    "pricing_billing",
    "cleanliness_ambience",
    "other",
]
SENTIMENTS = ["negative", "neutral", "positive"]

# Verified free on OpenRouter (2026-10-29); hardcoded so public use can never bill.
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RAW_BASE = "https://raw.githubusercontent.com/pranav8764/cynbit-ai-opportunity-assessment/main"
MAX_CHARS = 2000

SYSTEM_PROMPT = """You are a review-analysis component in a reputation-management pipeline.
Extract every distinct issue or praise point in the customer's review as an "aspect".

Rules:
- Use ONLY these categories: {categories}
- One review can have several aspects (e.g. praise for product, complaint about staff).
- "evidence" must be a short verbatim quote copied from the review.
- "severity" is 1 (trivial) to 5 (serious) and rates the aspect, not the whole review.
- Reply with JSON only, in exactly this shape:
  {{"aspects": [{{"category": "...", "sentiment": "negative|neutral|positive", "severity": 1, "evidence": "..."}}]}}
""".strip()


def build_messages(review):
    user = (
        f"Branch: {review['branch']}\n"
        f"Date: {review['date']}\n"
        f"Star rating: {review['rating']}/5\n"
        f"Review: {review['text']}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(categories=", ".join(CATEGORIES))},
        {"role": "user", "content": user},
    ]


def call_openrouter(messages, api_key, model=DEFAULT_MODEL, timeout=60):
    """One chat-completions call. Returns (raw_text, model_that_served)."""
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body["choices"][0]["message"]["content"], body.get("model")


def parse_classification(raw):
    """Parse the model's reply into validated aspect dicts."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    data = json.loads(text[start:end + 1])
    aspects = data.get("aspects")
    if not isinstance(aspects, list):
        raise ValueError("model output has no 'aspects' list")
    cleaned = []
    for a in aspects:
        if not isinstance(a, dict):
            continue
        category = a.get("category") if a.get("category") in CATEGORIES else "other"
        sentiment = a.get("sentiment") if a.get("sentiment") in SENTIMENTS else "neutral"
        try:
            severity = int(a.get("severity", 3))
        except (TypeError, ValueError):
            severity = 3
        cleaned.append({
            "category": category,
            "sentiment": sentiment,
            "severity": max(1, min(5, severity)),
            "evidence": str(a.get("evidence", ""))[:200],
        })
    return cleaned


def summarize(classified):
    """Per-branch issue ranking (plain analytics over extracted aspects)."""
    by_branch = defaultdict(lambda: defaultdict(
        lambda: {"negative": 0, "neutral": 0, "positive": 0,
                 "sev_sum": 0, "sev_n": 0, "evidence": []}))
    for item in classified:
        for asp in item["aspects"]:
            slot = by_branch[item["branch"]][asp["category"]]
            slot[asp["sentiment"]] += 1
            if asp["sentiment"] == "negative":
                slot["sev_sum"] += asp["severity"]
                slot["sev_n"] += 1
                if asp["evidence"]:
                    slot["evidence"].append(asp["evidence"])
    summary = {}
    for branch, cats in by_branch.items():
        issues = []
        for cat, s in cats.items():
            avg_sev = round(s["sev_sum"] / s["sev_n"], 2) if s["sev_n"] else None
            issues.append({
                "category": cat,
                "negative_mentions": s["negative"],
                "avg_negative_severity": avg_sev,
                "priority_score": round(s["sev_sum"], 2),
                "evidence_examples": s["evidence"][:3],
            })
        issues.sort(key=lambda x: x["priority_score"], reverse=True)
        summary[branch] = issues
    return summary


def get_server_key():
    if st is not None:
        try:
            key = st.secrets.get("OPENROUTER_API_KEY", "")
            if key:
                return key.strip()
        except Exception:
            pass
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def get_api_key():
    """Visitor-supplied session key wins; otherwise the deployment secret.

    Either way the model slug is hardcoded to a :free model, so no request this
    app makes can ever use a paid model.
    """
    if st is not None:
        try:
            visitor = st.session_state.get("visitor_key", "").strip()
            if visitor:
                return visitor
        except Exception:
            pass
    return get_server_key()


def load_sample_data():
    """Fetch the sample reviews and canned fixtures from this repo."""
    def get(path):
        with urllib.request.urlopen(f"{RAW_BASE}/{path}", timeout=20) as resp:
            return json.loads(resp.read().decode())
    return get("data/reviews_synthetic.json"), get("data/mock_classifications.json")


LIVE_CUSTOM_LIMIT = 8   # per browser session
LIVE_BATCH_LIMIT = 2    # per browser session


def _bump(counter, limit):
    if st is None:
        return True
    try:
        n = st.session_state.get(counter, 0)
        if n >= limit:
            return False
        st.session_state[counter] = n + 1
    except Exception:
        pass
    return True


def analyze_custom(branch, rating, text):
    branch = (branch or "").strip() or "Unspecified branch"
    text = (text or "").strip()
    if not text:
        return "Paste a review first.", []
    if len(text) > MAX_CHARS:
        return f"Review is {len(text)} characters; the demo limit is {MAX_CHARS}.", []
    if not _bump("live_custom_n", LIVE_CUSTOM_LIMIT):
        return (f"This session has reached its {LIVE_CUSTOM_LIMIT} live-analysis demo "
                "limit. Reload the app or paste your own OpenRouter key."), []
    api_key = get_api_key()
    if not api_key:
        return ("Live analysis is not configured on this deployment right now. "
                "The sample-batch tab can still run with canned fixtures."), []
    review = {"branch": branch, "date": "live-demo", "rating": int(rating), "text": text}
    try:
        raw, served = call_openrouter(build_messages(review), api_key)
        aspects = parse_classification(raw)
    except urllib.error.HTTPError as e:
        return (f"Model API returned HTTP {e.code}. The free model can be rate-limited "
                "at busy times; wait a moment and try again."), []
    except Exception as e:
        return f"Could not analyze this review: {e}", []
    rows = [{"category": a["category"], "sentiment": a["sentiment"],
             "severity": a["severity"], "evidence": a["evidence"]} for a in aspects]
    return f"Analyzed by `{served or DEFAULT_MODEL}` via OpenRouter.", rows


def run_sample(mode):
    try:
        reviews, canned = load_sample_data()
    except Exception as e:
        return f"Could not load the sample data from the repo: {e}", [], []
    classified = []
    if mode.startswith("Live"):
        if not _bump("live_batch_n", LIVE_BATCH_LIMIT):
            return (f"This session has reached its {LIVE_BATCH_LIMIT} live-batch demo "
                    "limit. Try the canned-fixtures mode, or reload."), [], []
        api_key = get_api_key()
        if not api_key:
            return ("Live model is not configured on this deployment; "
                    "switch to canned fixtures."), [], []
        try:
            for r in reviews:
                raw, _ = call_openrouter(build_messages(r), api_key)
                classified.append({"review_id": r["id"], "branch": r["branch"],
                                   "aspects": parse_classification(raw)})
        except urllib.error.HTTPError as e:
            return (f"Model API returned HTTP {e.code} mid-batch. The free model can be "
                    "rate-limited at busy times; try the canned-fixtures mode meanwhile."), [], []
        except Exception as e:
            return f"Batch failed: {e}", [], []
        note = f"Live run: {len(reviews)} reviews classified by `{DEFAULT_MODEL}` via OpenRouter."
    else:
        for r in reviews:
            classified.append({"review_id": r["id"], "branch": r["branch"],
                               "aspects": canned[r["id"]]})
        note = ("Canned-fixtures run: replays saved model responses against the same "
                "pipeline (proves the pipeline, not the model).")
    summary = summarize(classified)
    summary_rows = []
    for branch, issues in summary.items():
        for i in issues:
            if i["negative_mentions"] == 0:
                continue
            summary_rows.append({
                "branch": branch, "category": i["category"],
                "negative_mentions": i["negative_mentions"],
                "avg_negative_severity": i["avg_negative_severity"],
                "priority_score": i["priority_score"],
                "evidence_examples": " | ".join(i["evidence_examples"])})
    aspect_rows = []
    for item in classified:
        for a in item["aspects"]:
            aspect_rows.append({
                "branch": item["branch"], "review_id": item["review_id"],
                "category": a["category"], "sentiment": a["sentiment"],
                "severity": a["severity"], "evidence": a["evidence"]})
    return note, summary_rows, aspect_rows


def main():
    st.set_page_config(page_title="Review issue miner - live demo")
    st.title("Cross-review issue mining - live demo")
    st.markdown(
        "Proof of concept for the Cynbit AI/ML Engineer Intern assessment: extend a "
        "review-management product with *cross-review* issue mining. An LLM extracts "
        "structured aspects (category, sentiment, severity, verbatim evidence) from "
        "free-text reviews; plain analytics then rank each branch's top issues by "
        "frequency x severity. Code and reports: "
        "[github.com/pranav8764/cynbit-ai-opportunity-assessment]"
        "(https://github.com/pranav8764/cynbit-ai-opportunity-assessment)")
    with st.expander("Use your own OpenRouter key (optional)"):
        st.text_input("OpenRouter API key", type="password", key="visitor_key",
                      help="Used only for this browser session, never stored or shown. "
                           "Leave empty to use the demo's shared free-model quota.")
    tab1, tab2 = st.tabs(["Analyze your own review", "Run the 15-review sample batch"])
    with tab1:
        branch = st.text_input("Branch / location", value="Jaipur C-Scheme")
        rating = st.slider("Star rating", 1, 5, 2)
        text = st.text_area("Review text (max 2000 characters)", height=140)
        if st.button("Analyze review", type="primary"):
            with st.spinner("Asking the model..."):
                note, rows = analyze_custom(branch, rating, text)
            st.markdown(note)
            if rows:
                st.dataframe(rows, use_container_width=True)
    with tab2:
        live_label = "Live LLM (OpenRouter free model)"
        mock_label = "Canned fixtures (offline replay)"
        mode = st.radio("Mode", [mock_label, live_label], index=0)
        st.caption("Live mode sends the 15 synthetic reviews from the repo to the free "
                   "model one by one (about a minute). Canned mode replays saved "
                   "responses instantly.")
        if st.button("Run batch", type="primary"):
            with st.spinner("Running the pipeline..."):
                note, summary_rows, aspect_rows = run_sample(mode)
            st.markdown(note)
            if summary_rows:
                st.subheader("Per-branch issue ranking")
                st.dataframe(summary_rows, use_container_width=True)
                st.subheader("All extracted aspects")
                st.dataframe(aspect_rows, use_container_width=True)
    st.caption(f"Live calls use the hardcoded OpenRouter free model `{DEFAULT_MODEL}`; "
               "the API key is stored only as a deployment secret and a `:free` model "
               "can never bill. Inputs are capped at 2000 characters per review.")


if st is not None:
    main()

"""Review issue miner PoC (Cynbit AI/ML intern assessment).

Idea: extend a reputation-management product (like Cynbit's Revio) with
cross-review issue mining. An LLM API turns free-text reviews into structured
aspects (fixed categories + sentiment + severity + evidence quote); plain
Python analytics then rank each branch's top issues.

Live mode calls OpenRouter (default model: typesafe/jev-router).
Mock mode (--mock) replays canned responses so the pipeline and tests run
with no API key and no network.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

CATEGORIES = [
    "service_speed",
    "staff_behavior",
    "product_quality",
    "pricing_billing",
    "cleanliness_ambience",
    "other",
]
SENTIMENTS = ["negative", "neutral", "positive"]

# Candidate default: verified free on OpenRouter (2026-10-29), structured_outputs supported.
# Any OpenRouter slug works via --model or OPENROUTER_MODEL.
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"
JEV_ROUTER_MODEL = "typesafe/jev-router"  # optional alternative; routes to variable models, cost not fixed
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

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


def call_openrouter(messages, api_key, model=DEFAULT_MODEL, cost_tier="low", timeout=60):
    """One chat-completions call. Returns (raw_text, model_that_served)."""
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    # cost_tier is a Jev Router routing preference, not a price cap.
    if model == JEV_ROUTER_MODEL and cost_tier:
        payload["plugins"] = [{"id": "jev-router", "cost_tier": cost_tier}]
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
    """Parse the model's reply into validated aspect dicts.

    Tolerates code fences and surrounding prose; unknown categories and
    sentiments fall back to safe defaults instead of crashing the pipeline.
    """
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
    """Per-branch issue ranking. This step is deliberately plain analytics:
    the ML part ended once aspects were structured."""
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
                "priority_score": round(s["sev_sum"], 2),  # raw severity sum; rounding avg first would drift (3.67*3=11.01 vs true 11)
                "evidence_examples": s["evidence"][:3],
            })
        issues.sort(key=lambda x: x["priority_score"], reverse=True)
        summary[branch] = issues
    return summary


def classify_all(reviews, mode, api_key=None, model=DEFAULT_MODEL,
                 cost_tier="low", mock_path=None):
    out, served = [], set()
    if mode == "mock":
        canned = json.loads(Path(mock_path).read_text(encoding="utf-8"))
        for r in reviews:
            out.append({"review_id": r["id"], "branch": r["branch"],
                        "aspects": canned[r["id"]], "source": "mock"})
        return out, []
    for r in reviews:
        raw, serving_model = call_openrouter(build_messages(r), api_key, model, cost_tier)
        if serving_model:
            served.add(serving_model)
        out.append({"review_id": r["id"], "branch": r["branch"],
                    "aspects": parse_classification(raw), "source": "live"})
    return out, sorted(served)


def print_summary(summary):
    for branch, issues in summary.items():
        print(f"\n== {branch} ==")
        for i in issues:
            if i["negative_mentions"] == 0:
                continue
            print(f"  {i['category']}: {i['negative_mentions']} negative mentions, "
                  f"avg severity {i['avg_negative_severity']}, "
                  f"priority {i['priority_score']}")
            for ev in i["evidence_examples"]:
                print(f'    - "{ev}"')


def main(argv=None):
    p = argparse.ArgumentParser(description="Review issue miner PoC")
    p.add_argument("--input", default="data/reviews_synthetic.json")
    p.add_argument("--out", default="out")
    p.add_argument("--mock", action="store_true",
                   help="replay canned responses from data/mock_classifications.json (no API key)")
    p.add_argument("--mock-file", default="data/mock_classifications.json")
    p.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
    p.add_argument("--cost-tier", default=os.environ.get("JEV_COST_TIER", "low"),
                   help="Jev Router cost tier preference: low|medium|high (not a price cap)")
    args = p.parse_args(argv)

    reviews = json.loads(Path(args.input).read_text(encoding="utf-8"))
    mode = "mock" if args.mock else "live"
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if mode == "live" and not api_key:
        sys.exit("OPENROUTER_API_KEY is not set. Use --mock for an offline run.")

    classified, served = classify_all(reviews, mode, api_key, args.model,
                                      args.cost_tier, args.mock_file)
    summary = summarize(classified)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classifications.json").write_text(
        json.dumps(classified, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"mode={mode} reviews={len(reviews)}")
    if served:
        print(f"served by model(s): {', '.join(served)}")
    print_summary(summary)
    print(f"\nwrote {out_dir}/classifications.json and {out_dir}/summary.json")


if __name__ == "__main__":
    main()

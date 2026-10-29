import json

import pytest

from poc.review_miner import (CATEGORIES, classify_all, parse_classification,
                              summarize)


def test_parse_plain_json():
    raw = '{"aspects": [{"category": "service_speed", "sentiment": "negative", "severity": 4, "evidence": "waited 40 min"}]}'
    aspects = parse_classification(raw)
    assert aspects == [{"category": "service_speed", "sentiment": "negative",
                        "severity": 4, "evidence": "waited 40 min"}]


def test_parse_strips_code_fence_and_prose():
    raw = 'Sure! Here you go:\n```json\n{"aspects": [{"category": "staff_behavior", "sentiment": "positive", "severity": 1, "evidence": "polite"}]}\n```\nHope that helps.'
    aspects = parse_classification(raw)
    assert aspects[0]["category"] == "staff_behavior"


def test_parse_unknown_category_and_bad_severity_fall_back():
    raw = '{"aspects": [{"category": "vibes", "sentiment": "angry", "severity": "high", "evidence": "x"}]}'
    aspects = parse_classification(raw)
    assert aspects[0]["category"] == "other"
    assert aspects[0]["sentiment"] == "neutral"
    assert aspects[0]["severity"] == 3


def test_parse_rejects_non_json():
    with pytest.raises(ValueError):
        parse_classification("I could not analyze this review.")


def test_summarize_ranks_by_mentions_times_severity():
    classified = [
        {"review_id": "a", "branch": "B1", "aspects": [
            {"category": "service_speed", "sentiment": "negative", "severity": 4, "evidence": "slow"},
            {"category": "staff_behavior", "sentiment": "positive", "severity": 1, "evidence": "nice"}]},
        {"review_id": "b", "branch": "B1", "aspects": [
            {"category": "service_speed", "sentiment": "negative", "severity": 2, "evidence": "wait"},
            {"category": "pricing_billing", "sentiment": "negative", "severity": 5, "evidence": "overcharged"}]},
    ]
    summary = summarize(classified)
    issues = summary["B1"]
    # service_speed: 2 negatives, avg sev 3 -> priority 6; pricing: 1 x 5 = 5
    assert issues[0]["category"] == "service_speed"
    assert issues[0]["negative_mentions"] == 2
    assert issues[0]["avg_negative_severity"] == 3.0
    assert issues[0]["priority_score"] == 6.0
    # positive staff mention is counted but not in the negative ranking fields
    staff = next(i for i in issues if i["category"] == "staff_behavior")
    assert staff["negative_mentions"] == 0
    assert staff["priority_score"] == 0


def test_mock_pipeline_end_to_end(tmp_path):
    reviews = [{"id": "x1", "branch": "Test Branch", "date": "2026-10-01",
                "rating": 2, "text": "ignored in mock mode"}]
    canned = {"x1": [{"category": "cleanliness_ambience", "sentiment": "negative",
                      "severity": 5, "evidence": "dirty floor"}]}
    mock_file = tmp_path / "mock.json"
    mock_file.write_text(json.dumps(canned))
    in_file = tmp_path / "reviews.json"
    in_file.write_text(json.dumps(reviews))

    classified, served = classify_all(reviews, "mock", mock_path=str(mock_file))
    assert served == []
    assert classified[0]["source"] == "mock"
    summary = summarize(classified)
    top = summary["Test Branch"][0]
    assert top["category"] == "cleanliness_ambience"
    assert top["priority_score"] == 5.0


def test_categories_are_fixed_and_known():
    # the prompt contract: a small fixed category set, not free-form labels
    assert CATEGORIES == ["service_speed", "staff_behavior", "product_quality",
                          "pricing_billing", "cleanliness_ambience", "other"]

def test_summarize_priority_uses_unrounded_severity_sum():
    """Regression: priority must come from the raw severity sum, not the
    rounded display average (3.67*3 = 11.01 would be wrong; true is 11)."""
    classified = [
        {"review_id": "a", "branch": "B1", "aspects": [
            {"category": "product_quality", "sentiment": "negative", "severity": 4, "evidence": "x"}]},
        {"review_id": "b", "branch": "B1", "aspects": [
            {"category": "product_quality", "sentiment": "negative", "severity": 4, "evidence": "y"}]},
        {"review_id": "c", "branch": "B1", "aspects": [
            {"category": "product_quality", "sentiment": "negative", "severity": 3, "evidence": "z"}]},
    ]
    issue = summarize(classified)["B1"][0]
    assert issue["avg_negative_severity"] == 3.67  # display rounding stays
    assert issue["priority_score"] == 11  # not 11.01

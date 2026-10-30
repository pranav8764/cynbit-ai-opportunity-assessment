# Evaluation plan (baseline vs model)

Status: PLAN ONLY. No measurements have been run yet. Numbers below are targets, not results.

## Question
Does an LLM add measurable value over simple rules for turning free-text
reviews into issue categories? If not, the honest answer is to ship the rules.

## Baseline (ordinary approach)
Keyword/dictionary tagging: small hand-written keyword lists per category
(e.g. service_speed: "wait", "slow", "forever"; pricing_billing: "bill",
"charged", "price"). Severity from star rating. Pure Python, no model.

## Model
The API classifier in `poc/review_miner.py` (fixed categories + structured
output). Default model `nvidia/nemotron-3.5-super-vl:free` (a $0 OpenRouter model),
configurable via `OPENROUTER_MODEL`.

## Method
1. Hand-label the 15 synthetic reviews (plus any added later) with their true
   aspects. Cheap because the dataset is synthetic and small.
2. Run baseline and model over the same reviews.
3. Compare precision/recall of (category, sentiment) per aspect, and show
   concrete reviews where the two disagree (paraphrase clusters, mixed
   praise+complaint, implicit complaints).
4. Report whichever wins. If the baseline matches the model on this small
   clean set, say so and scope the model's value to the harder phrasings.

## Honesty rules
- Synthetic data shows the pipeline works; it is not evidence about real
  customer reviews.
- No superiority claim (model over baseline, or one model over another)
  without the measurements above.
- Live-model outputs are only reported from actual API runs, never from the
  mock fixtures.


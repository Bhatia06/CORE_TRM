# Implementation Plan: Representation-Consistent Session Recommendation

## 0. Project Metadata

```yaml
project_name: clickstream-core
project_type: machine_learning_research_and_demo
domain: ecommerce_session_based_recommendation
primary_dataset: Diginetica clickstream
primary_paper: "CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space (SIGIR 2022)"
prediction_task: next_item_prediction
primary_metrics:
  - Recall@20
  - MRR@20
main_language: Python
main_framework: PyTorch
final_demo: local web app with a recommendation API
```

## 1. Context and Problem Statement

An e-commerce visitor may browse anonymously, so the system has no reliable long-term user profile. The system observes only the products clicked during the active browsing session and must recommend the next product.

Example session:

```text
clicked_items = [running_shoes, sports_socks, running_shoes]
model_output = ranked list of likely next products
```

This is a **session-based next-item recommendation** project. It must not be described as a long-term personalized recommender because its primary model uses the current session only.

## 2. Research Goal

Primary research question:

> Does a representation-consistent session encoder improve next-item ranking compared with simple non-neural session recommendation baselines under one strict temporal evaluation protocol?

Secondary research question:

> When does learned weighting in CORE-trm help over simple averaging in CORE-ave, particularly across short and long observed session prefixes?

Optional extension question, only after baseline analysis:

> Can a simple behavior-aware weighting feature improve CORE without using future information?

## 3. Core Method: CORE

CORE represents the current session as a weighted sum of clicked-item embeddings:

```text
session_embedding = sum(alpha_i * item_embedding_i)
```

Definitions:

```yaml
item_embedding_i: trainable vector for clicked item i
alpha_i: importance weight assigned to clicked item i
CORE_ave: alpha_i = 1 / session_length
CORE_trm: Transformer computes alpha_i from the observed sequence
decoder: cosine similarity between session embedding and candidate item embeddings
training_objective: temperature-scaled cross-entropy ranking loss
```

Important claims to preserve:

- CORE constrains a session representation to be a combination of clicked-item embeddings.
- This is a modeling choice that may improve ranking. Do not claim that nonlinear encoders make dot products mathematically invalid.
- Do not claim that CORE or any extension is superior until measured under the same protocol.

## 4. Dataset and Data Contract

### 4.1 Input File

```yaml
raw_file: data/raw/diginetica.csv
separator: ";"
required_columns:
  - sessionId
  - userId
  - itemId
  - timeframe
  - eventdate
row_definition: one product-view event
session_definition: all events with the same sessionId
sequence_order: ascending timeframe within each session
```

### 4.2 Raw Data Rules

```yaml
ignore_file: data/raw/diginetica.corrupt.csv
never_train_on: corrupt_or_partially_downloaded_files
never_use_as_input:
  - future clicks in the same session
  - target item when forming the input prefix
  - test/validation statistics for training-only preprocessing
```

### 4.3 EDA Requirements

Generate and save an EDA report with:

- Number of events, sessions, items, and known user IDs.
- Missing values and exact duplicate events.
- Session-length distribution: count, mean, median, percentiles, maximum.
- Item-frequency distribution and number of items with fewer than five interactions.
- Date range and events/sessions per day.
- Check that each session is sorted by `timeframe` after preprocessing.
- Number of sessions crossing multiple `eventdate` values and decision for handling them.

Required plots:

```yaml
plots:
  - session_length_histogram
  - log_scaled_item_frequency_histogram
  - sessions_per_day_line_chart
  - cumulative_share_of_clicks_by_item_popularity
```

## 5. Experiment Protocol

### 5.1 Temporal Split

Use whole sessions. Never split prefixes of one source session across train, validation, and test.

```yaml
split_method: chronological_by_session_start_date
target_ratio:
  train: 0.80
  validation: 0.10
  test: 0.10
tie_handling: keep every session from the same calendar date in the same split
primary_protocol_name: strict_temporal_protocol
```

### 5.2 Leakage-Safe Preprocessing

Run these transformations in this order:

1. Read the raw data.
2. Drop malformed rows and exact duplicate events.
3. Sort each session by `timeframe` in ascending order.
4. Assign whole sessions to train, validation, and test by time.
5. Learn item frequency and item-to-index mapping using train events only.
6. Keep train items that occur at least `min_train_item_frequency = 5` times.
7. Remove unknown/rare items from validation and test prefixes and targets, then report the removed-event and removed-target rate.
8. Drop sessions that have fewer than two remaining events.
9. Create prefix-target pairs separately within each split.

Example:

```text
session: [A, B, C, D]
examples:
  input=[A]       target=B
  input=[A, B]    target=C
  input=[A, B, C] target=D
```

Do not use the test set to decide which items are frequent enough for the primary protocol.

### 5.3 Candidate Set and Evaluation Coverage

```yaml
candidate_items: all retained train-vocabulary items
unseen_test_target_policy: exclude_from_ranking_metrics_and_report_coverage
report:
  - number of original test targets
  - number of evaluable test targets
  - evaluation_coverage_percent
```

## 6. Baselines

Implement and evaluate all baselines on the identical split, candidate vocabulary, and metrics.

| ID | Model | Purpose |
|---|---|---|
| B0 | POP | Recommend the globally most-clicked training items. |
| B1 | Last-item transition | Recommend items most often observed after the most recent clicked item in training. |
| B2 | Session popularity | Rank items by frequency in the observed session, then backfill with POP. |
| B3 | Session KNN or VSKNN | Strong non-neural session baseline; implement if time permits. |
| M1 | CORE-ave | Main paper reproduction baseline. |
| M2 | CORE-trm | Learned item-weighting version of CORE. |

Minimum required comparison: `B0`, `B1`, `M1`, `M2`.

## 7. Metrics

For each test prefix, produce a ranked list of 20 candidate items.

```text
Recall@20 = fraction of cases where target appears in top 20
MRR@20 = average(1 / rank_of_target) if target is in top 20, else 0
```

Also record:

```yaml
efficiency_metrics:
  - training_time_per_epoch_seconds
  - total_training_time_seconds
  - inference_latency_ms_per_prefix
reliability_metrics:
  - mean_and_standard_deviation_over_3_random_seeds
  - validation_best_epoch
```

## 8. Model Implementation Milestones

### Milestone A: Data pipeline

```yaml
input: data/raw/diginetica.csv
outputs:
  - data/processed/train.parquet
  - data/processed/validation.parquet
  - data/processed/test.parquet
  - data/processed/item_mapping.json
  - reports/eda_summary.json
  - reports/data_split_summary.json
acceptance_checks:
  - no session appears in more than one split
  - every sequence is ordered by timeframe
  - each target appears after its input prefix
  - train vocabulary is the only candidate vocabulary
```

### Milestone B: Evaluation utilities

```yaml
components:
  - top_k_ranking
  - recall_at_k
  - mrr_at_k
  - grouped_metrics_by_prefix_length
acceptance_checks:
  - metric calculations hand-checked on a tiny synthetic example
```

### Milestone C: Baselines

```yaml
components:
  - popularity_recommender
  - transition_recommender
  - session_popularity_recommender
acceptance_checks:
  - all models return exactly 20 unique valid candidates
  - results saved to reports/baseline_metrics.json
```

### Milestone D: CORE-ave

```yaml
components:
  - item_embedding_table
  - masked_mean_pooling
  - L2_normalized_session_and_item_vectors
  - temperature_scaled_candidate_scores
  - cross_entropy_training_loop
acceptance_checks:
  - padded values do not affect session embedding
  - loss decreases on a small overfit subset
  - validation metrics exceed or match sensible baseline behavior
```

### Milestone E: CORE-trm

```yaml
components:
  - positional_embeddings
  - causal_transformer_encoder
  - attention_weight_projection
  - masked_softmax_over_observed_positions
  - weighted_sum_of_original_item_embeddings
acceptance_checks:
  - weights sum to 1 across valid positions
  - padded positions receive zero weight
  - output session vector uses original item embeddings, not transformed vectors
```

## 9. Analysis Plan

Run every method under the same protocol.

Required result tables:

```yaml
tables:
  - overall_recall20_and_mrr20
  - metrics_by_prefix_length: [1, 2_to_3, 4_to_6, 7_plus]
  - metrics_by_target_item_popularity: [head, torso, tail]
  - training_time_and_inference_latency
  - three_seed_summary_for_neural_models
```

Required qualitative analysis:

- Inspect at least 15 correct and incorrect test recommendations.
- Identify cases involving repeated items, short prefixes, and long prefixes.
- State whether CORE-trm's extra cost is justified by the observed improvement.

## 10. Optional Extension: Behavior-Aware Weighting

Only begin after Milestones A through E are complete and the analysis identifies a concrete weakness.

### Candidate Extension

```yaml
name: recency_and_repeat_aware_CORE
input_features_available_before_prediction:
  - click_position
  - recency_within_observed_prefix
  - repeated_item_count_within_observed_prefix
possible_weight_formula: "alpha_i = softmax(transformer_score_i + beta_1 * recency_i + beta_2 * repeat_count_i)"
required_comparisons:
  - CORE_ave
  - fixed_recency_weighting
  - CORE_trm
  - proposed_behavior_aware_weighting
required_ablation:
  - remove_recency_feature
  - remove_repeat_count_feature
```

Restrictions:

- Do not call inter-click gaps dwell time unless the dataset truly measures dwell time.
- Do not use the delay between the final observed click and the target click, because it is unavailable at recommendation time.
- Do not claim novelty or improvement before measured results across multiple seeds.

## 11. Frontend and Deployment

Build only after the selected model is measured.

```yaml
backend:
  framework: FastAPI
  endpoint: POST /recommend
  input: '{"session_item_ids": [12, 87, 12]}'
  output: '{"recommendations": [{"item_id": 45, "score": 0.93}]}'
frontend:
  purpose: demonstrate current-session clicks and top-k recommendations
  required_features:
    - clickable catalog sample
    - visible current-session sequence
    - recommendation panel
    - reset-session button
product_metadata_policy: use explicit placeholders if Diginetica lacks product names/images
```

Do not claim production latency targets, Qdrant, Redis, ONNX, or vector-database retrieval unless implemented and measured.

## 12. Suggested Repository Structure

```text
clickstream-core/
  data/
    raw/
    processed/
  src/
    data/
      prepare_data.py
      datasets.py
    models/
      core_ave.py
      core_trm.py
      baselines.py
    evaluation/
      metrics.py
      evaluate.py
    training/
      train.py
      config.py
  reports/
  tests/
  app/
    api.py
    frontend/
  implementation_plan.md
  README.md
  requirements.txt
```

## 13. Final Deliverables

```yaml
required:
  - reproducible data-preparation script
  - EDA report and plots
  - saved temporal splits and item mapping
  - baseline implementations and metrics
  - CORE-ave implementation and trained checkpoint
  - CORE-trm implementation and trained checkpoint
  - comparison and error-analysis report
  - optional extension with ablation, only if completed
  - local FastAPI inference endpoint
  - frontend demo
  - README with setup, data source, commands, results, and limitations
```

## 14. Definition of Done

The project is complete when all of the following are true:

- A new user can reproduce the experiment from raw data using documented commands.
- Every model uses the same temporal split and candidate vocabulary.
- Results report Recall@20, MRR@20, evaluation coverage, and variability across seeds.
- Claims distinguish measured findings from hypotheses.
- The frontend calls the selected trained model and returns actual ranked recommendations.
- The final report explains limitations: anonymous sessions, warm-item evaluation, item-ID-only features, and offline metrics.

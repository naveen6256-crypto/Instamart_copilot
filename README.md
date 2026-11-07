# Instamart Customer Intelligence & Support Copilot

## Overview

[WRITE: 3 to 4 sentences. What the project is (internal tool for Instamart: trusted data, customer segments and lapse prediction, support ticket triage, policy assistant and refund agent), that all data is synthetic and fictional, that all amounts are in INR, and that every language-model call uses the offline mock_llm.py.]

Repository layout:

| Path | Contents |
|---|---|
| `generate_data.py`, `mock_llm.py` | Supplied files, unmodified |
| `src/Part_A/` | `Data_cleaning.py`, `Data_validation.py`, `Sql_analysis.py`, `Pandas_analysis.py` |
| `src/Part_B/` | `Segments.py`, `Lapse.py` |
| `src/Part_C/` | `Triage.py` |
| `src/Part_D/` | `Policy_rag.py` and the agent script |
| `sql/queries.sql` | The seven SQL queries |
| `data/`, `outputs/`, `state/` | Raw data, clean tables and database, model outputs, agent checkpoints |

## Reproduce

Run from the repository root, in this order:

1. `python -m venv .venv`
2. Activate it: `.venv\Scripts\Activate.ps1` (Windows PowerShell) or `source .venv/bin/activate` (Mac/Linux)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python src/Part_A/Data_cleaning.py`
6. `python src/Part_A/Data_validation.py`
7. `python src/Part_A/Sql_analysis.py`
8. `python src/Part_A/Pandas_analysis.py`
9. `python src/Part_B/Segments.py`
10. `python src/Part_B/Lapse.py`
11. `python src/Part_C/Triage.py`
12. [ADD the Part D commands here when Part D is finished]

Part B's `Lapse.py` reads `outputs/customer_segments.csv` from `Segments.py`, and Part C reads the same file for routing, so keep this order.

## Part A

Data Foundation: Clean, Store, Query.

### Audit of the raw data

| Table | Shape | Missing values |
|---|---|---|
| orders | 13,084 x 9 | order_amount 53 (0.41%), delivery_minutes 2,541 (19.42%) |
| customers | 1,200 x 5 | age 61 (5.08%), gender 30 (2.50%) |
| products | 40 x 4 | none |
| tickets | 164 x 5 | none (read with keep_default_na=False) |

Data types: [ADD from your audit output. Text columns are strings; quantity and discount_pct are integers; delivery_minutes and age are floats; price_inr is an integer.]

| Audit figure | Value |
|---|---|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Pending rows | 656 |
| Distinct order_id | 12,309 |
| Non-numeric order_amount | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank age | 61 |
| Age outside 10-100 | 6 |
| Raw city spellings | 24 |
| Raw gender spellings (non-blank) | 10 |
| Blank gender | 30 |

### Cleaning rules

Customers:

| Rule | Effect |
|---|---|
| City stripped and title-cased | 6 cities remain |
| Gender mapped to male / female / unknown | blank becomes unknown (30) |
| Ages outside 10-100 set to missing | 6 values |
| Missing ages filled with the city median (groupby and transform) | 67 values |
| signup_date converted to datetime | all rows |

Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30; 0 missing ages, all ages between 18 and 60.

Orders, in this order:

| Rule | Effect |
|---|---|
| 1. order_ts converted to datetime | all rows |
| 2. Exact duplicates removed | 119 rows |
| 3. Latest order_ts kept per order_id | 656 earlier records removed |
| 4. Unparseable amounts filled with price x quantity x (1 - discount/100) | 112 values |
| 5a. Negative delivery_minutes set to missing | 105 values |
| 5b. Completed orders: missing delivery filled with the product-category median | 774 values |
| 5c. Cancelled orders left missing (never delivered, so imputing would invent data) | 1,199 values |

Result: 12,309 rows; 11,110 completed; 1,199 cancelled; no pending rows; 0 missing order_amount; completed revenue INR 2,998,668.80.

The counts for rules 4 and 5a are lower than the raw audit counts (120 and 106) because some affected rows were removed earlier as duplicates or older pending records.

### Storage and validation

The clean tables are saved to `data/clean/` as CSV files and to `data/instamart.db` as the tables customers, products and orders (order_ts stored as text, YYYY-MM-DD HH:MM:SS). `Data_validation.py` runs dynamic checks against the raw data, checks that the database matches the CSV files, compares the brief's reference numbers, and confirms no dollar amounts appear in the code or README. It prints ALL CHECKPOINTS PASSED.

### SQL analysis

| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows, total INR 2,998,668.80, largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTE | 550 customers above the average spend of INR 2,552.06 |
| Q6 window functions | 8 rows where RANK() differs from ROW_NUMBER() |
| Q7 running total | 183 daily rows, last cumulative value INR 2,998,668.80 |

Q6 top customer per city:

| City | Customer | Total spend (INR) |
|---|---|---|
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

[WRITE: one sentence on why ties make RANK() differ from ROW_NUMBER(). Hint: tied customers share one rank, but ROW_NUMBER() still gives each a different number.]

### pandas cross-check and exploratory analysis

The pivot table (margins=True) has a grand total of INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Orders |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed order_amount statistics:

| Statistic | Value |
|---|---|
| Mean / median | 269.91 / 135.00 |
| Skewness | 4.61 |
| Q1 / Q3 | 64.80 / 297.00 |
| Lower fence / upper fence | -283.50 / 645.30 |
| Outliers below / above | 0 / 897 |

Largest correlation (absolute value): order_amount and discount_pct, -0.095. The pair order_amount and quantity is +0.087.

Answers to the four questions (at most 200 words in total):

[WRITE: 1) What is this data about? 2) What do the distributions and statistics look like (mean far above median, right-skewed)? 3) Are there outliers or anomalies (897 large orders above the upper fence, none below)? 4) What relationships exist (strongest pair is amount and discount at about -0.095; say that a correlation this small does not establish causation).]

## Part B

Customer Segments and Lapse Prediction.

### RFM table (snapshot 2025-10-01)

| Measure | Value |
|---|---|
| Customers with a completed order | 1,175 |
| Recency median / min / max | 19 / 0 / 180 days |
| Frequency mean / max | 9.46 / 33 |
| Monetary mean / max | INR 2,552.06 / 14,766.55 |

### K-means (StandardScaler, n_init=10, random_state=42)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Chosen K: 4. [WRITE: why. Cover the elbow after K=4, the highest silhouette (0.526), and that four named segments are easy for the business to act on.]

[WRITE: 2 to 3 sentences on why the inertia of K=8 is always lower than that of K=2, yet that alone cannot pick K. Hint: more clusters always sit closer to their points, down to zero when every customer is its own cluster.]

### Segment profile (K=4)

| Segment id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Action |
|---|---|---|---|---|---|---|
| 0 | Occasional Buyers | 436 | 28.86 | 3.77 | 1,057.62 | Small discounts and bundles to raise order frequency |
| 1 | Premium Big Spenders (Premium) | 138 | 24.12 | 6.77 | 7,129.70 | VIP offers, early access to electronics, priority support |
| 2 | Loyal Frequent Buyers | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty points and subscription offers |
| 3 | Lapsing / At-Risk | 182 | 109.04 | 2.20 | 685.99 | Win-back coupons and reminders |

Cluster numbers are only labels with no ordering. The Premium flag goes to the segment with the highest mean monetary value. Output: `outputs/customer_segments.csv`.

PCA (2 components) on the scaled RFM matrix: explained variance ratios 0.634 and 0.218, total 0.852.

### Lapse problem

Cutoff 2025-09-01. The cohort is every customer with at least one completed order before the cutoff. The label lapsed is 1 if the customer has no completed order from 2025-09-01 to 2025-09-30.

| Item | Value |
|---|---|
| Cohort | 1,156 customers |
| Lapsed | 443 (38.32%) |
| Train / test (test_size=0.2, stratified, random_state=42) | 924 / 232 |
| Baseline (always "not lapsed") | accuracy 61.6%, lapsed-class F1 0.00 |

Features: recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure in days.

No data leakage: every feature uses only orders dated before 2025-09-01 (no data from 2025-09-01 onwards entered the features), and the segment labels were not used as features. No encoder, scaler or resampler saw test rows: OneHotEncoder(drop="first") and StandardScaler sit inside a Pipeline fitted on training rows only.

| Preprocessing check | Value |
|---|---|
| Frequency mean in the fitted scaler (training rows) | 8.115 |
| Frequency mean of the full cohort | 8.029 |
| One-hot columns (k-1) | city 5, gender 2, favourite category 4 |
| Missing values in any feature column | none (asserted in code) |

[WRITE: one sentence on why the two frequency means differ. Hint: the scaler saw only the 924 training rows, not the test rows.]

### Tuning and test results (GridSearchCV, cv=5, scoring=f1)

| Model | Best parameters | Best CV F1 |
|---|---|---|
| Logistic regression, C in {0.01, 0.1, 1, 10, 100} | C = 100 | 0.651 |
| Random forest, max_depth in {3, 5, 8}, min_samples_leaf in {5, 10, 20} | depth 8, leaf 10 | 0.681 |

Test set, lapsed class:

| Model | Confusion matrix (TN, FP, FN, TP) | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | 111, 32, 31, 58 | 0.644 | 0.652 | 0.648 |
| Random forest | 113, 30, 31, 58 | 0.659 | 0.652 | 0.655 |

Selected model: random forest (test F1 0.655, above the 0.55 minimum).

[WRITE: 3 to 5 sentences justifying the choice. Mention the cost of missing a lapsing customer (all their future revenue) versus wrongly targeting a loyal one (one small coupon), so recall matters more, and that the forest has the better F1 and precision at equal recall.]

### Class-balance experiment

Lapsed training rows were duplicated at random (with replacement, random_state=42) until both classes had 570 rows. Only the training set was resampled.

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Original training data | 0.644 | 0.652 | 0.648 |
| Balanced training data | 0.589 | 0.854 | 0.697 |

[WRITE: which metric moved most (recall) and what it means for the retention team: more leavers caught, but more loyal customers contacted.]

### Interpretation

| Top 5 random forest features | Importance |
|---|---|
| frequency | 0.280 |
| monetary | 0.271 |
| recency | 0.115 |
| avg_delivery | 0.061 |
| cancel_rate | 0.052 |

| Logistic coefficient | Value |
|---|---|
| recency | +0.040 |
| frequency | -1.698 |
| cancel_rate | -0.124 |

[WRITE: a one-line business reading for each of the five features and three coefficients. Example idea: frequent buyers rarely lapse.]

The selected model is saved to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions exactly. `outputs/retention_list.csv` lists the Premium customers in the test set ranked by predicted lapse probability: 10 rows, of which 1 truly lapsed. The retention list is illustrative and not a leakage-free evaluation, because the Premium flag comes from whole-window RFM that includes the September window defining the lapse label.

## Part C

Support Ticket Triage with Guardrails (`src/Part_C/Triage.py`).

| Guardrail | Result |
|---|---|
| Empty tickets (skipped_empty, never sent to the model) | 4 |
| Long tickets over 400 estimated tokens (flagged_long, never sent) | 4 (734 tokens each) |
| Redactions: emails / phones / order ids | 55 / 91 / 132 (278 in total) |
| Patterns found after re-scanning redacted texts | 0 |
| Prompts containing an email, phone or order id | none (asserted in code) |

Retries: maximum 3 per ticket, exponential backoff with jitter (BASE_DELAY 0.05 s). The run had 19 transient errors, each recovered by one retry (all retry number 0, waits between 0.0453 and 0.0562 seconds), and no permanent error.

Output validation (full run over the 156 eligible tickets):

| Status | Count |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |

### Zero-shot versus few-shot

| | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|---|
| Zero-shot | 147 | 126 | 85.7% | 9,088 | 61.8 |
| Few-shot | 147 | 147 | 100% | 21,568 | 146.7 |

Zero-shot confusion table (rows = true category, columns = predicted):

| | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

[WRITE: 3 to 4 sentences with your numbers on when the extra few-shot tokens are worth paying for. Points: about 85 extra tokens per ticket; zero-shot sent 11 fraud reports to billing; fraud tickets have a 2-hour security deadline; few-shot removed every error.]

### Budget demo and routing

Zero-shot run with TOKEN_BUDGET = 6000: 6,018 tokens spent, 53 tickets ended budget_exhausted. Output: `outputs/triage_budget_demo.csv`.

Final routing (few-shot run, TOKEN_BUDGET = 100000, segments from Part B): `outputs/triage_results.csv`, 164 rows.

| Route | Tickets |
|---|---|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

Priority: 119 NORMAL, 45 HIGH.

Rule-validated outputs: classified_json and classified_keyword. Sent to human review: skipped_empty (4), flagged_long (4) and flagged_invalid_output (9); budget_exhausted tickets also go to human review (none in the final run).

## Part D

[TO BE ADDED after Part D is built: chunking and retrieval results, evaluation with overlap 10 and 0, the seven-scenario result table, the statuses of the two stop-safely demonstrations, the tools executed in the crash-and-resume run, and the ReAct versus plan-and-execute comparison.]

## Limitations and Next Steps

[WRITE: 4 to 6 short points. Required: state that the Premium flag is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation. Other ideas: synthetic data only; a keyword-based mock instead of a real model; TF-IDF matches words, not meaning; the lapse model uses one month as its label.]
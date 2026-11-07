import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")

NUMERIC_COLUMNS = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
                   "avg_delivery", "age", "tenure_days"]
CATEGORY_COLUMNS = ["city", "gender", "fav_category"]

# Numbers from the brief. They are used ONLY to compare, never to compute.
REFERENCE = {
    "cohort customers": 1156,
    "lapsed customers": 443,
    "lapsed rate %": 38.32,
    "train rows": 924,
    "test rows": 232,
    "baseline accuracy %": 61.6,
}


# Prints the value we computed next to the brief's reference, with OK or CHECK.
# tolerance is how far apart they may be and still count as a match.
def compare_with_reference(name, actual, tolerance=0.0):
    expected = REFERENCE[name]
    status = "OK" if abs(actual - expected) <= tolerance else "CHECK"
    print(f"{name}: {actual} (reference {expected}) {status}")


# Task 4: builds one row per customer using ONLY orders before 2025-09-01.
# The label lapsed = 1 means no completed order in September.
def build_cohort():
    orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv("data/clean/products.csv")

    orders_before_cutoff = orders[orders["order_ts"] < CUTOFF]
    completed_before_cutoff = orders_before_cutoff[
        orders_before_cutoff["status"] == "completed"
    ].merge(products[["product_id", "category"]], on="product_id")

    customer_features = completed_before_cutoff.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean"),
    )
    customer_features["recency"] = (CUTOFF - customer_features["last_order"]).dt.days

    # share of orders (with a final status) that were cancelled
    customer_features["cancel_rate"] = orders_before_cutoff.groupby("customer_id")["status"].apply(
        lambda status: (status == "cancelled").mean())

    # most frequent product category (ties go to the first name alphabetically)
    favourite_category = (
        completed_before_cutoff.groupby(["customer_id", "category"]).size()
        .reset_index(name="order_count")
        .sort_values(["customer_id", "order_count", "category"], ascending=[True, False, True])
        .drop_duplicates("customer_id").set_index("customer_id")["category"])
    customer_features["fav_category"] = favourite_category

    customer_features = customer_features.reset_index().merge(
        customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")
    customer_features["tenure_days"] = (CUTOFF - customer_features["signup_date"]).dt.days

    # label: no completed order in September
    september_orders = orders[(orders["status"] == "completed") &
                              (orders["order_ts"] >= CUTOFF) &
                              (orders["order_ts"] < WINDOW_END)]
    customer_features["lapsed"] = (
        ~customer_features["customer_id"].isin(september_orders["customer_id"])).astype(int)

    print("=== LAPSE COHORT ===")
    compare_with_reference("cohort customers", len(customer_features))
    compare_with_reference("lapsed customers", int(customer_features["lapsed"].sum()))
    compare_with_reference("lapsed rate %",
                           round(customer_features["lapsed"].mean() * 100, 2), tolerance=0.01)
    return customer_features


# Returns precision, recall and F1 for the lapsed class (the minority class).
def lapsed_class_scores(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    return precision, recall, f1


# Scales the numeric columns and one-hot encodes the text columns. Inside a
# Pipeline it is fitted on training rows only, so test rows never influence it.
def make_preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ("categorical", OneHotEncoder(drop="first"), CATEGORY_COLUMNS),
    ])


# Task 6a: the baseline that always predicts "not lapsed".
def baseline(labels_test):
    predictions = [0] * len(labels_test)
    _, _, f1 = lapsed_class_scores(labels_test, predictions)
    print("\n=== BASELINE (always 'not lapsed') ===")
    compare_with_reference("baseline accuracy %",
                           round(accuracy_score(labels_test, predictions) * 100, 1), tolerance=0.05)
    print(f"lapsed F1: {f1:.2f} (reference 0)")


# Task 6b-d: tunes both models with GridSearchCV (5-fold, scoring F1) and
# evaluates the tuned models once on the untouched test set.
def tune_and_evaluate(features_train, features_test, labels_train, labels_test):
    logistic_regression = GridSearchCV(
        Pipeline([("preprocessor", make_preprocessor()),
                  ("model", LogisticRegression(max_iter=1000))]),
        {"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5, scoring="f1").fit(features_train, labels_train)

    random_forest = GridSearchCV(
        Pipeline([("preprocessor", make_preprocessor()),
                  ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),
        {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
        cv=5, scoring="f1").fit(features_train, labels_train)

    print("\n=== TUNING (cv=5, scoring=f1) ===")
    print("logistic regression best:", logistic_regression.best_params_,
          f"CV F1 {logistic_regression.best_score_:.3f}")
    print("random forest best:      ", random_forest.best_params_,
          f"CV F1 {random_forest.best_score_:.3f}")

    print("\n=== TEST SET (lapsed class) ===")
    for model_name, search in [("logistic regression", logistic_regression),
                               ("random forest", random_forest)]:
        predictions = search.predict(features_test)
        precision, recall, f1 = lapsed_class_scores(labels_test, predictions)
        print(f"\n{model_name}")
        print("confusion matrix [[TN FP] [FN TP]]:")
        print(confusion_matrix(labels_test, predictions))
        print(f"precision {precision:.3f} | recall {recall:.3f} | F1 {f1:.3f}")
    print("\nreference: LR C=1 F1 ~0.62; RF depth 8, leaf 10, F1 ~0.65; "
          "selected model F1 must be >= 0.55")
    return logistic_regression, random_forest


# Task 5: reads the scaler and encoder that were really fitted inside the
# trained pipeline (training rows only) and prints what they learned.
# Also confirms no feature column has a missing value.
def report_preprocessing(logistic_regression, features, features_train):
    preprocessor = logistic_regression.best_estimator_.named_steps["preprocessor"]
    scaler = preprocessor.named_transformers_["numeric"]
    encoder = preprocessor.named_transformers_["categorical"]
    position = NUMERIC_COLUMNS.index("frequency")

    print("\n=== PREPROCESSING (fitted on training rows only) ===")
    print(f"frequency mean learned by the fitted scaler: {scaler.mean_[position]:.3f}")
    print(f"frequency mean of the full cohort:           {features['frequency'].mean():.3f}")
    print(f"They differ because the scaler saw only the {len(features_train)} training rows, "
          "not the test rows.")
    for column, categories in zip(CATEGORY_COLUMNS, encoder.categories_):
        print(f"{column}: {len(categories)} categories -> {len(categories) - 1} columns (k-1)")

    assert not features[NUMERIC_COLUMNS + CATEGORY_COLUMNS].isna().any().any(), \
        "missing values found"
    print("no feature column contains a missing value: True")


# Picks the model with the higher cross-validated F1 (decided on training data
# only) and checks it reaches test F1 >= 0.55.
def select_model(logistic_regression, random_forest, features_test, labels_test):
    if random_forest.best_score_ >= logistic_regression.best_score_:
        selected_search, model_name = random_forest, "random forest"
    else:
        selected_search, model_name = logistic_regression, "logistic regression"
    precision, recall, f1 = lapsed_class_scores(labels_test, selected_search.predict(features_test))
    print(f"\n=== SELECTED MODEL: {model_name} ===")
    print(f"test precision {precision:.3f} | recall {recall:.3f} | F1 {f1:.3f}")
    assert f1 >= 0.55, "selected model must reach test F1 >= 0.55"
    return selected_search


# Task 7: adds random copies of lapsed TRAINING rows until both classes are the
# same size, refits the best logistic regression, and compares on the original
# test set.
def balance_experiment(logistic_regression, features_train, features_test, labels_train, labels_test):
    extra_needed = int((labels_train == 0).sum() - (labels_train == 1).sum())
    extra_rows = features_train[labels_train == 1].sample(
        n=extra_needed, replace=True, random_state=42)
    features_balanced = pd.concat([features_train, extra_rows])
    labels_balanced = pd.concat([labels_train, pd.Series(1, index=extra_rows.index)])

    print("\n=== CLASS-BALANCE EXPERIMENT ===")
    print("balanced training counts:", labels_balanced.value_counts().to_dict(),
          "(reference 570 each)")
    balanced_model = clone(logistic_regression.best_estimator_).fit(features_balanced, labels_balanced)

    p_before, r_before, f_before = lapsed_class_scores(
        labels_test, logistic_regression.predict(features_test))
    p_after, r_after, f_after = lapsed_class_scores(
        labels_test, balanced_model.predict(features_test))
    print(f"before: precision {p_before:.3f} | recall {r_before:.3f} | F1 {f_before:.3f}")
    print(f"after:  precision {p_after:.3f} | recall {r_after:.3f} | F1 {f_after:.3f}")
    print("reference: recall rises from about 0.64 to about 0.85, precision stays near 0.6")


# Task 8a: top 5 random forest features and the sign of three logistic coefficients.
def interpret(logistic_regression, random_forest):
    forest_pipeline = random_forest.best_estimator_
    feature_names = forest_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(forest_pipeline.named_steps["model"].feature_importances_,
                            index=feature_names).sort_values(ascending=False)
    print("\n=== TOP 5 RANDOM FOREST FEATURES ===")
    print(importances.head(5).round(3).to_string())

    logistic_pipeline = logistic_regression.best_estimator_
    coefficients = pd.Series(
        logistic_pipeline.named_steps["model"].coef_[0],
        index=logistic_pipeline.named_steps["preprocessor"].get_feature_names_out())
    print("\n=== LOGISTIC COEFFICIENTS (positive = more likely to lapse) ===")
    for name in ["numeric__recency", "numeric__frequency", "numeric__cancel_rate"]:
        print(f"{name}: {coefficients[name]:+.3f}")


# Task 8b-c: saves the model, reloads it and proves the predictions are identical,
# then writes the top Premium customers by lapse probability.
def save_and_build_retention_list(selected_search, cohort, features_test, labels_test):
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(selected_search.best_estimator_, "outputs/lapse_model.joblib")
    reloaded_model = joblib.load("outputs/lapse_model.joblib")
    identical = np.array_equal(reloaded_model.predict(features_test),
                               selected_search.predict(features_test))
    print("\n=== SAVE AND RELOAD ===")
    print("reloaded model reproduces test predictions exactly:", identical)
    assert identical

    segments = pd.read_csv("outputs/customer_segments.csv")[["customer_id", "is_premium"]]
    test_results = pd.DataFrame({
        "customer_id": cohort.loc[features_test.index, "customer_id"].values,
        "lapse_probability": reloaded_model.predict_proba(features_test)[:, 1].round(4),
        "actually_lapsed": labels_test.values,
    }).merge(segments, on="customer_id", how="left")

    retention_list = test_results[test_results["is_premium"] == True].sort_values(
        "lapse_probability", ascending=False).head(10)
    retention_list = retention_list[["customer_id", "lapse_probability", "actually_lapsed"]]
    retention_list.to_csv("outputs/retention_list.csv", index=False)

    print("\n=== RETENTION LIST (Premium customers in the test set) ===")
    print(retention_list.to_string(index=False))
    print("rows:", len(retention_list), "| truly lapsed:", int(retention_list["actually_lapsed"].sum()))
    print("Note: the Premium flag uses whole-window RFM (includes September), "
          "so this list is illustrative, not a leakage-free evaluation.")


# Start here.
if __name__ == "__main__":
    cohort = build_cohort()
    features = cohort[NUMERIC_COLUMNS + CATEGORY_COLUMNS]
    labels = cohort["lapsed"]
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42)
    compare_with_reference("train rows", len(features_train))
    compare_with_reference("test rows", len(features_test))

    baseline(labels_test)
    logistic_regression, random_forest = tune_and_evaluate(
        features_train, features_test, labels_train, labels_test)
    report_preprocessing(logistic_regression, features, features_train)
    selected_search = select_model(logistic_regression, random_forest, features_test, labels_test)
    balance_experiment(logistic_regression, features_train, features_test, labels_train, labels_test)
    interpret(logistic_regression, random_forest)
    save_and_build_retention_list(selected_search, cohort, features_test, labels_test)
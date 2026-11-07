import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CHOSEN_K = 4

ACTIONS = {
    "Premium Big Spenders": "VIP offers, early access to electronics, priority support",
    "Loyal Frequent Buyers": "Loyalty points and subscription offers to keep the habit",
    "Lapsing / At-Risk": "Win-back coupons and a reminder before they leave",
    "Occasional Buyers": "Small discounts and bundles to raise order frequency",
}


# Task 1: builds the RFM table. For each customer with a completed order:
# recency = days since last order, frequency = number of orders,
# monetary = total spend.
def build_rfm():
    orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    done = orders[orders["status"] == "completed"]
    rfm = done.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
    rfm = rfm[["recency", "frequency", "monetary"]].reset_index()

    print("=== RFM TABLE ===")
    print("customers:", len(rfm), "(expected 1175)")
    print(f"recency median {rfm['recency'].median():.0f} (19), min {rfm['recency'].min()} (0), "
          f"max {rfm['recency'].max()} (180)")
    print(f"frequency mean {rfm['frequency'].mean():.2f} (9.46), max {rfm['frequency'].max()} (33)")
    print(f"monetary mean {rfm['monetary'].mean():.2f} (2552.06), max {rfm['monetary'].max():.2f} (14766.55)")
    return rfm


# Task 2: tries K = 2..8 and prints inertia (how tight the clusters are)
# and silhouette (how well separated they are) for each K.
def kmeans_table(X):
    print("\n=== K-MEANS: INERTIA AND SILHOUETTE ===")
    print("K   inertia   silhouette")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_)
        print(f"{k}   {km.inertia_:8.1f}   {sil:.3f}")
    print("reference: K=2 1820.8/0.465, K=3 1268.2/0.445, K=4 743.1/0.526, K=5 617.4/0.481,")
    print("           K=6 503.2/0.482, K=7 408.5/0.416, K=8 366.5/0.412")


# Task 3: fits the final K-means, prints each segment's size and averages,
# gives each segment a name and marks the highest-spending one as Premium.
def build_segments(rfm, X):
    km = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=42).fit(X)
    rfm = rfm.copy()
    rfm["segment_id"] = km.labels_

    prof = rfm.groupby("segment_id").agg(
        size=("customer_id", "count"),
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
    ).round(2)
    print(f"\n=== SEGMENT PROFILE (K={CHOSEN_K}) ===")
    print(prof.to_string())
    print("reference K=4: sizes about 436/138/419/182; the 138 segment has monetary ~7130, "
          "frequency ~6.8; the 419 segment has frequency ~19.4, recency ~8.8")

    # Names are given by rule, because cluster numbers are only labels with no order.
    names = {}
    premium_id = prof["monetary"].idxmax()
    names[premium_id] = "Premium Big Spenders"
    rest = prof.drop(index=premium_id)
    loyal_id = rest["frequency"].idxmax()
    names[loyal_id] = "Loyal Frequent Buyers"
    rest = rest.drop(index=loyal_id)
    lapsing_id = rest["recency"].idxmax()
    names[lapsing_id] = "Lapsing / At-Risk"
    for sid in rest.drop(index=lapsing_id).index:
        names[sid] = "Occasional Buyers"

    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_id"] == premium_id

    print("\n=== SEGMENT NAMES AND ACTIONS ===")
    for sid in prof.index:
        print(f"segment {sid}: {names[sid]} -> {ACTIONS[names[sid]]}")

    os.makedirs("outputs", exist_ok=True)
    rfm[["customer_id", "recency", "frequency", "monetary",
         "segment_id", "segment_name", "is_premium"]].to_csv(
        "outputs/customer_segments.csv", index=False)
    print("\nsaved outputs/customer_segments.csv")


# Task 3c: PCA squeezes the 3 scaled columns into 2 directions and
# reports how much of the variation each direction keeps.
def pca_report(X):
    pca = PCA(n_components=2).fit(X)
    r = pca.explained_variance_ratio_
    print("\n=== PCA ===")
    print(f"explained variance: {r[0]:.3f} and {r[1]:.3f}, total {r.sum():.3f} "
          "(reference about 0.634, 0.218, total 0.852)")


# Start here.
if __name__ == "__main__":
    rfm = build_rfm()
    X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
    kmeans_table(X)
    build_segments(rfm, X)
    pca_report(X)
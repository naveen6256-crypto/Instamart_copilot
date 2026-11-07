import sqlite3

import numpy as np
import pandas as pd

CLEAN = "data/clean/"
DB = "data/instamart.db"


# Loads the clean tables and keeps only completed orders,
# with city and product category added to each order.
def load_completed():
    orders = pd.read_csv(CLEAN + "orders.csv")
    customers = pd.read_csv(CLEAN + "customers.csv")
    products = pd.read_csv(CLEAN + "products.csv")
    done = orders[orders["status"] == "completed"]
    done = done.merge(customers[["customer_id", "city"]], on="customer_id")
    done = done.merge(products[["product_id", "category"]], on="product_id")
    return done


# Task 7a: rebuilds Q2 as a pivot table and checks its grand total
# against the SQL total. Then prints sum, mean and count per city.
def pivot_and_city_stats(done):
    pivot = pd.pivot_table(done, values="order_amount", index="city",
                           columns="category", aggfunc="sum", margins=True)
    pandas_total = round(pivot.loc["All", "All"], 2)

    with sqlite3.connect(DB) as con:
        sql_total = con.execute(
            "SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status='completed'"
        ).fetchone()[0]

    print("\n=== PIVOT TABLE (revenue, INR) ===")
    print(pivot.round(2).to_string())
    print(f"pandas grand total: INR {pandas_total:,.2f}")
    print(f"SQL total:          INR {sql_total:,.2f}")
    print("totals equal:", abs(pandas_total - sql_total) < 0.005)

    city = done.groupby("city").agg(
        total=("order_amount", "sum"),
        average=("order_amount", "mean"),
        orders=("order_amount", "count"),
    ).round(2)
    print("\n=== PER CITY (completed orders) ===")
    print(city.to_string())


# Task 7b: describes completed order_amount (mean, median, skewness,
# quartiles, outlier fences and outlier counts).
def amount_statistics(done):
    a = done["order_amount"]
    q1, q3 = a.quantile(0.25), a.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    print("\n=== ORDER AMOUNT STATISTICS (completed) ===")
    print(f"mean {a.mean():.2f} | median {a.median():.2f} | skewness {a.skew():.2f}")
    print(f"Q1 {q1:.2f} | Q3 {q3:.2f} | IQR {iqr:.2f}")
    print(f"lower fence {low:.2f} | upper fence {high:.2f}")
    print("outliers below lower fence:", int((a < low).sum()))
    print("outliers above upper fence:", int((a > high).sum()))


# Task 7b: correlation matrix of four columns, then names the pair
# with the largest correlation (ignoring the sign).
def correlations(done):
    cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr = done[cols].corr()
    print("\n=== CORRELATION MATRIX (completed) ===")
    print(corr.round(3).to_string())

    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    best = pairs.abs().idxmax()
    print(f"largest |correlation|: {best[0]} and {best[1]} = {pairs[best]:.3f}")


# Start here.
if __name__ == "__main__":
    done = load_completed()
    pivot_and_city_stats(done)
    amount_statistics(done)
    correlations(done)
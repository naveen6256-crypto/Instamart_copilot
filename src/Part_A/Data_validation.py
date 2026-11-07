import sqlite3
import sys

import pandas as pd

# Numbers taken from the brief. Only these are hardcoded.
REFERENCE = {
    "order rows": 12309,
    "completed": 11110,
    "cancelled": 1199,
    "completed revenue": 2998668.80,
    "cities": {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
               "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "genders": {"female": 610, "male": 560, "unknown": 30},
}

failures = []


# Records a failed check. Passing checks print nothing.
def check(name, condition):
    if not condition:
        print("FAIL:", name)
        failures.append(name)


# Compares the clean files with the RAW files (no hardcoded numbers).
def dynamic_checks(raw, orders, cust):
    done = orders[orders["status"] == "completed"]
    canc = orders[orders["status"] == "cancelled"]

    check("one clean row per raw order_id", len(orders) == raw["order_id"].nunique())
    check("same order_ids as raw", set(orders["order_id"]) == set(raw["order_id"]))
    check("order_id is unique", orders["order_id"].is_unique)
    check("only completed or cancelled remain", len(done) + len(canc) == len(orders))
    check("no missing amounts", orders["order_amount"].notna().all())
    check("all amounts positive", (orders["order_amount"] > 0).all())
    check("no missing delivery for completed", done["delivery_minutes"].notna().all())
    check("no delivery time for cancelled", canc["delivery_minutes"].isna().all())
    check("no negative delivery", (orders["delivery_minutes"].dropna() >= 0).all())
    check("cities are trimmed and title case",
          (cust["city"] == cust["city"].str.strip().str.title()).all())
    check("six cities", cust["city"].nunique() == 6)
    check("gender is male, female or unknown",
          cust["gender"].isin(["male", "female", "unknown"]).all())
    check("no missing ages", cust["age"].notna().all())
    check("ages in 10-100", cust["age"].between(10, 100).all())


# Checks that the database holds the same data as the clean CSV files.
def database_checks(orders, cust):
    with sqlite3.connect("data/instamart.db") as con:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        check("db has the three tables", {"customers", "products", "orders"} <= tables)
        db_orders = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        db_cust = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        db_rev = con.execute(
            "SELECT SUM(order_amount) FROM orders WHERE status='completed'").fetchone()[0]
    csv_rev = orders.loc[orders["status"] == "completed", "order_amount"].sum()
    check("db orders = csv orders", db_orders == len(orders))
    check("db customers = csv customers", db_cust == len(cust))
    check("db revenue = csv revenue", abs(db_rev - csv_rev) < 0.005)


# Compares with the numbers given in the brief.
def reference_checks(orders, cust):
    done = orders[orders["status"] == "completed"]
    ref = REFERENCE
    check("reference: order rows", len(orders) == ref["order rows"])
    check("reference: completed", len(done) == ref["completed"])
    check("reference: cancelled",
          (orders["status"] == "cancelled").sum() == ref["cancelled"])
    check("reference: completed revenue",
          abs(done["order_amount"].sum() - ref["completed revenue"]) < 0.005)
    check("reference: city counts", cust["city"].value_counts().to_dict() == ref["cities"])
    check("reference: gender counts",
          cust["gender"].value_counts().to_dict() == ref["genders"])
    check("reference: ages 18-60", cust["age"].between(18, 60).all())


# Runs all checks. Prints ALL CHECKPOINTS PASSED only if nothing failed.
def validate():
    raw = pd.read_csv("data/raw/orders.csv")
    orders = pd.read_csv("data/clean/orders.csv")
    cust = pd.read_csv("data/clean/customers.csv")

    dynamic_checks(raw, orders, cust)
    database_checks(orders, cust)
    reference_checks(orders, cust)

    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    validate()
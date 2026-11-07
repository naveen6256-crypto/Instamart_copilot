import os
import sqlite3

import pandas as pd

RAW = "data/raw/"           # dirty input files
CLEAN = "data/clean/"       # cleaned CSV files go here
DB = "data/instamart.db"    # database file we create


# Prints shape, column types and missing values of one table.
def audit_table(name, df):
    print(f"\n=== {name} ===")
    print("shape:", df.shape)
    print(df.dtypes.to_string())
    miss = df.isna().sum()
    out = pd.DataFrame({"missing": miss, "pct": (miss / len(df) * 100).round(2)})
    print(out.to_string())


# Checks the raw data and prints the 12 audit numbers. Changes nothing.
def audit():
    orders = pd.read_csv(RAW + "orders.csv")
    customers = pd.read_csv(RAW + "customers.csv")
    products = pd.read_csv(RAW + "products.csv")
    tickets = pd.read_csv(RAW + "tickets.csv", keep_default_na=False)

    for name, df in [("orders", orders), ("customers", customers),
                     ("products", products), ("tickets", tickets)]:
        audit_table(name, df)

    amt = pd.to_numeric(orders["order_amount"], errors="coerce")
    deliv = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    age = pd.to_numeric(customers["age"], errors="coerce")

    figures = {
        "raw orders rows": len(orders),
        "exact duplicate rows": int(orders.duplicated().sum()),
        "pending rows": int((orders["status"] == "pending").sum()),
        "distinct order_id": int(orders["order_id"].nunique()),
        "non-numeric order_amount": int(amt.isna().sum()),
        "negative delivery_minutes": int((deliv < 0).sum()),
        "blank delivery_minutes": int(deliv.isna().sum()),
        "blank age": int(age.isna().sum()),
        "age outside 10-100": int(((age < 10) | (age > 100)).sum()),
        "distinct raw city spellings": int(customers["city"].nunique()),
        "distinct non-blank gender spellings": int(customers["gender"].dropna().nunique()),
        "blank gender": int(customers["gender"].isna().sum()),
    }
    print("\n=== AUDIT FIGURES ===")
    for k, v in figures.items():
        print(f"{k}: {v}")


# Fixes the customers table: city names, gender, age, signup date.
def clean_customers(df):
    df = df.copy()
    df["city"] = df["city"].str.strip().str.title()

    g = df["gender"].fillna("").str.strip().str.lower()
    df["gender"] = g.map({"male": "male", "m": "male",
                          "female": "female", "f": "female"}).fillna("unknown")

    age = pd.to_numeric(df["age"], errors="coerce")
    bad_age = int((age.notna() & ~age.between(10, 100)).sum())
    age = age.where(age.between(10, 100))
    missing_before_fill = int(age.isna().sum())
    df["age"] = age.fillna(age.groupby(df["city"]).transform("median"))

    df["signup_date"] = pd.to_datetime(df["signup_date"])

    print("\n=== CLEAN CUSTOMERS ===")
    print("ages set to missing (outside 10-100):", bad_age)
    print("ages filled with city median:", missing_before_fill)
    print(df["city"].value_counts().to_string())
    print(df["gender"].value_counts().to_string())
    print("missing ages:", int(df["age"].isna().sum()),
          "| min age:", df["age"].min(), "| max age:", df["age"].max())
    return df


# Fixes the orders table in 5 steps (dates, duplicates, amounts, delivery times).
def clean_orders(orders, products):
    df = orders.copy()
    print("\n=== CLEAN ORDERS ===")
    print("rows at start:", len(df))

    # Rule 1: text -> real dates
    df["order_ts"] = pd.to_datetime(df["order_ts"])

    # Rule 2: remove rows that are exact copies
    n = len(df)
    df = df.drop_duplicates()
    print("rule 2 - exact duplicates removed:", n - len(df))

    # Rule 3: if an order appears twice, keep the latest one
    n = len(df)
    df = df.sort_values("order_ts").drop_duplicates("order_id", keep="last")
    print("rule 3 - earlier records removed:", n - len(df))

    # Rule 4: bad amounts -> calculate from price x quantity x discount
    df = df.merge(products[["product_id", "category", "price_inr"]],
                  on="product_id", how="left")
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
    miss = df["order_amount"].isna()
    calc = (df["price_inr"] * df["quantity"] * (1 - df["discount_pct"] / 100)).round(2)
    df.loc[miss, "order_amount"] = calc[miss]
    print("rule 4 - amounts filled from price:", int(miss.sum()))

    # Rule 5: fix delivery times (cancelled orders stay blank)
    deliv = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    neg = int((deliv < 0).sum())
    df["delivery_minutes"] = deliv.where(deliv >= 0)
    completed = df["status"] == "completed"
    cat_median = (df["delivery_minutes"].where(completed)
                  .groupby(df["category"]).transform("median"))
    to_fill = completed & df["delivery_minutes"].isna()
    df.loc[to_fill, "delivery_minutes"] = cat_median[to_fill]
    print("rule 5 - negative delivery set to missing:", neg)
    print("rule 5 - completed delivery filled with category median:", int(to_fill.sum()))

    # remove helper columns
    df = df.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)

    print("final rows:", len(df))
    print(df["status"].value_counts().to_string())
    print("missing order_amount:", int(df["order_amount"].isna().sum()))
    print("missing delivery (completed):",
          int(df.loc[df["status"] == "completed", "delivery_minutes"].isna().sum()))
    print("missing delivery (cancelled):",
          int(df.loc[df["status"] == "cancelled", "delivery_minutes"].isna().sum()))
    rev = df.loc[df["status"] == "completed", "order_amount"].sum()
    print(f"completed revenue: INR {rev:,.2f}")
    return df


# Saves the clean tables as CSV files and into the SQLite database.
def store(customers, products, orders):
    os.makedirs("data/clean", exist_ok=True)
    cust_out = customers.copy()
    cust_out["signup_date"] = cust_out["signup_date"].dt.strftime("%Y-%m-%d")
    ord_out = orders.copy()
    ord_out["order_ts"] = ord_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    cust_out.to_csv(CLEAN + "customers.csv", index=False)
    products.to_csv(CLEAN + "products.csv", index=False)
    ord_out.to_csv(CLEAN + "orders.csv", index=False)

    with sqlite3.connect(DB) as con:
        cust_out.to_sql("customers", con, if_exists="replace", index=False)
        products.to_sql("products", con, if_exists="replace", index=False)
        ord_out.to_sql("orders", con, if_exists="replace", index=False)
    print("\nstored: data/clean/*.csv and", DB)


# Start here: audit -> clean customers -> clean orders -> save.
if __name__ == "__main__":
    audit()
    raw_customers = pd.read_csv(RAW + "customers.csv")
    raw_products = pd.read_csv(RAW + "products.csv")
    raw_orders = pd.read_csv(RAW + "orders.csv")

    customers = clean_customers(raw_customers)
    orders = clean_orders(raw_orders, raw_products)
    store(customers, raw_products, orders)
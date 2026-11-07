import re
import sqlite3

import pandas as pd

DB = "data/instamart.db"
SQL_FILE = "sql/Queries.sql"


# Reads sql/queries.sql and splits it into {"Q1": "...", "Q2": "...", ...}.
def load_queries():
    text = open(SQL_FILE, encoding="utf-8").read()
    parts = re.split(r"(?m)^-- (Q\d)\b.*$", text)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


# Runs the 7 queries on the database and prints the result of each,
# so you can compare with the numbers in the brief.
def run_all():
    queries = load_queries()
    with sqlite3.connect(DB) as con:
        res = {q: pd.read_sql_query(sql, con) for q, sql in queries.items()}

    print("Q1 customers:", len(res["Q1"]), "(expected 30)")

    q2 = res["Q2"]
    print("Q2 rows:", len(q2), "| total INR", f"{q2['revenue'].sum():,.2f}",
          "(expected 30 rows, 2,998,668.80)")
    top = q2.iloc[0]
    print("Q2 largest cell:", top["city"], "/", top["category"],
          f"INR {top['revenue']:,.2f}", "(expected Bengaluru / Electronics 233,909.45)")

    print("Q3 products:", len(res["Q3"]), "(expected 11)")
    print("Q4 customers with no completed order:", len(res["Q4"]), "(expected 25)")

    q5 = res["Q5"]
    print("Q5 customers above average:", len(q5),
          "| average INR", q5["avg_total"].iloc[0], "(expected 550, 2552.06)")

    q6 = res["Q6"]
    print("Q6 top customer per city:")
    print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
    print("Q6 rows where RANK differs from ROW_NUMBER:",
          int((q6["rnk"] != q6["row_num"]).sum()), "(expected 8)")

    q7 = res["Q7"]
    print("Q7 daily rows:", len(q7), "| last cumulative INR",
          f"{q7['cumulative_revenue'].iloc[-1]:,.2f}", "(expected 183, 2,998,668.80)")


# Start here.
if __name__ == "__main__":
    run_all()
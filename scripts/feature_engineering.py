"""
Stage 3 — Feature Engineering
Nassau Candy Distributor — Product Line Profitability Analysis

Adds profitability KPIs to the cleaned dataset:
- Gross Margin %
- Profit per Unit
- Revenue Contribution
- Profit Contribution
- Margin Volatility (per product, monthly)

Run from project root: python scripts/feature_engineering.py
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE = "data/processed/cleaned_data.csv"
OUTPUT_FILE = "data/processed/featured_data.csv"


def add_margin_metrics(df):
    df["Gross Margin %"] = (df["Gross Profit"] / df["Sales"]) * 100
    df["Profit per Unit"] = df["Gross Profit"] / df["Units"]
    logger.info("Added Gross Margin %% and Profit per Unit")
    return df


def add_contribution_metrics(df):
    total_sales = df["Sales"].sum()
    total_profit = df["Gross Profit"].sum()

    product_sales = df.groupby("Product Name")["Sales"].transform("sum")
    product_profit = df.groupby("Product Name")["Gross Profit"].transform("sum")

    df["Revenue Contribution %"] = (product_sales / total_sales) * 100
    df["Profit Contribution %"] = (product_profit / total_profit) * 100
    logger.info("Added Revenue Contribution %% and Profit Contribution %%")
    return df


def add_margin_volatility(df):
    df["Order Month"] = pd.to_datetime(df["Order Date"]).dt.to_period("M")

    monthly_margin = (
        df.groupby(["Product Name", "Order Month"])
        .apply(lambda x: (x["Gross Profit"].sum() / x["Sales"].sum()) * 100)
        .reset_index(name="Monthly Margin %")
    )

    volatility = (
        monthly_margin.groupby("Product Name")["Monthly Margin %"]
        .std()
        .reset_index(name="Margin Volatility")
    )

    df = df.merge(volatility, on="Product Name", how="left")
    df = df.drop(columns=["Order Month"])
    logger.info("Added Margin Volatility")
    return df


def main():
    df = pd.read_csv(INPUT_FILE)
    logger.info("Loaded %s -> shape %s", INPUT_FILE, df.shape)

    df = add_margin_metrics(df)
    df = add_contribution_metrics(df)
    df = add_margin_volatility(df)

    df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved featured dataset (%s) to %s", df.shape, OUTPUT_FILE)


if __name__ == "__main__":
    main()
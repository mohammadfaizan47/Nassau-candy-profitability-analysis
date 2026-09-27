"""
Reusable Data Cleaning Pipeline
--------------------------------
One file, works on any tabular dataset. To reuse for a new project, only
edit the CONFIG section at the top — the functions below never need to change.

Run: python clean_data.py
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG — only section you edit per dataset/project
# ============================================================
MAIN_FILE = "data/raw/Nassau Candy Distributor.csv"
REFERENCE_FILES = [
    {"path": "data/raw/Product_Factory_Mapping.csv", "on": "Product Name", "keep_cols": ["Product Name", "Factory"], "text_cols": ["Division", "Product Name", "Factory"]},
    {"path": "data/raw/Factory_Coordinates.csv", "on": "Factory", "keep_cols": None, "text_cols": ["Factory"]},
]
OUTPUT_FILE = "data/processed/cleaned_data.csv"

TEXT_COLS = ["Division", "Product Name", "Region", "Country/Region", "City", "State/Province", "Ship Mode"]
DATE_COLS = ["Order Date", "Ship Date"]
DATE_FORMAT = "%d-%m-%Y"          # set to None to auto-detect
NUMERIC_COLS = ["Sales", "Units", "Gross Profit", "Cost"]
POSITIVE_COLS = ["Sales", "Units"]        # must be > 0
NON_NEGATIVE_COLS = ["Cost"]              # must be >= 0
FILL_MEDIAN_COLS = ["Units"]              # missing values filled with median
DUPLICATE_KEY = ["Row ID"]                # unique-ID column(s) for de-duping
RELATIONSHIP_CHECK = {"result": "Gross Profit", "a": "Sales", "b": "Cost"}  # result == a - b


# ============================================================
# GENERIC FUNCTIONS — reusable across any dataset, don't edit
# ============================================================
def load_csv(path):
    df = pd.read_csv(path)
    logger.info("Loaded %s -> shape %s", path, df.shape)
    return df


def standardize_text(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def parse_dates(df, cols, fmt=None):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
    bad = df[cols].isnull().sum().sum() if cols else 0
    if bad:
        logger.warning("Unparseable dates across %s: %d", cols, bad)
    return df


def validate_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def drop_missing(df, cols):
    before = len(df)
    df = df.dropna(subset=cols)
    logger.info("Dropped %d rows with missing values in %s (%d -> %d)", before - len(df), cols, before, len(df))
    return df


def filter_positive(df, cols):
    before = len(df)
    for col in cols:
        df = df[df[col] > 0]
    logger.info("Filtered non-positive values in %s (%d -> %d)", cols, before, len(df))
    return df


def filter_non_negative(df, cols):
    before = len(df)
    for col in cols:
        df = df[df[col] >= 0]
    logger.info("Filtered negative values in %s (%d -> %d)", cols, before, len(df))
    return df


def fill_missing_with_median(df, cols):
    for col in cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Filled missing %s with median: %s", col, median_val)
    return df


def remove_duplicates(df, subset):
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    logger.info("Removed %d duplicate rows", before - len(df))
    return df


def flag_inconsistent_relationship(df, result_col, col_a, col_b, tolerance=0.01):
    expected = df[col_a] - df[col_b]
    mismatch = (df[result_col] - expected).abs() > tolerance
    if mismatch.any():
        logger.warning("Rows where %s != %s - %s: %d", result_col, col_a, col_b, mismatch.sum())
    return mismatch


def merge_reference_table(df, ref_df, on, keep_cols=None):
    """
    keep_cols: which columns from ref_df to bring in (besides the join key).
    If None, brings in all columns from ref_df not already in df.
    """
    if keep_cols is None:
        keep_cols = [c for c in ref_df.columns if c == on or c not in df.columns]
    ref_df = ref_df[keep_cols]

    new_cols = [c for c in keep_cols if c != on]
    merged = df.merge(ref_df, on=on, how="left")
    if new_cols:
        unmatched = merged[new_cols[0]].isnull().sum()
        if unmatched:
            logger.warning("%d rows unmatched after merging on '%s'", unmatched, on)
    return merged


def save_csv(df, path):
    df.to_csv(path, index=False)
    logger.info("Saved cleaned dataset (%s) to %s", df.shape, path)


# ============================================================
# PIPELINE — runs using the CONFIG above
# ============================================================
def main():
    df = load_csv(MAIN_FILE)
    df = standardize_text(df, TEXT_COLS)
    df = parse_dates(df, DATE_COLS, fmt=DATE_FORMAT)
    df = validate_numeric(df, NUMERIC_COLS)
    df = drop_missing(df, NUMERIC_COLS)
    df = filter_positive(df, POSITIVE_COLS)
    df = filter_non_negative(df, NON_NEGATIVE_COLS)
    df = fill_missing_with_median(df, FILL_MEDIAN_COLS)
    df = remove_duplicates(df, DUPLICATE_KEY)

    if RELATIONSHIP_CHECK:
        flag_inconsistent_relationship(
            df, RELATIONSHIP_CHECK["result"], RELATIONSHIP_CHECK["a"], RELATIONSHIP_CHECK["b"]
        )

    for ref in REFERENCE_FILES:
        ref_df = load_csv(ref["path"])
        ref_df = standardize_text(ref_df, ref["text_cols"])
        df = merge_reference_table(df, ref_df, on=ref["on"], keep_cols=ref.get("keep_cols"))

    save_csv(df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
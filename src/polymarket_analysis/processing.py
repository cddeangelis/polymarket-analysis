import pandas as pd
import numpy as np


def categorize_wallets(
    df: pd.DataFrame,
    small_threshold: float = 1000,
    large_threshold: float = 100000,
) -> pd.DataFrame:
    """Categorize wallets by their max cumulative wager on each outcome.

    For each wallet and each outcome (yes/no), track the cumulative net wager
    (buys minus sells) over time, recording the high-water mark. The wallet's
    size category is based on the maximum high-water mark across outcomes.
    """
    sorted_df = df.sort_values("timestamp")

    # Signed amount: positive for buys, negative for sells
    signed = sorted_df["amount_usd"].where(sorted_df["side"] == "buy", -sorted_df["amount_usd"])

    # Cumulative net wager per (wallet, outcome), then high-water mark
    cumulative = signed.groupby([sorted_df["wallet"], sorted_df["outcome"]]).cumsum()
    high_water = cumulative.groupby([sorted_df["wallet"], sorted_df["outcome"]]).cummax()

    # Max high-water mark per wallet (across outcomes and time)
    sorted_df = sorted_df.assign(high_water=high_water)
    wallet_df = (
        sorted_df.groupby("wallet")["high_water"]
        .max()
        .clip(lower=0)
        .reset_index()
        .rename(columns={"high_water": "max_wager"})
    )

    wallet_df["category"] = np.where(
        wallet_df["max_wager"] <= small_threshold, "small",
        np.where(wallet_df["max_wager"] <= large_threshold, "medium", "large"),
    )

    cat_counts = wallet_df["category"].value_counts()
    print(f"Wallet categories: {dict(cat_counts)}")

    return wallet_df


def aggregate_hourly(
    df: pd.DataFrame, wallet_df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate trades by hour, wallet category, outcome, and side."""
    merged = df.merge(wallet_df[["wallet", "category"]], on="wallet", how="left")
    merged["hour"] = merged["timestamp"].dt.floor("h")

    hourly = (
        merged.groupby(["hour", "category", "outcome", "side"])["amount_usd"]
        .sum()
        .reset_index()
    )

    return hourly

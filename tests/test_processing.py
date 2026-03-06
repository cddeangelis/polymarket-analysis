import pandas as pd
import numpy as np
from polymarket_analysis.processing import categorize_wallets, aggregate_hourly


def _make_trades(records):
    """Build a trades DataFrame from simplified records.

    Each record is (wallet, outcome, side, amount_usd, hours_offset).
    """
    rows = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for wallet, outcome, side, amount, hours in records:
        rows.append({
            "wallet": wallet,
            "outcome": outcome,
            "side": side,
            "amount_usd": amount,
            "timestamp": base + pd.Timedelta(hours=hours),
        })
    return pd.DataFrame(rows)


class TestCategorizeWallets:
    def test_small_wallet(self):
        df = _make_trades([("w1", "yes", "buy", 500, 0)])
        result = categorize_wallets(df, small_threshold=1000, large_threshold=100000)
        assert result.loc[result["wallet"] == "w1", "category"].item() == "small"

    def test_medium_wallet(self):
        df = _make_trades([("w1", "yes", "buy", 5000, 0)])
        result = categorize_wallets(df, small_threshold=1000, large_threshold=100000)
        assert result.loc[result["wallet"] == "w1", "category"].item() == "medium"

    def test_large_wallet(self):
        df = _make_trades([("w1", "yes", "buy", 200000, 0)])
        result = categorize_wallets(df, small_threshold=1000, large_threshold=100000)
        assert result.loc[result["wallet"] == "w1", "category"].item() == "large"

    def test_high_water_mark_after_sell(self):
        """Wallet buys 5000 then sells 4000 — high-water mark is 5000, not 1000."""
        df = _make_trades([
            ("w1", "yes", "buy", 5000, 0),
            ("w1", "yes", "sell", 4000, 1),
        ])
        result = categorize_wallets(df, small_threshold=1000, large_threshold=100000)
        row = result[result["wallet"] == "w1"].iloc[0]
        assert row["max_wager"] == 5000
        assert row["category"] == "medium"

    def test_max_across_outcomes(self):
        """Wallet bets small on Yes but large on No — categorized by the larger."""
        df = _make_trades([
            ("w1", "yes", "buy", 500, 0),
            ("w1", "no", "buy", 200000, 1),
        ])
        result = categorize_wallets(df, small_threshold=1000, large_threshold=100000)
        assert result.loc[result["wallet"] == "w1", "category"].item() == "large"

    def test_multiple_wallets(self):
        df = _make_trades([
            ("w1", "yes", "buy", 100, 0),
            ("w2", "yes", "buy", 5000, 0),
            ("w3", "no", "buy", 500000, 0),
        ])
        result = categorize_wallets(df, small_threshold=1000, large_threshold=100000)
        cats = result.set_index("wallet")["category"]
        assert cats["w1"] == "small"
        assert cats["w2"] == "medium"
        assert cats["w3"] == "large"

    def test_sell_only_wallet_gets_zero(self):
        """A wallet that only sells should have max_wager clipped to 0."""
        df = _make_trades([("w1", "yes", "sell", 1000, 0)])
        result = categorize_wallets(df, small_threshold=1000, large_threshold=100000)
        assert result.loc[result["wallet"] == "w1", "max_wager"].item() == 0


class TestAggregateHourly:
    def test_basic_aggregation(self):
        df = _make_trades([
            ("w1", "yes", "buy", 100, 0),
            ("w2", "yes", "buy", 200, 0),
            ("w1", "no", "buy", 50, 1),
        ])
        wallet_df = categorize_wallets(df)
        hourly = aggregate_hourly(df, wallet_df)

        assert "hour" in hourly.columns
        assert "category" in hourly.columns
        assert hourly["amount_usd"].sum() == 350

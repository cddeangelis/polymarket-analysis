import argparse
import os

from polymarket_analysis.data import get_market_info, fetch_order_fills
from polymarket_analysis.processing import categorize_wallets, aggregate_hourly
from polymarket_analysis.charts import plot_wallet_distribution, plot_hourly_volume


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and chart Polymarket betting data"
    )
    parser.add_argument("slug", help="Market slug from Polymarket URL")
    parser.add_argument(
        "--output-dir", default=".", help="Directory for output PNGs (default: .)"
    )
    parser.add_argument(
        "--small-threshold", type=float, default=1000,
        help="Max wager for 'small' wallet category (default: 1000)",
    )
    parser.add_argument(
        "--large-threshold", type=float, default=100000,
        help="Min wager for 'large' wallet category (default: 100000)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Fetching market info for '{args.slug}'...")
    market = get_market_info(args.slug)
    print(f"Market: {market['question']}")
    print(f"Yes token: {market['yes_token_id'][:20]}...")
    print(f"No token:  {market['no_token_id'][:20]}...")

    print("\nFetching trade data from subgraph...")
    trades_df = fetch_order_fills(market["yes_token_id"], market["no_token_id"])

    print("\nCategorizing wallets...")
    wallet_df = categorize_wallets(
        trades_df,
        small_threshold=args.small_threshold,
        large_threshold=args.large_threshold,
    )

    print("\nAggregating hourly data...")
    hourly_df = aggregate_hourly(trades_df, wallet_df)

    charts_dir = os.path.join(args.output_dir, "charts", args.slug)
    os.makedirs(charts_dir, exist_ok=True)
    chart1_path = os.path.join(charts_dir, "wallet-distribution.png")
    chart2_path = os.path.join(charts_dir, "hourly-volume.png")

    print("\nGenerating Chart 1 (wallet distribution)...")
    plot_wallet_distribution(
        wallet_df, market["question"], chart1_path,
        small_threshold=args.small_threshold,
        large_threshold=args.large_threshold,
    )

    print("\nGenerating Chart 2 (hourly volume)...")
    plot_hourly_volume(
        hourly_df, market["question"], chart2_path,
        small_threshold=args.small_threshold,
        large_threshold=args.large_threshold,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()

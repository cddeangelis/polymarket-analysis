import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde

COLORS = {"small": "#4A90D9", "medium": "#1a1a1a", "large": "#E8952C"}
CATEGORY_ORDER = ["small", "medium", "large"]
CATEGORY_LABELS = {"small": "Small", "medium": "Medium", "large": "Large wallets"}


def plot_wallet_distribution(
    wallet_df: pd.DataFrame,
    question: str,
    output_path: str,
    small_threshold: float = 1000,
    large_threshold: float = 100000,
):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Clamp zero/negative wagers to a small positive value for log scale
    wagers = wallet_df["max_wager"].clip(lower=1.0)
    log_wagers = np.log10(wagers)

    # Compute KDE on log-scale wagers for density-based jitter
    kde = gaussian_kde(log_wagers, bw_method=0.15)
    densities = kde(log_wagers)
    max_density = densities.max()

    # Plot each wallet as a dot with density-scaled y-jitter
    rng = np.random.default_rng(42)
    for cat in CATEGORY_ORDER:
        mask = wallet_df["category"] == cat
        cat_wagers = wagers[mask]
        cat_log = np.log10(cat_wagers)
        cat_densities = kde(cat_log)

        # y-jitter proportional to local density (creates teardrop shape)
        max_jitter = cat_densities / max_density * 0.45
        jitter = rng.uniform(-1, 1, size=len(cat_wagers)) * max_jitter

        ax.scatter(
            cat_wagers,
            jitter,
            c=COLORS[cat],
            s=3,
            alpha=0.6,
            label=CATEGORY_LABELS[cat],
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xscale("log")
    ax.set_xlim(50, 3e6)
    ax.set_ylim(-0.6, 0.8)

    # Remove y-axis
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # X-axis formatting
    ax.set_xticks([100, 1000, 10000, 100000, 1000000])
    ax.set_xticklabels(
        ["Less than 100", "1,000", "10,000", "100,000", "$1,000,000"],
        fontsize=8,
    )
    ax.tick_params(axis="x", length=0, pad=5)

    # Title and subtitle
    fig.text(
        0.05, 0.95,
        "Small Bettors Dominated Prediction Market",
        fontsize=13, fontweight="bold", ha="left", va="top",
        transform=fig.transFigure,
    )
    fig.text(
        0.05, 0.88,
        "Maximum amount each wallet wagered (high-water mark across outcomes)",
        fontsize=9, ha="left", va="top", color="#444",
        transform=fig.transFigure,
    )

    # Legend
    legend = ax.legend(
        loc="upper left", fontsize=7, frameon=False,
        markerscale=2, ncol=3, handletextpad=0.3, columnspacing=1,
    )
    for handle in legend.legend_handles:
        handle.set_sizes([20])

    # Annotations
    total_wallets = len(wallet_df)
    small_pct = (wallet_df["max_wager"] <= small_threshold).sum() / total_wallets * 100
    large_1m_count = (wallet_df["max_wager"] > 1_000_000).sum()

    ax.annotate(
        f"{small_pct:.0f}% of wallets wagered\nless than ${small_threshold:,.0f}",
        xy=(small_threshold, 0.35), fontsize=7.5,
        ha="center", va="bottom",
    )

    if large_1m_count > 0:
        ax.annotate(
            f"{large_1m_count} wallets placed\nbets larger than $1M",
            xy=(1_200_000, 0.30), fontsize=7.5,
            ha="left", va="bottom",
        )

    # Source note
    fig.text(
        0.05, 0.02,
        f"Source: Polymarket (Gamma API / Goldsky subgraph)\n"
        f'Note: Data for the "{question}" market on Polymarket. For each wallet, '
        f"the cumulative amount spent buying is offset by the amount sold; "
        f"the highest level reached over time is the maximum wager.",
        fontsize=5.5, color="#666", ha="left", va="bottom",
        transform=fig.transFigure,
    )

    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.84])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wallet distribution chart saved to {output_path}")


def plot_hourly_volume(
    hourly_df: pd.DataFrame,
    question: str,
    output_path: str,
    small_threshold: int = 1_000,
    large_threshold: int = 100_000,
):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get the full time range and create hourly bins
    all_hours = pd.date_range(
        hourly_df["hour"].min(), hourly_df["hour"].max(), freq="h"
    )
    bar_width = pd.Timedelta(hours=1)

    # For each category, plot yes (positive) and no (negative) bars
    # Stack: small on bottom, medium on top, large on top
    yes_bottoms = np.zeros(len(all_hours))
    no_bottoms = np.zeros(len(all_hours))

    for cat in CATEGORY_ORDER:
        cat_data = hourly_df[hourly_df["category"] == cat]

        # Yes buys (positive)
        yes_buys = cat_data[
            (cat_data["outcome"] == "yes") & (cat_data["side"] == "buy")
        ]
        yes_vals = (
            yes_buys.set_index("hour")["amount_usd"]
            .reindex(all_hours, fill_value=0)
            .values
        )

        # No buys (negative, mirrored below zero)
        no_buys = cat_data[
            (cat_data["outcome"] == "no") & (cat_data["side"] == "buy")
        ]
        no_vals = (
            no_buys.set_index("hour")["amount_usd"]
            .reindex(all_hours, fill_value=0)
            .values
        )

        ax.bar(
            all_hours, yes_vals, width=bar_width, bottom=yes_bottoms,
            color=COLORS[cat], label=CATEGORY_LABELS[cat],
            edgecolor="none", linewidth=0,
        )
        ax.bar(
            all_hours, -no_vals, width=bar_width, bottom=-no_bottoms,
            color=COLORS[cat],
            edgecolor="none", linewidth=0,
        )

        yes_bottoms += yes_vals
        no_bottoms += no_vals

    # Zero line
    ax.axhline(0, color="black", linewidth=0.5)

    # Y-axis formatting (dollar amounts)
    def dollar_formatter(x, pos):
        if x == 0:
            return "0"
        if abs(x) >= 1e6:
            return f"${abs(x)/1e6:.0f}M"
        if abs(x) >= 1e3:
            return f"${abs(x)/1e3:.0f}K"
        return f"${abs(x):.0f}"

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(dollar_formatter))
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    # X-axis date formatting
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b. %d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=9)

    # Spine styling
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Labels for "Yes" and "No" bets
    ax.text(
        all_hours[0] - pd.Timedelta(hours=3), ax.get_ylim()[1] * 0.01,
        '"Yes" bet \u25B2', fontsize=8, fontweight="bold", ha="left", va="bottom",
    )
    ax.text(
        all_hours[0] - pd.Timedelta(hours=3), -ax.get_ylim()[1] * 0.01,
        '"No" bet \u25BC', fontsize=8, fontweight="bold", ha="left", va="top",
    )

    # Title and subtitle
    fig.text(
        0.05, 0.97,
        "Larger Bettors Were Late to Wager",
        fontsize=14, fontweight="bold", ha="left", va="top",
        transform=fig.transFigure,
    )
    fig.text(
        0.05, 0.935,
        'Total amount spent on "Yes" and "No" bets per hour, by wallet size',
        fontsize=10, ha="left", va="top", color="#444",
        transform=fig.transFigure,
    )

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[cat])
        for cat in CATEGORY_ORDER
    ]
    labels = [CATEGORY_LABELS[cat] for cat in CATEGORY_ORDER]
    legend = ax.legend(
        handles, labels, loc="upper left", fontsize=8, frameon=False,
        ncol=3, handletextpad=0.4, columnspacing=1,
    )

    # Source note
    fig.text(
        0.05, 0.02,
        f"Source: Polymarket (Gamma API / Goldsky subgraph)\n"
        f'Note: Data for the "{question}" market on Polymarket. '
        f"All times are for the UTC timezone. "
        f"Wallet size is based on the maximum cumulative wager (high-water mark) "
        f"across outcomes: small (${small_threshold:,} or less); "
        f"medium (${small_threshold:,} to ${large_threshold:,}); large (more than ${large_threshold:,}).",
        fontsize=6.5, color="#666", ha="left", va="bottom",
        transform=fig.transFigure,
    )

    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.92])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Hourly volume chart saved to {output_path}")

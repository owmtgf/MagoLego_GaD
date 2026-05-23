"""Empirical game-theoretic analysis of eBay auction timing strategies.

The script builds a two-player normal-form game from the top two bidders in
each auction. Each bidder's strategy is classified by the timing of their final
bid: Early (< 80% of auction duration) or Late (>= 80%, bid sniping).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "auction.csv"
OUTPUT_DIR = ROOT / "output"
LATE_THRESHOLD = 0.80
STRATEGIES = ["Early", "Late"]


def auction_days(label: str) -> int:
    """Extract duration in days from labels such as '7 day auction'."""
    match = re.search(r"(\d+)", str(label))
    if not match:
        raise ValueError(f"Cannot parse auction duration from {label!r}")
    return int(match.group(1))


def classify_strategy(normalized_time: float, threshold: float = LATE_THRESHOLD) -> str:
    return "Late" if normalized_time >= threshold else "Early"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    expected = {
        "auctionid",
        "bid",
        "bidtime",
        "bidder",
        "bidderrate",
        "openbid",
        "price",
        "item",
        "auction_type",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["duration_days"] = df["auction_type"].map(auction_days)
    df["normalized_bidtime"] = df["bidtime"] / df["duration_days"]
    df["bid_strategy"] = df["normalized_bidtime"].map(classify_strategy)
    return df


def bidder_final_bids(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each bidder's maximum bid in each auction as their final serious bid."""
    final = (
        df.sort_values(["auctionid", "bidder", "bid", "bidtime"])
        .groupby(["auctionid", "bidder"], as_index=False)
        .agg(
            final_bid=("bid", "max"),
            final_bidtime=("bidtime", "max"),
            bidderrate=("bidderrate", "max"),
            openbid=("openbid", "last"),
            price=("price", "last"),
            item=("item", "last"),
            auction_type=("auction_type", "last"),
            duration_days=("duration_days", "last"),
        )
    )
    final["normalized_final_time"] = final["final_bidtime"] / final["duration_days"]
    final["strategy"] = final["normalized_final_time"].map(classify_strategy)
    return final


def build_finalist_games(df: pd.DataFrame, final: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create one two-player observation from the winner and runner-up per auction."""
    market_values = df.drop_duplicates("auctionid").groupby("item")["price"].median()
    rows: list[dict[str, object]] = []

    for auctionid, group in final.groupby("auctionid"):
        if len(group) < 2:
            continue

        ranked = group.sort_values(["final_bid", "final_bidtime"], ascending=[False, False])
        winner = ranked.iloc[0]
        runner_up = ranked.iloc[1]
        estimated_value = float(market_values.loc[winner["item"]])
        winner_payoff = max(estimated_value - float(winner["price"]), 0.0)

        rows.append(
            {
                "auctionid": auctionid,
                "item": winner["item"],
                "auction_type": winner["auction_type"],
                "price": float(winner["price"]),
                "estimated_market_value": estimated_value,
                "winner": winner["bidder"],
                "runner_up": runner_up["bidder"],
                "winner_bid": float(winner["final_bid"]),
                "runner_up_bid": float(runner_up["final_bid"]),
                "winner_strategy": winner["strategy"],
                "runner_up_strategy": runner_up["strategy"],
                "winner_payoff": winner_payoff,
                "runner_up_payoff": 0.0,
            }
        )

    return pd.DataFrame(rows), market_values


def payoff_matrix(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate a symmetric 2x2 payoff matrix from observed finalist pairs."""
    observations: list[dict[str, object]] = []

    for row in games.itertuples(index=False):
        observations.append(
            {
                "player_strategy": row.winner_strategy,
                "opponent_strategy": row.runner_up_strategy,
                "payoff": row.winner_payoff,
            }
        )
        observations.append(
            {
                "player_strategy": row.runner_up_strategy,
                "opponent_strategy": row.winner_strategy,
                "payoff": row.runner_up_payoff,
            }
        )

    obs = pd.DataFrame(observations)
    matrix = (
        obs.pivot_table(
            index="player_strategy",
            columns="opponent_strategy",
            values="payoff",
            aggfunc="mean",
        )
        .reindex(index=STRATEGIES, columns=STRATEGIES)
        .fillna(0.0)
    )
    return matrix, obs


def pure_nash_equilibria(matrix: pd.DataFrame) -> list[tuple[str, str]]:
    equilibria: list[tuple[str, str]] = []
    for row_strategy in STRATEGIES:
        for col_strategy in STRATEGIES:
            row_payoff = matrix.loc[row_strategy, col_strategy]
            col_payoff = matrix.loc[col_strategy, row_strategy]

            row_best = row_payoff >= matrix.loc[:, col_strategy].max() - 1e-9
            col_best = col_payoff >= matrix.loc[:, row_strategy].max() - 1e-9
            if row_best and col_best:
                equilibria.append((row_strategy, col_strategy))
    return equilibria


def mixed_equilibrium_probability_late(matrix: pd.DataFrame) -> float | None:
    """Return opponent probability of Late that makes a player indifferent.

    For payoffs:
        a = U(Early, Early), b = U(Early, Late)
        c = U(Late, Early),  d = U(Late, Late)
    indifference requires:
        (1-q)a + q b = (1-q)c + q d
    where q is Pr(opponent chooses Late).
    """
    a = matrix.loc["Early", "Early"]
    b = matrix.loc["Early", "Late"]
    c = matrix.loc["Late", "Early"]
    d = matrix.loc["Late", "Late"]
    denominator = a - b - c + d
    if abs(denominator) < 1e-12:
        return None
    q = (a - c) / denominator
    if 0 <= q <= 1:
        return float(q)
    return None


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    final = bidder_final_bids(df)
    games, market_values = build_finalist_games(df, final)
    matrix, payoff_observations = payoff_matrix(games)

    strategy_counts = final["strategy"].value_counts().reindex(STRATEGIES, fill_value=0)
    finalist_pairs = (
        games.groupby(["winner_strategy", "runner_up_strategy"])
        .size()
        .rename("count")
        .reset_index()
    )

    summary = {
        "dataset": {
            "bid_rows": int(len(df)),
            "auctions": int(df["auctionid"].nunique()),
            "unique_bidders": int(df["bidder"].nunique()),
            "final_bidder_auction_observations": int(len(final)),
            "two_finalist_games": int(len(games)),
        },
        "items": df["item"].value_counts().to_dict(),
        "auction_types": df["auction_type"].value_counts().to_dict(),
        "market_values_median_price": market_values.round(4).to_dict(),
        "strategy_counts": strategy_counts.astype(int).to_dict(),
        "strategy_shares": (strategy_counts / strategy_counts.sum()).round(4).to_dict(),
        "pure_nash_equilibria": pure_nash_equilibria(matrix),
        "mixed_equilibrium_probability_late": mixed_equilibrium_probability_late(matrix),
    }

    df.describe(include="all").to_csv(OUTPUT_DIR / "dataset_descriptive_statistics.csv")
    final.to_csv(OUTPUT_DIR / "bidder_final_bids.csv", index=False)
    games.to_csv(OUTPUT_DIR / "auction_finalist_games.csv", index=False)
    payoff_observations.to_csv(OUTPUT_DIR / "payoff_observations.csv", index=False)
    matrix.round(4).to_csv(OUTPUT_DIR / "payoff_matrix.csv")
    finalist_pairs.to_csv(OUTPUT_DIR / "finalist_strategy_pairs.csv", index=False)

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Auction game analysis complete.")
    print(f"Rows: {summary['dataset']['bid_rows']}")
    print(f"Auctions: {summary['dataset']['auctions']}")
    print(f"Two-finalist games: {summary['dataset']['two_finalist_games']}")
    print("\nPayoff matrix, U(row strategy, column strategy):")
    print(matrix.round(4))
    print(f"\nPure Nash equilibria: {summary['pure_nash_equilibria']}")
    mixed = summary["mixed_equilibrium_probability_late"]
    if mixed is None:
        print("Mixed equilibrium: no interior mixed equilibrium.")
    else:
        print(f"Mixed equilibrium Pr(Late): {mixed:.4f}")


if __name__ == "__main__":
    main()

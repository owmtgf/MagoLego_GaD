# Game-Theoretic Analysis of Online Auction Bid Timing

## Group Members

Add full names here:

- Ekstrin Mihail
- Averyanova Maria
- Kalyagin Dmitry

## Task Interpretation

This project studies strategic behavior in online auctions. The empirical question is whether bidders benefit from placing their final serious bid early or late. In online auctions, late bidding, often called bid sniping, is strategic because it gives other bidders less time to react. The game-theoretic tension is direct: each bidder's payoff depends not only on their own bid timing, but also on the timing chosen by competitors.

The selected dataset is suitable for the assignment because it contains real auction-level bid histories, bidder identifiers, bid amounts, bid timing, bidder ratings, final auction prices, item categories, opening bids, and auction duration types.

Dataset source: [Online Auctions Dataset on Kaggle](https://www.kaggle.com/datasets/onlineauctions/online-auctions-dataset)

Local dataset file: `data/auction.csv`

## Data Description

The dataset contains bid-level records from eBay-style online auctions. The local file has:

| Quantity | Value |
|---|---:|
| Bid rows | 10,681 |
| Auctions | 628 |
| Unique bidders | 3,387 |
| Bidder-auction final bid observations | 5,173 |
| Auctions with at least two finalist bidders used in game | 604 |

Items in the dataset:

| Item | Bid rows |
|---|---:|
| Palm Pilot M515 PDA | 5,917 |
| Xbox game console | 2,811 |
| Cartier wristwatch | 1,953 |

Auction duration types:

| Auction type | Bid rows |
|---|---:|
| 7 day auction | 7,041 |
| 3 day auction | 2,023 |
| 5 day auction | 1,617 |

Relevant columns:

| Column | Meaning |
|---|---|
| `auctionid` | Auction identifier |
| `bid` | Bid amount |
| `bidtime` | Time of bid, measured in days since auction start |
| `bidder` | Bidder identifier |
| `bidderrate` | Bidder reputation score |
| `openbid` | Opening bid |
| `price` | Final auction price |
| `item` | Auctioned item category |
| `auction_type` | Auction duration, such as 3, 5, or 7 days |

## Game-Theoretic Model

The situation is modeled as a simultaneous/static two-player game. Each auction can have many bidders, but the empirical game is reduced to the two most relevant players in each auction: the winner and the runner-up. This is a standard simplification because the final price and allocation are mainly determined by the top two competing bidders.

### Players

- Player 1: a representative finalist bidder.
- Player 2: the opponent finalist bidder in the same auction.

The players are symmetric: either bidder can choose either timing strategy.

### Strategies

Each bidder's strategy is based on the timing of their final serious bid in the auction.

Let:

```text
normalized_final_time = final_bidtime / auction_duration_days
```

Strategies:

| Strategy | Definition |
|---|---|
| Early | Final bid is placed before 80% of the auction duration has passed |
| Late | Final bid is placed at or after 80% of the auction duration has passed |

Using all bidder-auction final bid observations:

| Strategy | Count | Share |
|---|---:|---:|
| Early | 2,380 | 46.01% |
| Late | 2,793 | 53.99% |

### Outcomes

For each auction with at least two bidders:

1. For each bidder, keep their maximum bid in that auction as their final serious bid.
2. Rank bidders by final bid.
3. Treat the highest final bidder as the winner and the second-highest final bidder as the runner-up.
4. Record both players' strategies, the final price, and the winner's payoff.

Observed finalist strategy pairs:

| Winner strategy | Runner-up strategy | Auctions |
|---|---:|---:|
| Early | Early | 16 |
| Early | Late | 17 |
| Late | Early | 50 |
| Late | Late | 521 |

Late bidding is empirically very common among finalists: 571 of 604 observed winners are late bidders, and 538 of 604 runner-up bidders are late bidders.

## Payoff Estimation

The true private value of each item for each bidder is not observed. Therefore, the project uses an empirical market-value proxy. For each item category, the estimated value is the median final price of auctions for that item:

| Item | Estimated market value |
|---|---:|
| Cartier wristwatch | 510.00 |
| Palm Pilot M515 PDA | 231.50 |
| Xbox game console | 123.15 |

The payoff is interpreted as estimated bargain surplus:

```text
winner_payoff = max(estimated_market_value_of_item - final_price, 0)
runner_up_payoff = 0
```

This payoff captures how much value the winner gains relative to a typical market price for the same item. Losing bidders receive zero because they do not obtain the item and do not pay the final price.

To construct a symmetric normal-form game, each finalist pair contributes two ordered observations:

```text
U(winner_strategy, runner_up_strategy) += winner_payoff
U(runner_up_strategy, winner_strategy) += 0
```

Then the expected payoff for each strategy pair is the average payoff across all ordered observations with that pair of strategies.

## Payoff Matrix

The resulting payoff matrix is:

| Player strategy / Opponent strategy | Early | Late |
|---|---:|---:|
| Early | 41.5472 | 12.7396 |
| Late | 43.2943 | 12.8823 |

Interpretation:

- If the opponent bids early, late bidding gives a slightly higher expected payoff: 43.2943 instead of 41.5472.
- If the opponent bids late, late bidding also gives a slightly higher expected payoff: 12.8823 instead of 12.7396.
- Payoffs are much lower when the opponent bids late, which suggests that late bidding increases competition intensity and reduces surplus.

## Solving the Game

### Pure-Strategy Nash Equilibrium

A strategy profile is a Nash equilibrium if no player can improve their payoff by changing strategy unilaterally.

Best responses:

| Opponent strategy | Payoff from Early | Payoff from Late | Best response |
|---|---:|---:|---|
| Early | 41.5472 | 43.2943 | Late |
| Late | 12.7396 | 12.8823 | Late |

Late is a strict best response against both Early and Late. Therefore, Late is a dominant strategy in this empirical payoff matrix.

The pure-strategy Nash equilibrium is:

```text
(Late, Late)
```

### Mixed-Strategy Equilibrium

For a two-strategy symmetric game, an interior mixed equilibrium exists only if each player can be made indifferent between Early and Late.

Let:

```text
a = U(Early, Early) = 41.5472
b = U(Early, Late)  = 12.7396
c = U(Late, Early)  = 43.2943
d = U(Late, Late)   = 12.8823
```

If the opponent chooses Late with probability `q`, indifference requires:

```text
(1 - q)a + qb = (1 - q)c + qd
```

Solving gives a value outside the valid probability interval `[0, 1]`. Therefore, there is no interior mixed-strategy equilibrium. The relevant equilibrium remains the pure equilibrium `(Late, Late)`.

## Comparison With Empirical Behavior

The theoretical prediction is consistent with the observed behavior:

- The game predicts late bidding by both players.
- In the data, late bidding is the majority behavior among all final bidder-auction observations: 53.99%.
- Among finalist pairs, the strongest pattern is even clearer: 521 of 604 auctions have both winner and runner-up using Late.
- The winner is a late bidder in 571 of 604 auctions, or about 94.54%.

The empirical behavior is therefore close to the Nash prediction, especially for bidders who become winners. Runner-up behavior is also mostly late, which supports the idea that serious auction participants strategically wait until the end.

## Interpretation

The result suggests that late bidding is strategically attractive in online auctions. Even though late bidding does not guarantee a high surplus, it is a best response because it performs slightly better than early bidding against both possible opponent strategies.

The equilibrium `(Late, Late)` also has an important economic interpretation. When both bidders wait until the end, the expected payoff is low compared with cases where the opponent bids early. This resembles a strategic arms race: late bidding is individually rational, but when many bidders use it, competition near the deadline becomes intense and surplus falls.

In practical terms:

- Bidders who want to maximize their chance of winning should consider late bidding.
- Sellers may benefit from late-bidding competition because it can keep serious bidders engaged until the deadline.
- Platform rules that extend auctions after last-second bids could change the game by reducing the advantage of sniping.

## Assumptions and Limitations

The main assumptions are:

- The game is reduced to two players: winner and runner-up.
- The timing strategy is binary: Early versus Late.
- The 80% threshold is used to define late bidding.
- The highest final bid is treated as the winning bid.
- Bidder private valuations are not observed, so item median final price is used as an empirical market-value proxy.
- Losing bidders receive payoff zero.
- The game is modeled as simultaneous/static, even though real auctions unfold dynamically over time.

Limitations:

- Real auctions may involve more than two strategically relevant bidders.
- Bid amounts and bid timing are chosen jointly, but this model focuses on timing.
- The median item price is only an approximation of bidder valuation.
- Different item categories may have different bidder populations and strategic patterns.
- The dataset includes only three item categories, so conclusions should be interpreted within this sample.

## Reproducible Implementation

The implementation is in:

```text
scripts/analyze_auction_game.py
```

Run it with the `mago` conda environment in WSL:

```bash
cd /home/owmtgf/vscode/vscode_projects/MagoLego_GaD/HW2
/home/owmtgf/anaconda3/envs/mago/bin/python scripts/analyze_auction_game.py
```

Generated outputs:

| File | Description |
|---|---|
| `output/summary.json` | Main dataset statistics, strategy shares, equilibrium result |
| `output/payoff_matrix.csv` | Estimated 2x2 payoff matrix |
| `output/finalist_strategy_pairs.csv` | Observed winner/runner-up strategy pair counts |
| `output/auction_finalist_games.csv` | Auction-level two-player game observations |
| `output/bidder_final_bids.csv` | Final serious bid per bidder per auction |
| `output/payoff_observations.csv` | Ordered payoff observations used to estimate the matrix |
| `output/dataset_descriptive_statistics.csv` | Descriptive statistics for the raw dataset |

## Conclusion

This project models online auction bid timing as a static two-player game between finalist bidders. Using real bid-level auction data, the estimated payoff matrix shows that Late bidding is a dominant strategy. The resulting Nash equilibrium is `(Late, Late)`, and the empirical behavior strongly supports this prediction: most finalist pairs, and especially most winners, bid late.

The analysis demonstrates how real auction data can be transformed into a game-theoretic model with empirically estimated strategies, payoffs, and equilibrium predictions.

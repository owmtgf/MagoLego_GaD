import pandas as pd
import numpy as np

from pandas import Series


def replace_infinite(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype != "object":
            finite_vals = df.loc[np.isfinite(df[col]), col]
            if len(finite_vals) > 0:
                max_finite = finite_vals.max()
                df.loc[~np.isfinite(df[col]), col] = max_finite
    return df


def min_max_norm(df: pd.DataFrame):
    criteria_cols = df.columns.drop("model")

    for col in criteria_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0.0

    return df


def get_pareto_set(df:pd.DataFrame, model_names: Series):
    X = df.iloc[:, 1:].values     # criteria matrix
    n = X.shape[0]

    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        for k in range(n):
            if i != k:
                # Check if k dominates i
                if np.all(X[k] >= X[i]) and np.any(X[k] > X[i]):
                    pareto_mask[i] = False
                    break

    return df[pareto_mask]


def wlc_ranking(pareto_set: pd.DataFrame, weights: dict):
    pareto_weighted_ranking = pd.DataFrame(pareto_set["model"].copy())
    pareto_weighted_ranking["WLC_score"] = sum(pareto_set[col] * weight for col, weight in weights.items())

    return pareto_weighted_ranking.sort_values("WLC_score", ascending=False)


def wd_ranking(pareto_set: pd.DataFrame, weights: dict):
    # convert weights to array aligned with df columns
    pareto_set_criteria = pareto_set.loc[:, ~pareto_set.columns.isin(['model'])]
    w = np.array([weights[col] for col in pareto_set_criteria.columns])

    # matrix of metric values
    X = pareto_set[list(weights.keys())].values

    # ideal vector
    ideal = np.ones(X.shape[1])

    # compute weighted squared distance
    distances = np.sqrt(np.sum(w * (ideal - X)**2, axis=1))

    pareto_weighted_distance = pd.DataFrame(pareto_set["model"].copy())
    pareto_weighted_distance["distance_to_ideal"] = distances

    return pareto_weighted_distance.sort_values("distance_to_ideal", ascending=True)


def hare_stv(ranks):
    remaining = ranks.index.tolist()  # this is already model names
    round_num = 1
    
    while len(remaining) > 1:
        # count first-choice votes
        first_choices = ranks.loc[remaining].apply(lambda row: row.idxmin(), axis=0)
        vote_counts = first_choices.value_counts()
        
        # find model with fewest first-choice votes
        min_votes = vote_counts.min()
        to_eliminate = vote_counts[vote_counts == min_votes].index.tolist()
        
        # break ties arbitrarily if multiple
        eliminated = to_eliminate[0]
        print(f"Round {round_num}: eliminating {eliminated} with {min_votes} votes")
        
        # remove eliminated model
        remaining.remove(eliminated)
        round_num += 1
    
    winner = remaining[0]
    return winner


def get_tournament_matrix(pareto_set: pd.DataFrame, models: list):
    pareto_set_criteria = pareto_set.loc[:, ~pareto_set.columns.isin(['model'])]

    # initialize pairwise tournament matrix
    tournament_matrix = pd.DataFrame(0, index=models, columns=models)

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                continue
            # use .iloc instead of .loc
            m1_better = (pareto_set_criteria.iloc[i] > pareto_set_criteria.iloc[j]).sum()
            m2_better = (pareto_set_criteria.iloc[j] > pareto_set_criteria.iloc[i]).sum()
            if m1_better > m2_better:
                tournament_matrix.loc[m1, m2] = 1  # m1 beats m2
            elif m2_better > m1_better:
                tournament_matrix.loc[m1, m2] = -1  # m1 loses to m2
            else:
                tournament_matrix.loc[m1, m2] = 0  # tie

    return tournament_matrix
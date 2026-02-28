import re
import numpy as np
import pandas as pd

MISSING_TOKENS = {"?", "missing"}

def print_unique_values(df: pd.DataFrame, max_values: int=20):

    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)
        
        print(f"\nColumn: {col}")
        print(f"Number of unique values: {n_unique}")
        
        if n_unique <= max_values:
            print("Unique values:", unique_vals)
        else:
            print(f"First {max_values} unique values:", unique_vals[:max_values])
            print("... (truncated)")


def drop_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    
    missing_cols = [col for col in feature_names if col not in df.columns]
    assert len(missing_cols) == 0, (
        f"The following columns do not exist in DataFrame: {missing_cols}"
    )
    
    return df.copy().drop(columns=feature_names)


def map_features(df: pd.DataFrame, feature_mapping: dict) -> pd.DataFrame: 

    df_out = df.copy()
    
    for feature, mapping in feature_mapping.items():
        if not (feature in df_out.columns): 
            print(f"Feature '{feature}' not found in DataFrame. Skipping...")
            continue
        assert isinstance(mapping, dict), (
            f"Mapping for feature '{feature}' must be a dict"
        )
        
        existing_values = set(df_out[feature].unique())
        mapping_keys = set(mapping.keys())
        
        invalid_keys = mapping_keys - existing_values
        if len(invalid_keys) != 0: 
            print(
                f"Mapping for feature '{feature}' contains labels "
                f"not present in data: {invalid_keys}"
            )
        
        df_out[feature] = df_out[feature].map(
            lambda x: mapping.get(x, x)
        )
    
    return df_out


def replace_missing_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(list(MISSING_TOKENS), pd.NA)

# secret_scrimmage_lineup_guesser/data_processing.py

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and process player data for lineup optimization.

    Args:
        df (pd.DataFrame): DataFrame containing player stats with columns:
            - name (str): Player name
            - min (int): Minutes played
            - sec (int): Seconds played
            - plus_minus (int): Player's plus/minus for the game
            - allowed_positions (str): Allowed positions for the player (1-5, separated by underscores)

    Returns:
        pd.DataFrame: Processed DataFrame with additional columns:
            - name_lower (str): Lowercase player name with spaces replaced by underscores
            - time (float): Total time played in minutes (min + sec/60)
            - round_time (int): Rounded total time played in minutes
    """
    df["name_lower"] = df["name"].str.lower().apply(lambda x: "_".join(x.split()))
    df["time"] = df["min"] + df["sec"] / 60

    while True:
        df["round_time"] = df["time"].round(0).astype(int)
        if df["round_time"].sum() == 200:
            break
        elif df["round_time"].sum() < 200:
            df["time"] = df["time"] * 1.01
        else:
            df["time"] = df["time"] * 0.99

    return df


def extract_lineup(min: int, choices: dict, pm: dict) -> dict:
    pass  # Replace with actual code or import if already defined

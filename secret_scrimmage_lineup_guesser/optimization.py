# secret_scrimmage_lineup_guesser/optimization.py

import numpy as np
import pandas as pd
from pulp import (
    LpProblem,
    LpVariable,
    lpSum,
    LpInteger,
    LpMinimize,
    LpStatus,
    value,
    PULP_CBC_CMD,
)
import random

def run_opt(df: pd.DataFrame, seed: int, time_limit: int = 10) -> pd.DataFrame:
    # Create binary variables
    PLAYERS = df["name_lower"].tolist()
    MINS = range(1, 41)
    POS = range(1, 6)

    def extract_lineup(min: int, choices: dict, pm: dict):
        row = {"min": min}
        for pos in POS:
            for player in PLAYERS:
                if value(choices[player][min][pos]) == 1:
                    row[pos] = player
        row["plus_minus"] = value(pm[min])
        return row

    # Create the problem
    prob = LpProblem("gtown_optimize", LpMinimize)

    # Define variables
    choices = LpVariable.dicts("choice", (PLAYERS, MINS, POS), cat="Binary")
    pm = LpVariable.dicts("pm", (MINS), -10, 10, LpInteger)
    abs_min_changes = LpVariable.dicts("abs_min_changes", (range(1, 40)), lowBound=0)

    # Add a small random coefficient to each choice variable
    random.seed(seed)
    random_coefficients = {}
    for player in PLAYERS:
        for min in MINS:
            for pos in POS:
                random_coefficients[(player, min, pos)] = random.uniform(0, 1e-6)

    random_term = lpSum(
        [
            random_coefficients[(player, min, pos)] * choices[player][min][pos]
            for player in PLAYERS
            for min in MINS
            for pos in POS
        ]
    )

    prob += random_term, "sum_random"

    # Constraints
    # Each player can only be in one position per minute
    for player in PLAYERS:
        for min in MINS:
            prob += lpSum([choices[player][min][pos] for pos in POS]) <= 1

    # Exactly one player per position per minute
    for min in MINS:
        for pos in POS:
            prob += lpSum([choices[player][min][pos] for player in PLAYERS]) == 1
            for player in PLAYERS:
                if min < 40:
                    prob += (
                        abs_min_changes[min]
                        >= choices[player][min + 1][pos] - choices[player][min][pos]
                    )

    # Player-specific constraints
    for player in PLAYERS:
        # Player minutes have to match data
        total_min = df.loc[df["name_lower"] == player, "round_time"].iloc[0]
        prob += (
            lpSum([choices[player][min][pos] for min in MINS for pos in POS])
            == total_min
        )

        # Set specific player position constraints
        allowed_positions = (
            df.loc[df["name_lower"] == player, "allowed_positions"].iloc[0].split("_")
        )
        for pos in POS:
            if str(pos) not in allowed_positions:
                prob += lpSum([choices[player][min][pos] for min in MINS]) == 0

        # Set plus minus constraint
        player_pm = df.loc[df["name_lower"] == player, "plus_minus"].iloc[0]

        # Create binary variables to track if player is on court each minute
        on_court = LpVariable.dicts(f"on_court_{player}", MINS, cat="Binary")

        # Create variables for player's contribution to plus-minus each minute
        player_pm_per_min = LpVariable.dicts(
            f"pm_{player}", MINS, None, None, LpInteger
        )

        # Link on_court variable to choices
        for min in MINS:
            # Track if player is on court
            prob += on_court[min] == lpSum([choices[player][min][pos] for pos in POS])

            # If player is on court, then player_pm_per_min[min] = pm[min]
            # If player is off court, then player_pm_per_min[min] = 0
            M = 100  # A large number
            prob += player_pm_per_min[min] <= pm[min] + M * (1 - on_court[min])
            prob += player_pm_per_min[min] >= pm[min] - M * (1 - on_court[min])
            prob += player_pm_per_min[min] <= M * on_court[min]
            prob += player_pm_per_min[min] >= -M * on_court[min]

        # Sum of player's plus-minus contributions must equal their total plus-minus
        prob += lpSum([player_pm_per_min[min] for min in MINS]) == player_pm

    # Overall plus minus constraint
    prob += lpSum([pm[min] for min in MINS]) == -11

    # Write the problem
    prob.writeLP("gtown_optimize.lp")

    # Solve
    prob.solve(
        PULP_CBC_CMD(
            msg=True,
            timeLimit=time_limit,  # Time limit in seconds
        )
    )

    print("Status:", LpStatus[prob.status])
    print(prob.objective.value())

    rows = []
    for min in MINS:
        rows.append(extract_lineup(min, choices, pm))

    if prob.status != 1:
        return None
    else:
        return pd.DataFrame(rows)

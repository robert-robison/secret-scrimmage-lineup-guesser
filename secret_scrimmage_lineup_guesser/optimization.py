import pandas as pd
from pulp import (
    LpProblem,
    LpVariable,
    lpSum,
    LpInteger,
    LpMinimize,
    value,
    PULP_CBC_CMD,
)
from tqdm import tqdm


def run_opt(
    df: pd.DataFrame,
    position_preferences: dict,  # Use position_preferences directly
    invalid_pairs: list[tuple[tuple[str, int], tuple[str, int]]],
    overall_pm: int,
    pm_max: int = 3,
    time_limit: int = 10,
    objective: tuple[str, int, bool]
    | None = None,  # Updated to include position as int
) -> pd.DataFrame:
    """Runs optimization to generate lineup combinations.

    Args:
        df (pd.DataFrame): DataFrame containing player information.
        position_preferences (dict): Dictionary mapping position numbers (1-5) to lists of eligible players.
        invalid_pairs (list[tuple[tuple[str, int], tuple[str, int]]]): List of invalid player-position pairs that cannot be on court together.
        overall_pm (int): Target overall plus/minus for the game.
        pm_max (int, optional): Maximum plus/minus allowed per minute. Defaults to 3.
        time_limit (int, optional): Time limit in seconds for optimization. Defaults to 10.
        objective (tuple[str, int, bool] | None, optional): Optional objective function specification as (player, position, minimize). Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing optimized lineup assignments.
    """
    PROB_NAME = "Lineup_Optimization"
    prob = LpProblem(PROB_NAME, LpMinimize)

    PLAYERS = df["name_lower"].tolist()
    MINS = range(1, 41)  # Assuming 40 minutes
    POS = range(1, 6)  # Positions 1 through 5

    # Define choice variables: whether a player is assigned to a position at a minute
    choices = LpVariable.dicts("choice", (PLAYERS, MINS, POS), cat="Binary")

    # Define variables for plus-minus per minute
    pm = LpVariable.dicts("pm", MINS, -pm_max, pm_max, LpInteger)

    # Define additional objective if specified
    if objective is not None:
        player_obj, pos_obj, minimize = objective
        sign = 1 if minimize else -1
        prob += (
            sign * lpSum([choices[player_obj][min][pos_obj] for min in MINS]),
            "Additional_Objective",
        )

    # Constraint: Exactly one player per position per minute based on position_preferences
    for min in MINS:
        for pos in POS:
            # Players eligible for this position
            eligible_players = position_preferences[pos]
            prob += (
                lpSum([choices[player][min][pos] for player in eligible_players]) == 1,
                f"Fill_Pos_{pos}_Min_{min}",
            )

    # Constraint: Each player can only be in one position per minute
    for player in PLAYERS:
        for min in MINS:
            prob += (
                lpSum([choices[player][min][pos] for pos in POS]) <= 1,
                f"One_Position_Per_Min_{player}_{min}",
            )

    # Don't allow invalid player-position duo pairings
    for min in MINS:
        for invalid_pair in invalid_pairs:
            player1, pos1 = invalid_pair[0]
            player2, pos2 = invalid_pair[1]
            prob += (
                choices[player1][min][pos1] + choices[player2][min][pos2] <= 1,
                f"Invalid_Pair_{player1}_{pos1}_{player2}_{pos2}_{min}",
            )

    # Player-specific constraints
    for player in PLAYERS:
        # Total minutes must match the data
        total_min = df.loc[df["name_lower"] == player, "round_time"].iloc[0]
        prob += (
            lpSum([choices[player][min][pos] for min in MINS for pos in POS])
            == total_min,
            f"Total_Minutes_{player}",
        )

        # Plus-minus constraints
        player_pm = df.loc[df["name_lower"] == player, "plus_minus"].iloc[0]

        # Binary variables to track if player is on court each minute
        on_court = LpVariable.dicts(f"on_court_{player}", MINS, cat="Binary")

        # Variables for player's contribution to plus-minus each minute
        player_pm_per_min = LpVariable.dicts(
            f"pm_{player}", MINS, None, None, LpInteger
        )

        for min in MINS:
            # Link on_court variable to choices
            prob += (
                on_court[min] == lpSum([choices[player][min][pos] for pos in POS]),
                f"Link_OnCourt_{player}_{min}",
            )

            # If player is on court, player_pm_per_min[min] = pm[min]
            # If player is off court, player_pm_per_min[min] = 0
            M = 100  # A large number
            prob += (
                player_pm_per_min[min] <= pm[min] + M * (1 - on_court[min]),
                f"PM_Upper1_{player}_{min}",
            )
            prob += (
                player_pm_per_min[min] >= pm[min] - M * (1 - on_court[min]),
                f"PM_Lower1_{player}_{min}",
            )
            prob += (
                player_pm_per_min[min] <= M * on_court[min],
                f"PM_Upper2_{player}_{min}",
            )
            prob += (
                player_pm_per_min[min] >= -M * on_court[min],
                f"PM_Lower2_{player}_{min}",
            )

        # Sum of player's plus-minus contributions must equal total plus-minus
        prob += (
            lpSum([player_pm_per_min[min] for min in MINS]) == player_pm,
            f"Total_PM_{player}",
        )

    # Overall plus-minus constraint
    prob += (lpSum([pm[min] for min in MINS]) == overall_pm, "Overall_Plus_Minus")

    # Solve the problem
    prob.solve(PULP_CBC_CMD(timeLimit=time_limit))

    # Function to extract the lineup for a minute
    def extract_lineup(minute, choices):
        lineup = {pos: None for pos in POS}
        for pos in POS:
            for player in PLAYERS:
                if value(choices[player][minute][pos]) == 1:
                    lineup[pos] = player
                    break
        lineup["plus_minus"] = value(pm[minute])
        return {"min": minute, **lineup}

    rows = []
    for min in MINS:
        rows.append(extract_lineup(min, choices))

    if prob.status != 1:
        print("No optimal solution found.")
        return None
    else:
        return pd.DataFrame(rows)


def get_positional_min_max(
    df: pd.DataFrame,
    position_preferences: dict,
    invalid_pairs: list[tuple[tuple[str, int], tuple[str, int]]],
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the minimum and maximum minutes each player can play at each position.

    This function iterates through each player and determines the minimum and maximum number of
    minutes they can be assigned to each allowed position based on the overall plus-minus constraint.
    It utilizes the `run_opt` function to solve the optimization problem for each position.

    Args:
        df (pd.DataFrame): DataFrame containing player information, including name, allowed positions,
                           round time, and plus-minus values.
        position_preferences (dict): A dictionary where the keys are positions and the values are lists of players.
        invalid_pairs (list[tuple[tuple[str, int], tuple[str, int]]]): A list of tuples representing invalid player-position duo pairings.
        overall_pm (int): The overall plus-minus value to be achieved.
        pm_max (int, optional): The maximum absolute value for plus-minus per minute. Defaults to 3.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - The first DataFrame (`min_rows`) holds the minimum minutes each player can play at each position.
            - The second DataFrame (`max_rows`) holds the maximum minutes each player can play at each position.
    """
    min_rows = []
    max_rows = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        # initialize new rows
        name = row["name_lower"]
        min_row = {"player": name, **{k: 0 for k in range(1, 6)}}
        max_row = {"player": name, **{k: 0 for k in range(1, 6)}}

        # update new rows with allowed positions
        possible_positions = [
            pos for pos in position_preferences if name in position_preferences[pos]
        ]
        if len(possible_positions) == 1:
            min_row[possible_positions[0]] = row["round_time"]
            max_row[possible_positions[0]] = row["round_time"]
        else:
            for pos in possible_positions:
                # Get minimum possible minutes for player at position
                min_time_df = run_opt(
                    df,
                    position_preferences,
                    invalid_pairs,
                    objective=(row["name_lower"], pos, True),
                    **kwargs,
                )
                if min_time_df is not None:
                    min_minutes = (min_time_df[pos] == name).sum()
                else:
                    min_minutes = 0
                min_row[pos] = min_minutes

                # Get maximum possible minutes for player at position
                max_time_df = run_opt(
                    df,
                    position_preferences,
                    invalid_pairs,
                    objective=(row["name_lower"], pos, False),
                    **kwargs,
                )
                if max_time_df is not None:
                    max_minutes = (max_time_df[pos] == name).sum()
                else:
                    max_minutes = 0
                max_row[pos] = max_minutes
        min_rows.append(min_row)
        max_rows.append(max_row)
    min_df = pd.DataFrame(min_rows)
    max_df = pd.DataFrame(max_rows)
    # Format as min-max strings, showing single value when min equals max
    result_df = min_df.copy()
    for col in range(1, 6):
        result_df[col] = min_df.apply(
            lambda x: str(int(x[col]))
            if x[col] == max_df.loc[x.name, col]
            else f"{int(x[col])}-{int(max_df.loc[x.name, col])}",
            axis=1,
        )
    return result_df

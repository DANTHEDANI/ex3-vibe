"""
Feature Engineering Module.

Responsible for creating the trainee trajectory from raw Kaggle CSVs.
"""
from pathlib import Path

import pandas as pd

from ex3.shared.config import cfg
from ex3.shared.constants import (
    COL_DAY,
    COL_EQUIPMENT,
    COL_PROGRAM_ID,
    COL_PROGRAM_LENGTH,
    COL_REPS,
    COL_SETS,
    COL_WEEK,
)


def extract_trajectory() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts a synthetic trajectory for a single program.

    Returns:
        A tuple of (states_df, actions_df).
    """
    data_dir = Path(cfg.data_dir)
    summary_path = data_dir / "program_summary.csv"
    detailed_path = data_dir / "programs_detailed_boostcamp_kaggle.csv"

    if not summary_path.exists() or not detailed_path.exists():
        raise FileNotFoundError("Dataset not found. Please run data_pipeline first.")

    summary_df = pd.read_csv(summary_path)
    detailed_df = pd.read_csv(detailed_path)

    # 1. Filter program
    possible_programs = summary_df.copy()
    if COL_EQUIPMENT in possible_programs.columns:
        possible_programs = possible_programs[possible_programs[COL_EQUIPMENT].astype(str).str.contains("Full Gym", na=False, case=False)]
    if COL_PROGRAM_LENGTH in possible_programs.columns:
        possible_programs[COL_PROGRAM_LENGTH] = pd.to_numeric(possible_programs[COL_PROGRAM_LENGTH], errors="coerce")
        possible_programs = possible_programs[possible_programs[COL_PROGRAM_LENGTH] >= 8]
    if "time_per_workout" in possible_programs.columns:
        # Time per workout might be something like "45 - 60 minutes". We can extract the lower bound.
        time_ext = possible_programs["time_per_workout"].astype(str).str.extract(r'(\d+)')
        if not time_ext.empty and 0 in time_ext.columns:
            possible_programs["time_lower"] = time_ext[0].astype(float)
            possible_programs = possible_programs[(possible_programs["time_lower"] >= 45) & (possible_programs["time_lower"] <= 120)]

    if possible_programs.empty:
        possible_programs = summary_df

    chosen_program_id = possible_programs.iloc[0][COL_PROGRAM_ID]

    # 2. Extract detailed rows
    program_rows = detailed_df[detailed_df[COL_PROGRAM_ID] == chosen_program_id].copy()

    # 3. Clean and Build Daily Sessions
    program_rows[COL_SETS] = pd.to_numeric(program_rows[COL_SETS], errors="coerce").fillna(0)
    program_rows[COL_REPS] = pd.to_numeric(program_rows[COL_REPS], errors="coerce").fillna(0)

    # Process negative values as time in seconds
    def compute_volume_and_duration(row):
        sets = row[COL_SETS]
        reps = row[COL_REPS]
        if sets < 0:
            sets = abs(sets)

        if reps < 0:
            # Negative reps = time in seconds
            duration = abs(reps) * sets
            volume = duration / 60.0 # simple heuristic to map seconds to volume
        else:
            duration = sets * reps * 4.0 # assume 4 seconds per rep
            volume = sets * reps
        return pd.Series([volume, duration])

    program_rows[['volume', 'duration']] = program_rows.apply(compute_volume_and_duration, axis=1)

    # Muscle distribution mapping (mocking based on exercise name hashing)
    def assign_muscle_group(name):
        muscles = ["chest", "back", "legs", "core", "arms"]
        return muscles[hash(str(name)) % len(muscles)]

    program_rows["muscle_group"] = program_rows["exercise_name"].apply(assign_muscle_group)

    # Group by week and day
    daily_stats = program_rows.groupby([COL_WEEK, COL_DAY]).agg(
        total_volume=("volume", "sum"),
        session_duration_t=("duration", "sum"),
        exercise_count=("volume", "count")
    ).reset_index()

    # Calculate muscle distribution per day
    muscle_vols = program_rows.groupby([COL_WEEK, COL_DAY, "muscle_group"])["volume"].sum().unstack(fill_value=0)
    # Avoid division by zero
    row_sums = muscle_vols.sum(axis=1).replace(0, 1)
    muscle_vols = muscle_vols.div(row_sums, axis=0) # Normalize

    daily_stats = daily_stats.merge(muscle_vols, on=[COL_WEEK, COL_DAY], how="left")

    # 4. Fill missing days with Rest Days
    if not daily_stats.empty:
        min_week, max_week = int(daily_stats[COL_WEEK].min()), int(daily_stats[COL_WEEK].max())
    else:
        min_week, max_week = 1, 1

    all_weeks_days = pd.MultiIndex.from_product(
        [range(min_week, max_week + 1), range(1, 8)], names=[COL_WEEK, COL_DAY]
    ).to_frame(index=False)

    full_schedule = pd.merge(all_weeks_days, daily_stats, on=[COL_WEEK, COL_DAY], how="left").fillna(0)

    # Build continuous trajectory
    states = []
    actions = []

    for _, row in full_schedule.iterrows():
        state = {
            "rolling_load": row["total_volume"],
            "session_duration_t": row["session_duration_t"],
            "week_index_t": row[COL_WEEK],
            "day_in_cycle_t": row[COL_DAY],
            "exercise_count": row["exercise_count"],
            "chest": row.get("chest", 0),
            "back": row.get("back", 0),
            "legs": row.get("legs", 0),
            "core": row.get("core", 0),
            "arms": row.get("arms", 0)
        }
        states.append(state)

        # Action: Cluster total_volume into discrete bins
        action_val = 0 if row["total_volume"] == 0 else min(int(row["total_volume"] // 10) + 1, 5)
        actions.append({"action": action_val})

    states_df = pd.DataFrame(states)
    actions_df = pd.DataFrame(actions)

    return states_df, actions_df

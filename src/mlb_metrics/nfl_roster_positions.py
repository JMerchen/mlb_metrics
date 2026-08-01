"""DK Classic roster-slot eligibility for NFL - simpler than MLB's
roster_positions.py, which needs a whole separate MLB Stats API fetch
because Statcast carries no fielding-position data at all. `position` is
already a real column on every nflreadpy table this project uses
(confirmed live - see nfl_data.py's module docstring), so this module is
a pure mapping function, no network fetch needed.

The one real piece of NFL-specific logic: RB/WR/TE are FLEX-eligible, so
`build_eligibility_table` emits TWO rows for each of them (their own
slot AND "FLEX"), while QB/DST get exactly one. This is the entire
mechanism Phase 6's optimizer needs for FLEX - feeding two pool rows for
the same player into dfs_optimizer.py's already-existing "cap any
duplicated key group at 1 selected row" MILP constraint
(dfs_optimizer.py:199) handles FLEX with zero new constraint types, the
same way `roster_positions.py` feeds `dfs_optimizer.build_player_pool`
one row per player - this just sometimes feeds two.
"""

import pandas as pd

# nflreadpy `position` -> DraftKings Classic NFL slot. Only DK Classic's
# four skill/passing slots have a real fantasy role - every other real
# NFL position (OL/DL/LB/DB/K/etc.) has no DK Classic slot at all and is
# excluded, not defaulted (mirrors roster_positions.py's own "DH has no
# slot, excluded outright" precedent). DST is not a `position` value in
# nflreadpy's player-level tables at all - it's a team-level unit
# (nfl_dst.py), handled entirely separately from this player-level table.
NFL_POSITION_TO_DK_SLOT = {
    "QB": "QB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
}

NFL_FLEX_ELIGIBLE_POSITIONS = ("RB", "WR", "TE")


def build_eligibility_table(players_df: pd.DataFrame) -> pd.DataFrame:
    """Parses `players_df` (must carry `player_id`, `position`) into
    [player_id, position, dk_slot]. A player whose `position` has no DK
    Classic slot (NFL_POSITION_TO_DK_SLOT) is excluded entirely, not
    defaulted. Every RB/WR/TE gets TWO rows - one with dk_slot equal to
    their own position, one with dk_slot="FLEX" - QB gets exactly one."""
    eligible = players_df[players_df["position"].isin(NFL_POSITION_TO_DK_SLOT)].copy()
    eligible["dk_slot"] = eligible["position"].map(NFL_POSITION_TO_DK_SLOT)
    own_slot = eligible[["player_id", "position", "dk_slot"]]

    flex_eligible = eligible[eligible["position"].isin(NFL_FLEX_ELIGIBLE_POSITIONS)].copy()
    flex_eligible["dk_slot"] = "FLEX"
    flex_rows = flex_eligible[["player_id", "position", "dk_slot"]]

    return pd.concat([own_slot, flex_rows], ignore_index=True).sort_values(["player_id", "dk_slot"]).reset_index(drop=True)

import pandas as pd
import pytest

from mlb_metrics import config, hitter_views


def _pa(game_pk, date, batter, pitcher, event, ab_no=1, game_type="R"):
    return {"game_pk": game_pk, "game_date": pd.Timestamp(date), "game_type": game_type, "batter": batter,
            "pitcher": pitcher, "p_throws": "R", "events": event, "at_bat_number": ab_no, "pitch_number": 1}


def _season():
    """Hitter 1 faces only stingy pitcher 900; hitter 2 only generous 901.
    Both go 1-for-4 every game. Hitter 3 doubles every game but stops after 5."""
    rows = []
    stingy = ["field_out"] * 9 + ["single"]           # pitcher 900 allows .100 to everyone else
    generous = ["single", "double", "field_out", "home_run", "field_out"]
    for day in range(1, 41):
        date = pd.Timestamp("2026-05-01") + pd.Timedelta(days=day)
        for i, event in enumerate(["single", "field_out", "field_out", "field_out"]):
            rows.append(_pa(day, date, 1, 900, event, i + 1))
            rows.append(_pa(day, date, 2, 901, event, i + 10))
        for i, event in enumerate(stingy):
            rows.append(_pa(day, date, 50 + i, 900, event, i + 20))
        for i, event in enumerate(generous):
            rows.append(_pa(day, date, 70 + i, 901, event, i + 40))
        if day <= 5:
            rows.append(_pa(day, date, 3, 901, "double", 60))
    rows.append(_pa(99, "2026-10-05", 1, 900, "home_run", 1, game_type="F"))   # playoffs: left out
    return pd.DataFrame(rows)


def test_plate_appearances_count_obp_and_slg_pieces():
    pas = hitter_views.plate_appearances(pd.DataFrame([
        _pa(1, "2026-04-01", 1, 9, "double"), _pa(1, "2026-04-01", 1, 9, "walk", 2),
        _pa(1, "2026-04-01", 1, 9, "sac_fly", 3), _pa(1, "2026-04-01", 1, 9, None, 4),
        _pa(2, "2026-10-01", 1, 9, "single", 1, game_type="F"),
    ]))
    assert pas[["PA", "AB", "H", "TB", "OB"]].sum().tolist() == [3, 1, 1, 2, 2]


def test_wops_blends_obp_and_slg_and_carries_a_missed_window():
    pas = hitter_views.plate_appearances(_season())
    wops = hitter_views.compute_wops(pas).set_index("key_mlbam")
    # Hitter 3 hit a double every game but stopped after day 5: his 10- and
    # 30-day windows are empty, so they carry his longer windows, not zeros.
    league = hitter_views.league_rates(pas)
    k = config.WAVE_SHRINKAGE_STRENGTH
    season_slg = (10 + league["tb"] * k) / (5 + k)
    assert wops.loc[3, "WOPS_SLG"] == pytest.approx(season_slg)
    assert wops.loc[1, "WOPS"] == pytest.approx(wops.loc[1, "WOPS_OBP"] + wops.loc[1, "WOPS_SLG"])


def test_pitchers_faced_credits_hits_off_tough_pitching():
    pas = hitter_views.plate_appearances(_season())
    factors = hitter_views.pitchers_faced_factors(pas).set_index("key_mlbam")
    # Same 1-for-4 every day, but hitter 1 did it against the stingy pitcher.
    assert factors.loc[1, "wave_factor"] > 1 > factors.loc[2, "wave_factor"]
    assert factors.loc[1, "cave_factor"] > 1 > factors.loc[2, "cave_factor"]
    assert factors.loc[1, "slg_factor"] > factors.loc[2, "slg_factor"]


def test_hitter_views_wave_cave_wops_and_adjusted_versions():
    views = hitter_views.hitter_views(_season()).set_index("key_mlbam")
    assert views.loc[1, "WAVE"] == pytest.approx(views.loc[2, "WAVE"])            # identical lines
    assert views.loc[1, "WAVE_adj"] > views.loc[2, "WAVE_adj"]
    assert views.loc[1, "CAVE"] == pytest.approx(views.loc[2, "CAVE"])
    assert views.loc[1, "CAVE_adj"] > views.loc[1, "CAVE"]
    assert views.loc[1, "WOPS_adj"] > views.loc[2, "WOPS_adj"]
    # Hitter 3's season ended early; his WAVE still reflects how he hit.
    assert views.loc[3, "WAVE"] > views.loc[1, "WAVE"]


def test_carry_forward_is_opt_in_and_only_fills_missed_windows():
    from mlb_metrics import data, hitters

    dt = data.completed_events(_season().query("game_type == 'R'"), ["game_date", "batter", "p_throws", "events"])
    plain = hitters.compute_wave(dt).set_index("key_mlbam")
    carried = hitters.compute_wave(dt, carry_forward=True).set_index("key_mlbam")
    # The pick models' WAVE is unchanged for a hitter who played every window ...
    assert carried.loc[1, "WAVE"] == pytest.approx(plain.loc[1, "WAVE"])
    # ... while one who sat out the recent windows no longer reads them as zero.
    assert carried.loc[3, "WAVE"] > plain.loc[3, "WAVE"]


def test_wave_is_on_the_batting_average_scale_walks_left_out():
    """A walk isn't an at-bat: two hitters who go 1-for-4 every game read
    the same WAVE whether or not one also walks (the picks' wave.csv rate
    still counts the walk as a trial)."""
    season = _season()
    walks = [_pa(day, pd.Timestamp("2026-05-01") + pd.Timedelta(days=day), 2, 901, "walk", 9)
             for day in range(1, 41)]
    views = hitter_views.season_wave_cave(pd.concat([season, pd.DataFrame(walks)], ignore_index=True)).set_index("key_mlbam")
    assert views.loc[2, "WAVE"] == pytest.approx(views.loc[1, "WAVE"])


def test_compute_wave_at_bats_only_switch():
    from mlb_metrics import data, hitters

    walks = [_pa(day, pd.Timestamp("2026-05-01") + pd.Timedelta(days=day), 2, 901, "walk", 9) for day in range(1, 41)]
    season = pd.concat([_season(), pd.DataFrame(walks)], ignore_index=True).query("game_type == 'R'")
    dt = data.completed_events(season, ["game_date", "batter", "p_throws", "events"])
    per_pa = hitters.compute_wave(dt).set_index("key_mlbam")                       # the default: walks count
    assert not config.WAVE_AT_BATS_ONLY
    assert per_pa.loc[2, "WAVE"] < per_pa.loc[1, "WAVE"]
    per_ab = hitters.compute_wave(dt, at_bats_only=True).set_index("key_mlbam")
    assert per_ab.loc[2, "WAVE"] == pytest.approx(per_ab.loc[1, "WAVE"])
    assert per_ab.loc[1, "WAVE"] > per_pa.loc[1, "WAVE"]

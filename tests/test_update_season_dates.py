import datetime
import importlib.util
import json
import os

from mlb_metrics import config

_spec = importlib.util.spec_from_file_location(
    "update_season_dates", os.path.join(os.path.dirname(__file__), "..", "scripts", "update_season_dates.py")
)
update_season_dates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(update_season_dates)

D = datetime.date
SEASONS = {2026: (D(2026, 3, 25), D(2026, 9, 27)), 2027: (D(2027, 3, 25), D(2027, 9, 26))}


def test_off_season_keeps_the_finished_season_until_opening_day():
    assert update_season_dates.current_season(D(2026, 10, 23), SEASONS.get) == (2026, *SEASONS[2026])
    assert update_season_dates.current_season(D(2027, 2, 20), SEASONS.get) == (2026, *SEASONS[2026])
    assert update_season_dates.current_season(D(2027, 3, 25), SEASONS.get) == (2027, *SEASONS[2027])


def test_next_season_not_published_yet_stays_on_last_season():
    only_2026 = {2026: SEASONS[2026]}
    assert update_season_dates.current_season(D(2027, 1, 5), only_2026.get) == (2026, *SEASONS[2026])


def test_config_reads_the_season_file_and_falls_back_without_it(tmp_path):
    path = tmp_path / "season_dates.json"
    path.write_text(json.dumps({"season": 2027, "regular_season_start": "2027-03-25", "regular_season_end": "2027-09-26"}))
    assert config._season_dates(str(path)) == (D(2027, 3, 25), D(2027, 9, 26))
    assert config._season_dates(str(tmp_path / "missing.json")) == (D(2026, 3, 25), D(2026, 9, 27))
    assert config.SEASON_END - config.REGULAR_SEASON_END == datetime.timedelta(days=13)

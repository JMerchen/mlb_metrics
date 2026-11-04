import datetime

import pandas as pd
import pytest

from mlb_metrics import config, nfl_totals, nfl_weather


def test_precip_from_text_reads_rain_snow_and_dry():
    assert nfl_weather.precip_from_text("Rain Temp: 51° F, Humidity: 91%, Wind: SW 17 mph") == "rain"
    assert nfl_weather.precip_from_text("Light Snow Temp: 25° F") == "snow"
    assert nfl_weather.precip_from_text("Scattered Showers Temp: 60° F") == "rain"
    assert nfl_weather.precip_from_text("Sunny Temp: 53° F, Wind: SE 7 mph") == "dry"
    assert nfl_weather.precip_from_text(None) is None
    assert nfl_weather.precip_from_text(float("nan")) is None


def test_historical_weather_takes_readings_from_the_schedule_and_rain_from_the_text():
    schedules = pd.DataFrame({"game_id": ["a", "b", "c"], "temp": [51, 25, None], "wind": [17, 3, None]})
    pbp = pd.DataFrame({"game_id": ["a", "a", "b"], "weather": ["Rain Temp: 51° F", "Rain Temp: 51° F", "Snow"]})
    out = nfl_weather.historical_weather(schedules, pbp).set_index("game_id")
    assert out.loc["a", "weather_rain"] == 1.0 and out.loc["b", "weather_rain"] == 0.0
    assert pd.isna(out.loc["c", "weather_rain"])
    assert out.loc["a", "weather_wind"] == 17 and out.loc["b", "weather_temp"] == 25


def test_weather_features_are_zero_indoors_and_the_average_when_unknown():
    games = pd.DataFrame({"weather_temp": [30.0, 30.0, None], "weather_wind": [15.0, 15.0, None],
                          "weather_rain": [0.6, 0.6, None]})
    features = nfl_weather.weather_features(games, pd.Series([0.0, 1.0, 0.0]))
    assert features.loc[0].to_dict() == pytest.approx({"wx_wind": 10.0, "wx_cold": 10.0, "wx_rain": 0.6})
    assert features.loc[1].to_dict() == {"wx_wind": 0.0, "wx_cold": 0.0, "wx_rain": 0.0}
    assert features.loc[2].to_dict() == pytest.approx(config.NFL_WEATHER_OUTDOOR_MEANS)
    # No weather columns at all: every outdoor game is the average one.
    bare = nfl_weather.weather_features(pd.DataFrame(index=[0]), pd.Series([0.0]))
    assert bare.loc[0].to_dict() == pytest.approx(config.NFL_WEATHER_OUTDOOR_MEANS)


def _hourly(start="2026-11-08T12:00", hours=8, temp=40.0, wind=12.0, chance=None, code=61):
    times = pd.date_range(start, periods=hours, freq="h")
    return {
        "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
        "temperature_2m": [temp] * hours, "wind_speed_10m": [wind] * hours,
        "precipitation_probability": chance or [10] * hours, "weather_code": [code] * hours,
    }


def test_summarize_hourly_covers_kickoff_through_the_game():
    chance = [90, 10, 20, 70, 30, 10, 0, 95]  # 12:00 ... 19:00
    out = nfl_weather.summarize_hourly(_hourly(chance=chance), datetime.datetime(2026, 11, 8, 13, 25))
    # 13:00-16:00: highest chance 70%, not the 90% before or 95% after.
    assert out["weather_rain"] == 0.7
    assert out["weather_temp"] == 40.0 and out["weather_wind"] == 12.0 and out["weather_code"] == 61
    assert nfl_weather.summarize_hourly(_hourly(), datetime.datetime(2026, 11, 12, 13, 0)) is None


def test_snow_is_no_rain():
    out = nfl_weather.summarize_hourly(_hourly(chance=[90] * 8, code=73), datetime.datetime(2026, 11, 8, 13, 0))
    assert out["weather_rain"] == 0.0 and out["weather_code"] == 73


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_game_forecasts_skips_indoor_games_and_survives_a_failure():
    games = pd.DataFrame({
        "game_id": ["out", "dome", "broken", "later"], "stadium_id": ["CHI98", "DET00", "GNB00", "BUF00"],
        "gameday": ["2026-11-08"] * 4, "gametime": ["13:00"] * 4,
    })
    calls = []

    def get(url, params, timeout):
        calls.append(params["latitude"])
        if params["latitude"] == nfl_weather.STADIUMS["GNB00"][0]:
            raise ConnectionError("no network")
        return _Response({"hourly": _hourly(start="2026-11-08T00:00", hours=48)})

    out = nfl_weather.game_forecasts(games, pd.Series([0.0, 1.0, 0.0, 0.0]), ["out", "dome", "broken"], get=get)
    assert out["game_id"].tolist() == ["out"]
    assert len(calls) == 2  # not the dome, not the game outside the window
    assert out.loc[0, "weather_rain"] == 0.1


def test_rain_and_wind_lower_the_predicted_total():
    games = pd.DataFrame({"game_id": ["dry", "wet", "dome"], "home_team": ["CHI"] * 3, "away_team": ["GB"] * 3,
                          "roof": ["outdoors", "outdoors", "dome"],
                          "weather_temp": [60.0, 35.0, 35.0], "weather_wind": [5.0, 20.0, 20.0],
                          "weather_rain": [0.0, 1.0, 1.0]})
    ratings = pd.DataFrame({"team": ["CHI", "GB"], "off_epa": 0.0, "def_epa": 0.0, "off_plays": 60.0, "def_plays": 60.0})
    features = nfl_totals.matchup_features(ratings, games).fillna(0.0)
    coefficients = {name: 0.0 for name in nfl_totals.TOTALS_FEATURE_COLUMNS}
    coefficients.update(const=20.0, wx_wind=-0.18, wx_cold=-0.1, wx_rain=-2.826)
    totals = nfl_totals.predict_team_points(features, coefficients).groupby(features["game_id"]).sum()
    # Per team: 15 mph over 5 (-2.7), 5 degrees under 40 (-0.5), rain (-2.826).
    assert totals["dry"] - totals["wet"] == pytest.approx(2 * (2.7 + 0.5 + 2.826))
    # Under a roof the weather outside doesn't count.
    assert totals["dome"] == pytest.approx(40.0)

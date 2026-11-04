"""Game-day weather for NFL games: the forecast at kickoff for the live
picks, and what the weather was for past games, for fitting.

User request (2026-11-04): pull weather forecasts into the NFL models and
simulations the day before a game (the morning of it for Monday and
Thursday games) - including rain, after a soaked Falcons-Bucs game in
which hardly a pass was completed.

Forecasts come from Open-Meteo (free, no key): the hourly forecast at each
stadium from the kickoff hour through the next three (about how long a
game lasts) - mean temperature and wind and the highest chance of rain.
They're pulled only for games whose final injury report is out (played
today or tomorrow - the Thursday, Saturday and Monday morning runs), the
same games those runs reprice, so a pick made the day before has the
day-before forecast. Indoor games (a dome, or a retractable roof usually
closed - nfl_totals.dome_flags) get none.

For fitting, past games use nflverse's own readings: `temp` and `wind`
from the schedule, and rain from the play-by-play's `weather` text
("Rain Temp: 51° F, ..."). Rain is a 0/1 there; live it's the forecast's
chance, so a 60% chance moves a prediction 60% as far as a sure soaking.

What it changes (scripts/backtest_nfl_totals.py, 2026-11-04): each team
scores about 0.18 points less per mph of wind over 5, 0.1 less per degree
below 40°F and 2.8 less in rain (130 rainy outdoor games since 2016 came
in 3.5 points under Vegas's closing total). Snow measured nothing on its
own over 22 games, so it isn't a term; the cold term covers most snow
games.
"""

import datetime

import pandas as pd

from mlb_metrics import config

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# The game-level weather columns a games frame carries (NaN = unknown).
WEATHER_INPUT_COLUMNS = ["weather_temp", "weather_wind", "weather_rain"]
# What the pick log and the game card keep: the inputs plus the forecast's
# weather code (for the card's symbol) and whether the game is indoors.
WEATHER_DISPLAY_COLUMNS = [*WEATHER_INPUT_COLUMNS, "weather_code", "weather_indoor"]
# The points model's terms (nfl_totals.TOTALS_FEATURE_COLUMNS).
WEATHER_FEATURE_COLUMNS = ["wx_wind", "wx_cold", "wx_rain"]

WIND_FROM_MPH = 5
COLD_BELOW_F = 40
GAME_HOURS = 4

# Stadium locations by nflverse stadium_id (latitude, longitude).
STADIUMS = {
    "ATL97": (33.7554, -84.4008), "BAL00": (39.2780, -76.6227), "BOS00": (42.0909, -71.2643),
    "BUF00": (42.7738, -78.7870), "BUF01": (42.7738, -78.7870), "CAR00": (35.2258, -80.8528),
    "CHI98": (41.8623, -87.6167), "CIN00": (39.0955, -84.5161), "CLE00": (41.5061, -81.6995),
    "DAL00": (32.7473, -97.0945), "DEN00": (39.7439, -105.0201), "DET00": (42.3400, -83.0456),
    "GNB00": (44.5013, -88.0622), "HOU00": (29.6847, -95.4107), "IND00": (39.7601, -86.1639),
    "JAX00": (30.3239, -81.6373), "KAN00": (39.0489, -94.4839), "LAX01": (33.9535, -118.3392),
    "MIA00": (25.9580, -80.2389), "MIN01": (44.9737, -93.2581), "NAS00": (36.1665, -86.7713),
    "NOR00": (29.9511, -90.0812), "NYC01": (40.8135, -74.0745), "PHI00": (39.9008, -75.1675),
    "PHO00": (33.5276, -112.2626), "PIT00": (40.4468, -80.0158), "SEA00": (47.5952, -122.3316),
    "SFO01": (37.4030, -121.9700), "TAM00": (27.9759, -82.5033), "VEG00": (36.0909, -115.1833),
    "WAS00": (38.9077, -76.8645),
    # Games abroad.
    "LON00": (51.5560, -0.2796), "LON01": (51.6043, -0.0664), "LON02": (51.6043, -0.0664),
    "MEX00": (19.3029, -99.1505), "GER00": (48.2188, 11.6247), "FRA00": (50.0686, 8.6455),
    "MUN01": (48.2188, 11.6247), "SAO00": (-23.5453, -46.4742), "MAD01": (40.4531, -3.6883),
    "RIO00": (-22.9122, -43.2302), "MEL00": (-37.8200, 144.9834), "PAR00": (48.9245, 2.3602),
    "DUB00": (53.3609, -6.2512),
}

# Open-Meteo weather codes (WMO) that mean snow.
SNOW_CODES = {71, 73, 75, 77, 85, 86}


def precip_from_text(text) -> str | None:
    """'rain', 'snow' or 'dry' from nflverse's play-by-play weather text;
    None when there's no text (an indoor game, mostly)."""
    if not isinstance(text, str) or not text.strip():
        return None
    lowered = text.lower()
    if "snow" in lowered or "flurr" in lowered:
        return "snow"
    if any(word in lowered for word in ("rain", "shower", "drizzle", "storm", "thunder")):
        return "rain"
    return "dry"


def historical_weather(schedules: pd.DataFrame, pbp: pd.DataFrame | None) -> pd.DataFrame:
    """game_id plus WEATHER_INPUT_COLUMNS for past games: nflverse's
    temperature and wind readings, and rain (1/0) from the play-by-play's
    weather text. A game with no text has unknown rain."""
    out = pd.DataFrame({"game_id": schedules["game_id"]})
    out["weather_temp"] = pd.to_numeric(schedules.get("temp"), errors="coerce").to_numpy()
    out["weather_wind"] = pd.to_numeric(schedules.get("wind"), errors="coerce").to_numpy()
    rain = pd.Series(float("nan"), index=out.index)
    if pbp is not None and not pbp.empty and "weather" in pbp.columns:
        text = pbp.drop_duplicates("game_id").set_index("game_id")["weather"]
        kind = out["game_id"].map(text).map(precip_from_text)
        rain = kind.map({"rain": 1.0, "snow": 0.0, "dry": 0.0})
    out["weather_rain"] = rain.astype(float)
    return out


def weather_features(games: pd.DataFrame, indoor: pd.Series) -> pd.DataFrame:
    """The points model's weather terms for each game (indexed like
    `games`): wind over WIND_FROM_MPH, degrees below COLD_BELOW_F and the
    chance of rain - all 0 indoors. An outdoor game with no reading or
    forecast (the Tuesday pick, made before one is pulled) takes the
    average outdoor game's (config.NFL_WEATHER_OUTDOOR_MEANS), so leaving
    weather out doesn't push its total up."""
    def column(name):
        if name in games.columns:
            return pd.to_numeric(games[name], errors="coerce")
        return pd.Series(float("nan"), index=games.index)

    raw = pd.DataFrame({
        "wx_wind": (column("weather_wind") - WIND_FROM_MPH).clip(lower=0),
        "wx_cold": (COLD_BELOW_F - column("weather_temp")).clip(lower=0),
        "wx_rain": column("weather_rain").clip(0, 1),
    }, index=games.index)
    outdoor = pd.Series(indoor, index=games.index).fillna(0.0) < 0.5
    for name in WEATHER_FEATURE_COLUMNS:
        raw[name] = raw[name].fillna(config.NFL_WEATHER_OUTDOOR_MEANS[name]).where(outdoor, 0.0)
    return raw


def _kickoff(gameday, gametime) -> datetime.datetime | None:
    """Kickoff on the US East Coast clock nflverse's times are in."""
    try:
        day = pd.to_datetime(gameday).date()
        hour, minute = (int(part) for part in str(gametime or "13:00").split(":")[:2])
    except (TypeError, ValueError):
        return None
    return datetime.datetime.combine(day, datetime.time(hour, minute))


def summarize_hourly(hourly: dict, kickoff: datetime.datetime) -> dict | None:
    """Open-Meteo's hourly arrays (times on the East Coast clock) reduced
    to the game: mean temperature and wind and the highest chance of rain
    from the kickoff hour through GAME_HOURS hours, and the most common
    weather code. Snow counts as no rain (it isn't a term; see the module
    docstring). None if the forecast doesn't cover the game."""
    times = pd.to_datetime(pd.Series(hourly.get("time", [])))
    if times.empty:
        return None
    start = kickoff.replace(minute=0)
    window = (times >= start) & (times < start + datetime.timedelta(hours=GAME_HOURS))
    if not window.any():
        return None

    def values(key):
        return pd.to_numeric(pd.Series(hourly.get(key, [None] * len(times))), errors="coerce")[window.to_numpy()]

    codes = values("weather_code").dropna().astype(int)
    code = int(codes.mode().iloc[0]) if not codes.empty else None
    chance = values("precipitation_probability").max()
    rain = 0.0 if code in SNOW_CODES else (float(chance) / 100 if pd.notna(chance) else float("nan"))
    temp, wind = values("temperature_2m").mean(), values("wind_speed_10m").mean()
    return {
        "weather_temp": round(float(temp), 1) if pd.notna(temp) else float("nan"),
        "weather_wind": round(float(wind), 1) if pd.notna(wind) else float("nan"),
        "weather_rain": round(rain, 2) if pd.notna(rain) else float("nan"),
        "weather_code": code,
    }


def fetch_forecast(lat: float, lon: float, kickoff: datetime.datetime, get=None) -> dict | None:
    """The game's forecast at (lat, lon), or None if it can't be had."""
    if get is None:
        import requests
        get = requests.get
    day = kickoff.date()
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,precipitation_probability,weather_code",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "America/New_York",
        "start_date": day.isoformat(), "end_date": (day + datetime.timedelta(days=1)).isoformat(),
    }
    response = get(FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    return summarize_hourly(response.json().get("hourly", {}), kickoff)


def game_forecasts(games: pd.DataFrame, indoor: pd.Series, game_ids, get=None) -> pd.DataFrame:
    """game_id plus WEATHER_INPUT_COLUMNS and weather_code for the outdoor
    games among `game_ids` (needs stadium_id, gameday, gametime). A game
    whose stadium isn't known or whose forecast fails is left out - it
    takes the average outdoor game's weather - and a failure never stops
    the run."""
    rows = []
    wanted = set(game_ids)
    outdoor = pd.Series(indoor, index=games.index).fillna(0.0) < 0.5
    for index, game in games[games["game_id"].isin(wanted) & outdoor].iterrows():
        where = STADIUMS.get(str(game.get("stadium_id")))
        kickoff = _kickoff(game.get("gameday"), game.get("gametime"))
        if where is None or kickoff is None:
            print(f"No weather for {game['game_id']}: stadium {game.get('stadium_id')} or kickoff unknown.")
            continue
        try:
            forecast = fetch_forecast(*where, kickoff, get=get)
        except Exception as exc:  # network, HTTP or a malformed reply
            print(f"WARNING: weather forecast failed for {game['game_id']} ({exc}).")
            continue
        if forecast:
            rows.append({"game_id": game["game_id"], **forecast})
    return pd.DataFrame(rows, columns=["game_id", *WEATHER_INPUT_COLUMNS, "weather_code"])

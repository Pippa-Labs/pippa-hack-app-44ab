"""UK Tide Times: when is the next high tide at a UK beach?

How it works, in three requests at most:

1. Turn the place name into coordinates. Open-Meteo's geocoder first (towns),
   then Nominatim if that finds nothing (it knows beaches and bays by name).
   Both are limited to the UK.
2. Ask the Open-Meteo marine forecast for the hourly sea level at the nearest
   sea grid cell. `sea_level_height_msl` includes the tide, in metres above
   mean sea level.
3. Find the first peak after now. The data is hourly, so the time of each
   peak is refined with a parabola through the three points around it, which
   puts it within a few minutes of the model's own high water.

No API key is needed for any of them.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from pippa_agentapp_sdk import (
    AgentAppError,
    Call,
    Manifest,
    RateLimitedError,
    Result,
    ServiceUnavailableError,
    load_manifest_file,
    validate_manifest,
)

MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")

#: Nominatim answers 403 without a descriptive User-Agent.
USER_AGENT = "pippa-agent-app-uk-tide-times/0.1 (https://github.com/Pippa-Labs)"

OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"

UK_TIME = ZoneInfo("Europe/London")

#: If the nearest sea grid cell is further than this from the place, the place
#: is not really on the coast, and a tide "there" would be made up.
MAX_SEA_DISTANCE_KM = 25.0

#: Three requests share a 20 second budget.
REQUEST_TIMEOUT = 5.0


class NoPlaceGiven(AgentAppError):
    user_message = "Which UK beach or seaside town should I check the tide for?"
    retryable = False


class PlaceNotFound(AgentAppError):
    user_message = "I could not find that place in the UK."
    retryable = False


class NotCoastal(AgentAppError):
    user_message = "That place does not seem to be on the coast, so it has no tide."
    retryable = False


class NoTideFound(AgentAppError):
    user_message = "The tide forecast for that place had no high tide in it."
    retryable = True


class App:
    def __init__(self) -> None:
        self._manifest = validate_manifest(load_manifest_file(MANIFEST_PATH))

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    # ── The one door to the network ──────────────────────────────────────────

    def _fetch(self, url: str, timeout: float = REQUEST_TIMEOUT) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def _get_json(self, url: str, what: str):
        try:
            raw = self._fetch(url)
        except urllib.error.HTTPError as exc:  # must come before URLError
            if exc.code == 429:
                retry = exc.headers.get("Retry-After") if exc.headers else None
                seconds = int(retry) if retry and retry.isdigit() else None
                raise RateLimitedError(f"{what} rate limited", retry_after_seconds=seconds)
            raise ServiceUnavailableError(f"{what} returned {exc.code}")
        except TimeoutError as exc:  # not a URLError
            raise ServiceUnavailableError(f"{what} timed out: {exc}")
        except (urllib.error.URLError, OSError) as exc:
            raise ServiceUnavailableError(f"could not reach {what}: {exc}")
        try:
            return json.loads(raw)
        except ValueError:
            raise ServiceUnavailableError(f"{what} sent something that is not JSON")

    # ── The request ──────────────────────────────────────────────────────────

    def handle(self, call: Call) -> Result:
        if call.outcome_id != "tide.next_high":
            raise AgentAppError(f"this app cannot do {call.outcome_id!r}")

        place = (call.inputs.get("place") or "").strip()
        if not place:
            raise NoPlaceGiven("no place in the message")

        name, lat, lon = self._geocode(place)
        times, levels, sea_lat, sea_lon = self._sea_levels(lat, lon, place)

        if sea_lat is not None and sea_lon is not None:
            if _distance_km(lat, lon, sea_lat, sea_lon) > MAX_SEA_DISTANCE_KM:
                error = NotCoastal(f"{name} is far from the nearest sea cell")
                error.user_message = (
                    f"{name} does not seem to be on the coast, so it has no tide."
                )
                raise error

        now = call.context.now.astimezone(timezone.utc)
        peaks = [p for p in _high_tides(times, levels) if p[0] > now]
        if not peaks:
            raise NoTideFound(f"no peak after {now.isoformat()} for {name}")

        peak_time, peak_height = peaks[0]
        local = peak_time.astimezone(UK_TIME)
        minutes = max(0, int((peak_time - now).total_seconds() // 60))
        clock = f"{local:%a} {local.day} {local:%b}, {local:%H:%M}"

        data = {
            "place_used": name,
            "high_tide_time": local.isoformat(timespec="seconds"),
            "high_tide_clock": clock,
            "minutes_until": minutes,
            "height_m": round(float(peak_height), 2),
            "tide_rising": _rising_at(times, levels, now),
        }
        if len(peaks) > 1:
            data["following_high_time"] = (
                peaks[1][0].astimezone(UK_TIME).isoformat(timespec="seconds")
            )

        return Result(
            data=data,
            narrative=f"The next high tide at {name} is at {local:%H:%M} on "
            f"{local:%A} {local.day} {local:%B} ({_in_words(minutes)}).",
        )

    # ── Step 1: where is it? ─────────────────────────────────────────────────

    def _geocode(self, place: str) -> tuple[str, float, float]:
        url = OPEN_METEO_GEOCODE + "?" + urlencode(
            {"name": place, "countryCode": "GB", "count": 1, "language": "en"}
        )
        data = self._get_json(url, "the Open-Meteo geocoder")
        results = data.get("results") if isinstance(data, dict) else None
        if results:  # a place that does not exist has no "results" key at all
            top = results[0]
            name = top.get("name") or place
            area = top.get("admin2") or top.get("admin1")
            label = f"{name}, {area}" if area and area != name else name
            return label, float(top["latitude"]), float(top["longitude"])

        # Beaches and bays ("Camber Sands", "Woolacombe Bay") are often only
        # in OpenStreetMap.
        url = NOMINATIM_SEARCH + "?" + urlencode(
            {"q": place, "countrycodes": "gb", "format": "jsonv2", "limit": 1}
        )
        data = self._get_json(url, "Nominatim")
        if not data:  # 200 with []
            error = PlaceNotFound(f"no UK match for {place!r}")
            error.user_message = f"I could not find anywhere in the UK called {place}."
            raise error
        top = data[0]
        label = top.get("name") or top.get("display_name", place).split(",")[0]
        return label, float(top["lat"]), float(top["lon"])

    # ── Step 2: the sea level there ──────────────────────────────────────────

    def _sea_levels(self, lat: float, lon: float, place: str):
        url = OPEN_METEO_MARINE + "?" + urlencode(
            {
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "hourly": "sea_level_height_msl",
                "timeformat": "unixtime",
                "timezone": "GMT",
                "past_days": 1,
                "forecast_days": 3,
                "cell_selection": "sea",
            }
        )
        data = self._get_json(url, "the Open-Meteo marine forecast")
        hourly = (data or {}).get("hourly") or {}
        stamps = hourly.get("time") or []
        levels = hourly.get("sea_level_height_msl") or []
        if not stamps or sum(v is not None for v in levels) < 3:
            error = NotCoastal(f"no sea level data near {place!r}")
            error.user_message = (
                f"{place} does not seem to be on the coast, so it has no tide."
            )
            raise error
        times = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in stamps]
        return times, levels, data.get("latitude"), data.get("longitude")


# ── Pure helpers (no network, easy to test) ──────────────────────────────────


def _high_tides(times, levels):
    """Every local maximum in an hourly series, as (time, height)."""
    return _turning_points(times, levels, highs=True)


def _low_tides(times, levels):
    """Every local minimum in an hourly series, as (time, height)."""
    return _turning_points(times, levels, highs=False)


def _turning_points(times, levels, highs: bool):
    """Peaks (or troughs), refined between the hourly samples.

    A parabola through the turning sample and its two neighbours gives the
    time and height of the turn to within a minute or two.
    """
    points = []
    for i in range(1, len(levels) - 1):
        a, b, c = levels[i - 1], levels[i], levels[i + 1]
        if a is None or b is None or c is None:
            continue
        if highs and not (a < b >= c):
            continue
        if not highs and not (a > b <= c):
            continue
        denom = a - 2 * b + c
        offset = 0.0 if denom == 0 else 0.5 * (a - c) / denom
        offset = max(-0.5, min(0.5, offset))
        height = b - 0.25 * (a - c) * offset
        when = times[i] + timedelta(hours=offset)
        when = (when + timedelta(seconds=30)).replace(second=0, microsecond=0)
        points.append((when, height))
    return points


def _rising_at(times, levels, now) -> bool:
    """Is the tide coming in at `now`? Yes if the next turn is a high water.

    Comparing the two hourly samples either side of `now` is wrong for up to
    half an hour around every turn, so this looks at the turns instead.
    """
    next_high = next((t for t, _ in _high_tides(times, levels) if t > now), None)
    next_low = next((t for t, _ in _low_tides(times, levels) if t > now), None)
    if next_high is None:
        return False
    if next_low is None:
        return True
    return next_high < next_low


def _distance_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _in_words(minutes: int) -> str:
    if minutes < 1:
        return "about now"
    hours, mins = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if mins:
        parts.append(f"{mins} minute{'s' if mins != 1 else ''}")
    return "in " + " and ".join(parts)

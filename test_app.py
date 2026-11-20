"""Tests for UK Tide Times. The build runs these; a failing test stops the deploy.

Everything runs through `LocalRunner`, the same code the production runtime
uses, so a Result that breaks the manifest fails here rather than after
deploying. The network is replaced with canned responses: a made-up tide
with a known high water, so the expected answers never change.
"""

from __future__ import annotations

import json
import math
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest
from pippa_agentapp_sdk import (
    LocalRunner,
    OutputShapeViolation,
    RateLimitedError,
    ServiceUnavailableError,
)

from app import App, NoPlaceGiven, NotCoastal, PlaceNotFound

# 14:00 UTC on Friday 20 November 2026 (GMT, so UK time is the same).
NOW = datetime(2026, 11, 20, 14, 0, tzinfo=timezone.utc)
# The high water the fake tide is built around.
PEAK = datetime(2026, 11, 20, 17, 42, tzinfo=timezone.utc)
PERIOD_HOURS = 12.42  # the main lunar tide

BRIGHTON = {
    "results": [
        {"name": "Brighton", "admin2": "Brighton and Hove",
         "latitude": 50.8284, "longitude": -0.1395}
    ]
}


def tide(peak=PEAK, start=NOW - timedelta(hours=24), hours=96, amplitude=2.1):
    """An hourly sea level series shaped like a real semidiurnal tide."""
    times = [start + timedelta(hours=i) for i in range(hours)]
    levels = [
        round(amplitude * math.cos(
            2 * math.pi * ((t - peak).total_seconds() / 3600) / PERIOD_HOURS), 3)
        for t in times
    ]
    return {
        "latitude": 50.79, "longitude": -0.125,
        "hourly": {"time": [int(t.timestamp()) for t in times],
                   "sea_level_height_msl": levels},
    }


def make_fetch(geocode=BRIGHTON, nominatim=None, marine=None, log=None):
    marine = tide() if marine is None else marine

    def fetch(url, timeout=5.0):
        if log is not None:
            log.append(url)
        if "geocoding-api.open-meteo.com" in url:
            return json.dumps(geocode).encode()
        if "nominatim.openstreetmap.org" in url:
            return json.dumps(nominatim or []).encode()
        if "marine-api.open-meteo.com" in url:
            return json.dumps(marine).encode()
        raise AssertionError(f"unexpected request to {url}")

    return fetch


def call(inputs, fetch=None, *, now=NOW, explicit=False):
    app = App()
    app._fetch = fetch or make_fetch()
    return LocalRunner(app).call(
        "tide.next_high", inputs, granted_secrets={}, now=now,
        request_id="test-1", explicit=explicit,
    ).result


def minutes_apart(iso: str, expected: datetime) -> float:
    return abs((datetime.fromisoformat(iso) - expected).total_seconds()) / 60


def test_the_manifest_is_valid():
    app = App()
    assert app.manifest.app_id == "hack.app_44ab"
    assert [o.id for o in app.manifest.outcomes] == ["tide.next_high"]


def test_it_finds_the_next_high_tide():
    result = call({"place": "Brighton"})
    data = result.data
    assert data["place_used"] == "Brighton, Brighton and Hove"
    assert minutes_apart(data["high_tide_time"], PEAK) <= 3
    assert data["high_tide_clock"] == "Fri 20 Nov, 17:42"
    assert 219 <= data["minutes_until"] <= 225
    assert data["height_m"] == pytest.approx(2.1, abs=0.05)
    assert data["tide_rising"] is True
    assert minutes_apart(
        data["following_high_time"], PEAK + timedelta(hours=PERIOD_HOURS)) <= 3
    assert "17:42" in result.narrative


def test_times_carry_a_uk_offset_in_summer():
    """In July UK time is BST, an hour ahead of UTC."""
    july_now = datetime(2027, 7, 1, 9, 0, tzinfo=timezone.utc)
    july_peak = datetime(2027, 7, 1, 12, 15, tzinfo=timezone.utc)
    fetch = make_fetch(marine=tide(peak=july_peak, start=july_now - timedelta(hours=24)))
    data = call({"place": "Brighton"}, fetch, now=july_now).data
    assert data["high_tide_time"].endswith("+01:00")
    assert data["high_tide_clock"] == "Thu 1 Jul, 13:15"


def test_a_high_tide_already_gone_is_skipped():
    """Just after high water, the answer is the next one, about 12h25m later."""
    later = PEAK + timedelta(minutes=10)
    data = call({"place": "Brighton"}, now=later).data
    assert minutes_apart(
        data["high_tide_time"], PEAK + timedelta(hours=PERIOD_HOURS)) <= 3
    assert data["tide_rising"] is False


def test_a_beach_missing_from_the_town_list_falls_back_to_nominatim():
    log = []
    sea = tide()
    sea["latitude"], sea["longitude"] = 50.92, 0.79  # just off Camber
    fetch = make_fetch(
        geocode={"generationtime_ms": 0.3},  # no "results" key: nothing found
        marine=sea,
        nominatim=[{"name": "Camber Sands", "lat": "50.934", "lon": "0.795",
                    "display_name": "Camber Sands, Rother, East Sussex"}],
        log=log,
    )
    data = call({"place": "Camber Sands"}, fetch).data
    assert data["place_used"] == "Camber Sands"
    assert any("nominatim" in u and "countrycodes=gb" in u for u in log)


def test_place_names_with_spaces_are_encoded():
    log = []
    call({"place": "St Ives, Cornwall"}, make_fetch(log=log))
    assert "St+Ives%2C+Cornwall" in log[0]


def test_no_place_asks_for_one():
    with pytest.raises(NoPlaceGiven) as failure:
        call({"place": ""})
    assert "Which UK beach" in failure.value.user_message


def test_an_unknown_place_says_so_by_name():
    fetch = make_fetch(geocode={}, nominatim=[])
    with pytest.raises(PlaceNotFound) as failure:
        call({"place": "Narnia"}, fetch)
    assert failure.value.user_message == "I could not find anywhere in the UK called Narnia."


def test_an_inland_place_has_no_tide():
    empty = {"latitude": 52.48, "longitude": -1.9,
             "hourly": {"time": [], "sea_level_height_msl": []}}
    birmingham = {"results": [{"name": "Birmingham", "latitude": 52.48,
                               "longitude": -1.89}]}
    with pytest.raises(NotCoastal) as failure:
        call({"place": "Birmingham"}, make_fetch(geocode=birmingham, marine=empty))
    assert "not seem to be on the coast" in failure.value.user_message


def test_a_place_far_from_the_nearest_sea_cell_has_no_tide():
    far = tide()
    far["latitude"], far["longitude"] = 51.5, -3.2  # ~150 km away
    with pytest.raises(NotCoastal):
        call({"place": "Brighton"}, make_fetch(marine=far))


def test_a_busy_service_is_retryable():
    def busy(url, timeout=5.0):
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)

    with pytest.raises(RateLimitedError):
        call({"place": "Brighton"}, busy)


def test_a_timeout_is_reported_as_unavailable():
    def slow(url, timeout=5.0):
        raise TimeoutError("timed out")

    with pytest.raises(ServiceUnavailableError):
        call({"place": "Brighton"}, slow)


def test_picked_by_name_it_answers_and_refuses_as_usual():
    picked = call({"place": "Brighton"}, explicit=True)
    assert picked.data == call({"place": "Brighton"}).data
    with pytest.raises(NoPlaceGiven):
        call({"place": ""}, explicit=True)


def test_the_runner_enforces_the_manifest(monkeypatch):
    from pippa_agentapp_sdk import Result

    monkeypatch.setattr(App, "handle", lambda self, c: Result(data={"surprise": 1}))
    with pytest.raises(OutputShapeViolation):
        call({"place": "Brighton"})

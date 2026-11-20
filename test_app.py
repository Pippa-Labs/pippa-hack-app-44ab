"""Tests for London Side Quests. The build runs these; a failing test stops the deploy.

Everything goes through `LocalRunner`, the same code the runtime uses, so a
Result that breaks the manifest fails here. The network is replaced with
canned responses shaped like the real APIs, so nothing here goes online.
"""

from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest
from pippa_agentapp_sdk import (
    LocalRunner,
    OutputShapeViolation,
    RateLimitedError,
    ServiceUnavailableError,
)

from app import (
    App,
    NoPlaceGiven,
    NothingNearby,
    PlaceNotFound,
    Spot,
    _events_in_window_first,
    _first_sentence,
    _open_now,
)

# 14:00 on Friday 20 November 2026 (GMT, so UK time is the same as UTC).
NOW = datetime(2026, 11, 20, 14, 0, tzinfo=timezone.utc)
FAKE_KEY = "test-not-a-real-key-123"  # pragma: allowlist secret
FAKE_SKIDDLE_KEY = "test-not-a-real-skiddle-456"  # pragma: allowlist secret

# Around Tower Hill. Roughly 111 m per 0.001 deg of latitude.
ORIGIN = {"lat": "51.5100", "lon": "-0.0760", "name": "Tower Hill"}


def node(i, lat, lon, **tags):
    return {"type": "node", "id": i, "lat": lat, "lon": lon, "tags": tags}


OVERPASS_ELEMENTS = [
    node(1, 51.5110, -0.0760, amenity="cafe", name="Rooftop Beans"),
    node(2, 51.5120, -0.0760, amenity="restaurant", name="The Hungry Knight"),
    node(3, 51.5103, -0.0770, historic="memorial", name="Plague Pit Stone",
         start_date="1665"),
    node(4, 51.5130, -0.0750, tourism="viewpoint", name="Secret Wall Lookout"),
    {"type": "way", "id": 5, "center": {"lat": 51.5081, "lon": -0.0759},
     "tags": {"historic": "castle", "name": "Tower of London",
              "wikipedia": "en:Tower of London", "wikidata": "Q62378",
              "start_date": "1078"}},
    node(6, 51.5140, -0.0790, tourism="museum", name="Museum of Odd Keys"),
    node(7, 51.5105, -0.0745, amenity="cafe", name="Ravens Rest"),
    node(8, 51.5108, -0.0765, historic="wall", name="Roman Wall Fragment"),
    node(9, 51.5101, -0.0761, amenity="cafe", name=""),  # no name: ignored
    # A named bench: tagged historic, but not a quest.
    node(10, 51.5101, -0.0759, historic="memorial", memorial="bench",
         name="In Memory of Ada"),
    # A blue plaque: stays.
    node(11, 51.5109, -0.0771, historic="memorial", memorial="plaque",
         name="Pepys Plaque"),
    # Closed at 14:00 on a Friday.
    node(12, 51.5102, -0.0762, amenity="cafe", name="Breakfast Only",
         opening_hours="Mo-Fr 07:00-11:00; Sa,Su 08:00-12:00"),
    # Open now.
    node(13, 51.5115, -0.0765, amenity="cafe", name="All Day Brew",
         opening_hours="Mo-Su 08:00-18:00"),
]

TICKETMASTER = {
    "_embedded": {"events": [
        {"name": "Ghost Walk of the City",
         "dates": {"start": {"dateTime": "2026-11-21T19:00:00Z"},
                   "end": {"dateTime": "2026-11-21T21:00:00Z"},
                   "status": {"code": "onsale"}},
         "_embedded": {"venues": [{"name": "Tower Hill Steps",
                                   "location": {"latitude": "51.5098",
                                                "longitude": "-0.0765"}}]}},
        {"name": "Cancelled Gig",
         "dates": {"start": {"dateTime": "2026-11-22T19:00:00Z"},
                   "status": {"code": "cancelled"}}},
    ]}
}

# Skiddle does not document its reply; this is the shape its API returns.
SKIDDLE = {
    "error": 0,
    "totalcount": "3",
    "results": [
        {"id": "1", "eventname": "Basement Disco Night",
         "venue": {"name": "The Crypt Club", "latitude": 51.5112,
                   "longitude": -0.0772},
         "date": "2026-11-20",
         "startdate": "2026-11-20T22:00:00+00:00",
         "enddate": "2026-11-21T03:00:00+00:00", "cancelled": "0"},
        {"id": "2", "eventname": "The Ghost Walk of the City!",  # also on Ticketmaster
         "venue": {"name": "Tower Hill Steps", "latitude": 51.5098,
                   "longitude": -0.0765},
         "date": "2026-11-21",
         "startdate": "2026-11-21T19:00:00+00:00", "cancelled": "0"},
        {"id": "3", "eventname": "Called-Off Comedy",
         "venue": {"name": "Somewhere"}, "date": "2026-11-22",
         "cancelled": "1"},
    ],
}

WIKI = {"extract": "The Tower of London is a historic castle on the north bank "
                   "of the River Thames. It was founded in 1066."}


def make_fetch(geocode=None, overpass=None, events=None, wiki=None, log=None,
               fail_events=None, skiddle=None, fail_skiddle=None):
    def fetch(url, timeout=6.0):
        if log is not None:
            log.append(url)
        if "nominatim.openstreetmap.org" in url:
            return json.dumps([ORIGIN] if geocode is None else geocode).encode()
        if "overpass-api.de" in url:
            return json.dumps({"elements": OVERPASS_ELEMENTS if overpass is None
                               else overpass}).encode()
        if "app.ticketmaster.com" in url:
            if fail_events is not None:
                raise fail_events
            return json.dumps(TICKETMASTER if events is None else events).encode()
        if "www.skiddle.com" in url:
            if fail_skiddle is not None:
                raise fail_skiddle
            return json.dumps(SKIDDLE if skiddle is None else skiddle).encode()
        if "en.wikipedia.org" in url:
            return json.dumps(WIKI if wiki is None else wiki).encode()
        raise AssertionError("unexpected request")

    return fetch


def call(inputs, fetch=None, *, explicit=False):
    app = App()
    app._fetch = fetch or make_fetch()
    return LocalRunner(app).call(
        "quests.board", inputs, granted_secrets={}, now=NOW,
        request_id="test-1", explicit=explicit,
    ).result


@pytest.fixture
def key(monkeypatch):
    """Ticketmaster only."""
    monkeypatch.setenv("TICKETMASTER_KEY", FAKE_KEY)
    monkeypatch.delenv("SKIDDLE_KEY", raising=False)


@pytest.fixture
def skiddle_key(monkeypatch):
    """Skiddle only."""
    monkeypatch.delenv("TICKETMASTER_KEY", raising=False)
    monkeypatch.setenv("SKIDDLE_KEY", FAKE_SKIDDLE_KEY)


@pytest.fixture
def both_keys(monkeypatch):
    monkeypatch.setenv("TICKETMASTER_KEY", FAKE_KEY)
    monkeypatch.setenv("SKIDDLE_KEY", FAKE_SKIDDLE_KEY)


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("TICKETMASTER_KEY", raising=False)
    monkeypatch.delenv("SKIDDLE_KEY", raising=False)


def test_the_manifest_is_valid():
    app = App()
    assert app.manifest.app_id == "hack.app_44ab"
    assert [o.id for o in app.manifest.outcomes] == ["quests.board"]


def test_a_full_quest_board(key):
    result = call({"place": "Tower Hill"})
    data = result.data
    assert data["place_used"] == "Tower Hill"
    assert 4 <= data["quest_count"] <= 6 == len(data["quests"])
    board = "\n".join(data["quests"])
    for tier in ("Common", "Rare", "Legendary", "Limited-time"):
        assert tier in board
    assert data["events_included"] is True
    assert data["total_xp"] > 0
    assert all(" m | " in q and " XP | " in q for q in data["quests"])


def test_legendary_quests_carry_lore(key):
    quests = call({"place": "Tower Hill"}).data["quests"]
    tower = next(q for q in quests if "Tower of London" in q)
    assert tower.startswith("🌟 Legendary")
    assert "Lore: The Tower of London is a historic castle" in tower
    assert "1066" not in tower  # one sentence only


def test_lore_falls_back_to_map_tags_when_wikipedia_fails(key):
    def fetch(url, timeout=6.0):
        if "wikipedia" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return make_fetch()(url, timeout)

    quests = call({"place": "Tower Hill"}, fetch).data["quests"]
    tower = next(q for q in quests if "Tower of London" in q)
    assert "Lore: It has stood here since 1078." in tower
    # No Wikipedia and no facts on the map: no lore, rather than made-up lore.
    museum = next(q for q in quests if "Museum of Odd Keys" in q)
    assert "Lore:" not in museum


def test_events_say_when_they_end_and_skip_cancelled(key):
    quests = call({"place": "Tower Hill"}).data["quests"]
    event = next(q for q in quests if q.startswith("⚡"))
    assert "Ghost Walk of the City" in event
    assert "ends Sat 21 Nov, 21:00" in event
    assert not any("Cancelled Gig" in q for q in quests)


def test_lesser_known_spots_are_rare(key):
    quests = call({"place": "Tower Hill"}).data["quests"]
    rare = [q for q in quests if q.startswith("🔷 Rare")]
    assert rare
    assert not any("Tower of London" in q for q in rare)


def test_a_quest_chain_of_three_places(key):
    data = call({"place": "Tower Hill"}).data
    chain = data["quest_chain"]
    assert len(chain) == 3
    assert chain[0].startswith("1.") and chain[2].startswith("3.")
    assert "m from you" in chain[0] and "m on" in chain[1]
    assert data["chain_bonus_xp"] >= 150


def test_a_broken_events_api_still_returns_the_board(key):
    fetch = make_fetch(fail_events=urllib.error.HTTPError(
        "https://app.ticketmaster.com/x", 401, "Unauthorized", {}, None))
    result = call({"place": "Tower Hill"}, fetch)
    assert result.data["events_included"] is False
    assert result.data["quest_count"] >= 4
    assert not any(q.startswith("⚡") for q in result.data["quests"])
    assert any("Live events could not be loaded" in w for w in result.warnings)


def test_a_timed_out_events_api_still_returns_the_board(key):
    result = call({"place": "Tower Hill"}, make_fetch(fail_events=TimeoutError()))
    assert result.data["events_included"] is False
    assert result.data["quest_count"] >= 4


def test_without_a_key_events_are_skipped_and_never_requested(no_key):
    log = []
    result = call({"place": "Tower Hill"}, make_fetch(log=log))
    assert result.data["events_included"] is False
    assert not any("ticketmaster" in u or "skiddle" in u for u in log)
    assert any("No events key" in w for w in result.warnings)


def test_skiddle_events_become_limited_time_quests(skiddle_key):
    result = call({"place": "Tower Hill"})
    assert result.data["events_included"] is True
    events = [q for q in result.data["quests"] if q.startswith("⚡")]
    assert events
    disco = next(q for q in events if "Basement Disco Night" in q)
    assert "At The Crypt Club" in disco and "ends Sat 21 Nov, 03:00" in disco
    assert not any("Called-Off Comedy" in q for q in result.data["quests"])


def test_skiddle_is_asked_for_the_right_area_and_week(skiddle_key):
    log = []
    call({"place": "Tower Hill"}, make_fetch(log=log))
    url = next(u for u in log if "skiddle" in u)
    for part in ("latitude=51.51000", "longitude=-0.07600", "radius=1",
                 "minDate=2026-11-20", "maxDate=2026-11-27", "country=GB"):
        assert part in url


def test_an_event_on_both_services_appears_once(both_keys):
    from app import App as _App
    app = _App()
    app._fetch = make_fetch()
    events = app._ticketmaster_events(51.51, -0.076, NOW, FAKE_KEY) + \
        app._skiddle_events(51.51, -0.076, NOW, FAKE_SKIDDLE_KEY)
    from app import _merge_events
    names = [e.name for e in _merge_events(events)]
    assert len([n for n in names if "Ghost Walk" in n]) == 1
    assert names[0] == "Basement Disco Night"  # soonest first


def test_one_events_service_failing_keeps_the_other(both_keys):
    result = call({"place": "Tower Hill"}, make_fetch(fail_events=TimeoutError()))
    assert result.data["events_included"] is True
    assert any("Basement Disco Night" in q for q in result.data["quests"])
    assert any("Events from Ticketmaster could not be loaded" in w
               for w in result.warnings)


def test_both_events_services_failing_still_returns_the_board(both_keys):
    result = call({"place": "Tower Hill"}, make_fetch(
        fail_events=TimeoutError(), fail_skiddle=TimeoutError()))
    assert result.data["events_included"] is False
    assert result.data["quest_count"] >= 4
    assert any("Live events could not be loaded" in w for w in result.warnings)


def test_a_skiddle_error_reply_is_a_failure_not_an_empty_list(skiddle_key):
    """A bad key comes back as 200 with an error flag, not as an HTTP error."""
    bad = {"error": 1, "errormessage": f"Invalid api_key {FAKE_SKIDDLE_KEY}"}
    result = call({"place": "Tower Hill"}, make_fetch(skiddle=bad))
    assert result.data["events_included"] is False
    everything = json.dumps(result.data) + " ".join(result.warnings or ())
    assert FAKE_SKIDDLE_KEY not in everything


def test_the_skiddle_key_goes_only_to_skiddle(both_keys):
    log = []
    call({"place": "Tower Hill"}, make_fetch(log=log))
    with_key = [u for u in log if FAKE_SKIDDLE_KEY in u]
    assert with_key and all(u.startswith("https://www.skiddle.com/") for u in with_key)


def test_the_key_never_appears_in_what_is_returned(key):
    fetch = make_fetch(fail_events=urllib.error.HTTPError(
        "https://app.ticketmaster.com/x", 500, "Server Error", {}, None))
    result = call({"place": "Tower Hill"}, fetch)
    everything = json.dumps(result.data) + (result.narrative or "") + \
        " ".join(result.warnings or ())
    assert FAKE_KEY not in everything


def test_the_key_goes_only_to_ticketmaster(key):
    log = []
    call({"place": "Tower Hill"}, make_fetch(log=log))
    with_key = [u for u in log if FAKE_KEY in u]
    assert with_key and all(u.startswith("https://app.ticketmaster.com/") for u in with_key)


def test_little_time_keeps_quests_close_and_short(key):
    """30 minutes: no 45-minute Legendary visit, and only nearby places."""
    data = call({"place": "Tower Hill", "minutes_available": 30}).data
    places = [q for q in data["quests"] if not q.startswith("⚡")]
    assert places
    assert not any(q.startswith("🌟") for q in places)
    for line in places:
        metres = int(line.split(" | ")[2].lstrip("~").split()[0])
        assert metres <= 600


def test_place_names_are_encoded_and_kept_in_london(key):
    log = []
    call({"place": "King's Cross"}, make_fetch(log=log))
    assert "King%27s+Cross" in log[0] and "bounded=1" in log[0]


def test_no_place_asks_for_one():
    with pytest.raises(NoPlaceGiven):
        call({"place": ""})


def test_an_unknown_place_says_so_by_name(no_key):
    with pytest.raises(PlaceNotFound) as failure:
        call({"place": "Narnia"}, make_fetch(geocode=[]))
    assert failure.value.user_message == "I could not find anywhere in London called Narnia."


def test_nothing_nearby_is_an_honest_error(no_key):
    with pytest.raises(NothingNearby):
        call({"place": "Tower Hill"}, make_fetch(overpass=[]))


def test_a_busy_map_service_is_retryable(no_key):
    def fetch(url, timeout=6.0):
        if "overpass" in url:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        return make_fetch()(url, timeout)

    with pytest.raises(RateLimitedError):
        call({"place": "Tower Hill"}, fetch)


def test_an_unreachable_map_service_is_reported(no_key):
    def fetch(url, timeout=6.0):
        if "overpass" in url:
            raise TimeoutError("timed out")
        return make_fetch()(url, timeout)

    with pytest.raises(ServiceUnavailableError):
        call({"place": "Tower Hill"}, fetch)


def test_first_sentence_survives_abbreviations():
    text = "St. Paul's Cathedral is an Anglican cathedral in London. It is big."
    assert _first_sentence(text) == "St. Paul's Cathedral is an Anglican cathedral in London."


def test_picked_by_name_it_answers_and_refuses_as_usual(no_key):
    picked = call({"place": "Tower Hill"}, explicit=True)
    assert picked.data == call({"place": "Tower Hill"}).data
    with pytest.raises(NoPlaceGiven):
        call({"place": ""}, explicit=True)


def test_the_runner_enforces_the_manifest(monkeypatch):
    from pippa_agentapp_sdk import Result

    monkeypatch.setattr(App, "handle", lambda self, c: Result(data={"surprise": 1}))
    with pytest.raises(OutputShapeViolation):
        call({"place": "Tower Hill"})



# ── 0.2.1: the review fixes ──────────────────────────────────────────────────


def everything_returned(no_key_result):
    data = no_key_result.data
    return "\n".join(data["quests"] + data.get("quest_chain", []))


def test_cafes_are_searched_closer_in_than_sights(no_key):
    log = []
    call({"place": "Tower Hill"}, make_fetch(log=log))
    query = next(u for u in log if "overpass" in u)
    assert "around%3A450%2C" in query      # cafés and restaurants
    assert "around%3A1000%2C" in query     # landmarks, museums, hidden spots


def test_memorial_benches_are_not_quests_but_plaques_are(no_key):
    text = everything_returned(call({"place": "Tower Hill"}))
    assert "In Memory of Ada" not in text
    app = App()
    app._fetch = make_fetch()
    names = [s.name for s in app._places(51.51, -0.076, 1000)]
    assert "Pepys Plaque" in names and "In Memory of Ada" not in names


def test_places_closed_right_now_are_left_out(no_key):
    text = everything_returned(call({"place": "Tower Hill"}))
    assert "Breakfast Only" not in text


@pytest.mark.parametrize("spec, when, expected", [
    ("Mo-Fr 09:00-17:30", datetime(2026, 11, 20, 14, 0), True),    # Friday
    ("Mo-Fr 09:00-17:30", datetime(2026, 11, 21, 14, 0), False),   # Saturday
    ("Mo-Fr 09:00-17:30; Sa 10:00-16:00", datetime(2026, 11, 21, 14, 0), True),
    ("Mo-Su 08:00-12:00,13:00-18:00", datetime(2026, 11, 20, 12, 30), False),
    ("Fr,Sa 18:00-02:00", datetime(2026, 11, 21, 1, 0), True),     # Fri night
    ("Fr,Sa 18:00-02:00", datetime(2026, 11, 20, 14, 0), False),
    ("Mo-Su 10:00-22:00; We off", datetime(2026, 11, 18, 14, 0), False),
    ("10:00-18:00", datetime(2026, 11, 22, 11, 0), True),
    ("24/7", datetime(2026, 11, 22, 3, 0), True),
    ("Mo-Fr 09:00-17:00; PH off", datetime(2026, 11, 20, 10, 0), True),
    ("sunrise-sunset", datetime(2026, 11, 20, 10, 0), None),       # too fancy
    ("Jan-Mar Mo-Fr 10:00-16:00", datetime(2026, 11, 20, 10, 0), None),
    ("", datetime(2026, 11, 20, 10, 0), None),
    (None, datetime(2026, 11, 20, 10, 0), None),
])
def test_opening_hours(spec, when, expected):
    assert _open_now(spec, when) is expected


def test_a_busy_main_overpass_server_falls_back_to_the_mirror(no_key):
    log = []

    def fetch(url, timeout=6.0):
        log.append(url)
        if "overpass-api.de" in url:
            raise urllib.error.HTTPError(url, 504, "Gateway Timeout", {}, None)
        if "overpass.private.coffee" in url:
            return json.dumps({"elements": OVERPASS_ELEMENTS}).encode()
        return make_fetch()(url, timeout)

    result = call({"place": "Tower Hill"}, fetch)
    assert result.data["quest_count"] >= 4
    assert any("overpass.private.coffee" in u for u in log)


def test_both_overpass_servers_down_is_reported(no_key):
    def fetch(url, timeout=6.0):
        if "overpass" in url:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        return make_fetch()(url, timeout)

    with pytest.raises(RateLimitedError):
        call({"place": "Tower Hill"}, fetch)


def test_an_event_without_a_venue_location_says_nearby(key):
    no_location = {"_embedded": {"events": [
        {"name": "Mystery Pop-Up",
         "dates": {"start": {"dateTime": "2026-11-20T18:00:00Z"}},
         "_embedded": {"venues": [{"name": "Secret Venue"}]}},
    ]}}
    quests = call({"place": "Tower Hill"}, make_fetch(events=no_location)).data["quests"]
    event = next(q for q in quests if "Mystery Pop-Up" in q)
    assert " | nearby | " in event and " 0 m" not in event


def test_with_a_time_limit_events_happening_now_come_first():
    def event(name, start_h, end_h=None):
        start = NOW + timedelta(hours=start_h)
        end = NOW + timedelta(hours=end_h) if end_h is not None else None
        return Spot(name=name, kind="event", rarity="Limited", distance_m=100,
                    start=start, end=end)

    thursday = event("Thursday Gig", 72)
    tonight = event("Tonight Show", 1.5)
    on_now = event("Market On Now", -1, 2)
    ordered = _events_in_window_first([thursday, tonight, on_now], NOW, 120)
    assert [e.name for e in ordered[:2]] == ["Tonight Show", "Market On Now"]
    assert ordered[-1].name == "Thursday Gig"


def test_distances_are_marked_as_approximate(no_key):
    data = call({"place": "Tower Hill"}).data
    assert all(" | ~" in q for q in data["quests"])
    assert all("~" in step for step in data["quest_chain"])

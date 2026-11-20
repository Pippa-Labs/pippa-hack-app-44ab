"""London Side Quests: a quest board of things to do near a London location.

One call, four sources, no state:

1. Nominatim turns the place name into coordinates (kept inside London).
2. In parallel:
   - the OpenStreetMap Overpass API lists named cafés, restaurants,
     attractions, viewpoints, historic sites and museums within ~1 km;
   - the Ticketmaster Discovery API and the Skiddle API list events nearby in
     the next 7 days (each only if its key is set: TICKETMASTER_KEY,
     SKIDDLE_KEY). Ticketmaster is strongest on big concerts and shows,
     Skiddle on club nights, gigs, comedy and festivals. Events listed on
     both appear once. If one fails, the other still counts; if both fail,
     the board still comes back without Limited-time quests.
3. Wikipedia gives a one-line lore fact for the Legendary quests.

Places OpenStreetMap says are closed right now are left out, and so are
memorial benches and the like, which are tagged "historic" but are not quests.

Rarity tiers:
    Common      cafés and restaurants
    Rare        lesser-known spots: viewpoints, attractions and historic
                sites with no Wikipedia article
    Legendary   museums, and attractions or historic sites famous enough to
                have a Wikipedia article (with a lore fact)
    Limited     live events, with when they end
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
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

#: Nominatim, Overpass and Wikipedia refuse requests without a real User-Agent.
USER_AGENT = "pippa-agent-app-london-side-quests/0.2 (https://github.com/Pippa-Labs)"

NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
#: Public Overpass servers, tried in order. The main one is often busy at
#: peak times; private.coffee is a public mirror with the same data.
OVERPASS_SERVERS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
TICKETMASTER_EVENTS = "https://app.ticketmaster.com/discovery/v2/events.json"
SKIDDLE_EVENTS = "https://www.skiddle.com/api/v1/events/search/"
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

#: Greater London, as Nominatim's viewbox: west, north, east, south.
LONDON_VIEWBOX = "-0.5104,51.6919,0.3340,51.2868"

UK_TIME = ZoneInfo("Europe/London")

SEARCH_RADIUS_M = 1000
#: Cafés and restaurants are searched closer in. Central London has hundreds
#: within a kilometre, and Overpass cuts a long list in database order, not by
#: distance, so a wide search would drop the café next door.
FOOD_RADIUS_M = 450
EVENT_RADIUS_KM = 2
WALK_M_PER_MIN = 80

#: Timeouts, sharing one 20 second budget: geocode, then Overpass and
#: the two event services side by side, then Wikipedia lookups side by side.
GEOCODE_TIMEOUT = 5.0
OVERPASS_TIMEOUT = 7.0
#: Everything is planned to finish within this many seconds of the call
#: starting, leaving room under Pippa's 20 second limit.
BUDGET_SECONDS = 17.0
EVENTS_TIMEOUT = 6.0
LORE_TIMEOUT = 3.0

MAX_QUESTS = 6
COMMON, RARE, LEGENDARY, LIMITED = "Common", "Rare", "Legendary", "Limited"
BADGE = {COMMON: "⚪ Common", RARE: "🔷 Rare", LEGENDARY: "🌟 Legendary",
         LIMITED: "⚡ Limited-time"}
BASE_XP = {COMMON: 50, RARE: 120, LEGENDARY: 250, LIMITED: 200}
#: Minutes spent at a place, for fitting quests into the time available.
DWELL_MIN = {COMMON: 25, RARE: 15, LEGENDARY: 45}

#: Tagged "historic" in OpenStreetMap, but not somewhere to send anybody on a
#: quest: named benches, memorial trees, single graves and ghost bikes.
#: Statues, plaques, monuments and war memorials stay in.
NOT_QUESTS = frozenset({"bench", "tree", "grave", "ghost_bike", "tomb"})

FLAVOUR = {
    "cafe": ["Refuel your mana with a brew at {n}.",
             "Rest at the {n} tavern and restore 20 HP.",
             "Trade coins for a potion of caffeine at {n}."],
    "restaurant": ["Feast like a guild champion at {n}.",
                   "Claim a hearty meal buff at {n}.",
                   "Brave the menu at {n} and leave victorious."],
    "viewpoint": ["Climb to {n} and survey your realm.",
                  "Reveal more of the map from {n}."],
    "museum": ["Raid the relic vaults of {n}.",
               "Collect ancient artefacts (with your eyes) at {n}."],
    "historic": ["Uncover the secrets of {n}.",
                 "Find {n} before the townsfolk forget it.",
                 "Pay your respects at {n} and gain wisdom."],
    "attraction": ["Seek out {n} and claim the bragging rights.",
                   "Discover the wonder that is {n}."],
    "event": ["Catch {n} before the portal closes.",
              "A rare spawn: {n}, for a limited time only."],
}


class NoPlaceGiven(AgentAppError):
    user_message = "Where in London should I find side quests around?"
    retryable = False


class PlaceNotFound(AgentAppError):
    user_message = "I could not find that place in London."
    retryable = False


class NothingNearby(AgentAppError):
    user_message = "I could not find anything to explore within a kilometre of there."
    retryable = False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects, so an events key never follows one anywhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


@dataclass(eq=False)
class Spot:
    name: str
    kind: str  # cafe, restaurant, viewpoint, museum, historic, attraction, event
    rarity: str
    distance_m: int | None  # None when an event's venue has no location
    lat: float | None = None
    lon: float | None = None
    wikipedia: str = ""
    lore: str = ""
    when: str = ""  # events only: when it ends
    start: datetime | None = None  # events only, for sorting soonest first
    end: datetime | None = None  # events only
    extra: dict = field(default_factory=dict)

    @property
    def xp(self) -> int:
        # A little more for walking further.
        bonus = 0 if self.distance_m is None else 5 * round(self.distance_m / 50)
        return BASE_XP[self.rarity] + bonus


class App:
    def __init__(self) -> None:
        self._manifest = validate_manifest(load_manifest_file(MANIFEST_PATH))

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    # ── The one door to the network ──────────────────────────────────────────

    def _fetch(self, url: str, timeout: float = 6.0) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with _OPENER.open(request, timeout=timeout) as response:
            return response.read()

    def _get_json(self, url: str, what: str, timeout: float):
        """Fetch and parse, turning failures into the SDK's own errors.

        Never puts the URL or the response body in a message: the Ticketmaster
        and Skiddle URLs carry their keys.
        """
        try:
            raw = self._fetch(url, timeout=timeout)
        except urllib.error.HTTPError as exc:  # must come before URLError
            if exc.code == 429:
                raise RateLimitedError(f"{what} rate limited")
            raise ServiceUnavailableError(f"{what} returned HTTP {exc.code}")
        except TimeoutError:  # not a URLError
            raise ServiceUnavailableError(f"{what} timed out")
        except (urllib.error.URLError, OSError):
            raise ServiceUnavailableError(f"could not reach {what}")
        try:
            return json.loads(raw)
        except ValueError:
            raise ServiceUnavailableError(f"{what} sent something that is not JSON")

    # ── The request ──────────────────────────────────────────────────────────

    def handle(self, call: Call) -> Result:
        if call.outcome_id != "quests.board":
            raise AgentAppError(f"this app cannot do {call.outcome_id!r}")

        place = (call.inputs.get("place") or "").strip()
        if not place:
            raise NoPlaceGiven("no place in the message")
        minutes = call.inputs.get("minutes_available")
        if minutes is not None and minutes <= 0:
            minutes = None
        now = call.context.now.astimezone(timezone.utc)
        # A stopwatch, not the date: the wall-clock time always comes from
        # call.context.now.
        deadline = time.monotonic() + BUDGET_SECONDS

        name, lat, lon = self._geocode(place)
        radius = _radius_for(minutes)

        warnings: list[str] = []
        sources = {
            "Ticketmaster": (self._ticketmaster_events, os.environ.get("TICKETMASTER_KEY", "").strip()),
            "Skiddle": (self._skiddle_events, os.environ.get("SKIDDLE_KEY", "").strip()),
        }
        with ThreadPoolExecutor(max_workers=3) as pool:
            places_job = pool.submit(self._places, lat, lon, radius, deadline)
            event_jobs = {
                label: pool.submit(fetch, lat, lon, now, key)
                for label, (fetch, key) in sources.items() if key
            }
            found: list[Spot] = []
            failed: list[str] = []
            for label, job in event_jobs.items():
                try:
                    found += job.result()
                except Exception:  # noqa: BLE001 — events are optional by design
                    failed.append(label)
            spots = places_job.result()

        events_ok = len(failed) < len(event_jobs)
        events = _merge_events(found)
        if not event_jobs:
            warnings.append(
                "No events key is set up yet, so there are no Limited-time quests.")
        elif not events_ok:
            warnings.append(
                "Live events could not be loaded, so there are no "
                "Limited-time quests this time.")
        else:
            for label in failed:
                warnings.append(f"Events from {label} could not be loaded this time.")

        local_now = now.astimezone(UK_TIME)
        spots = [s for s in spots
                 if _open_now(s.extra.get("opening_hours"), local_now) is not False]
        if minutes is not None:
            spots = [s for s in spots if _fits(s, minutes)]
            events = _events_in_window_first(events, now, minutes)

        board = _pick_board(spots, events)
        if not board:
            error = NothingNearby(f"no quests within {radius} m of {name!r}")
            error.user_message = (
                f"I could not find anything to explore near {name}"
                + (f" in {minutes} minutes." if minutes else ".")
            )
            raise error

        chain = _pick_chain(spots, board)
        self._add_lore([s for s in board + chain if s.rarity == LEGENDARY][:4], deadline)

        data = {
            "place_used": name,
            "quests": [_quest_line(s) for s in board],
            "quest_count": len(board),
            "total_xp": sum(s.xp for s in board),
            "events_included": events_ok,
        }
        if len(chain) == 3:
            steps, walked = _chain_lines(chain)
            data["quest_chain"] = steps
            data["chain_bonus_xp"] = 150 + 5 * round(walked / 50)

        narrative = (
            f"London Side Quests near {name}: {len(board)} quests worth "
            f"{data['total_xp']} XP"
            + (", plus a 3-stop quest chain" if len(chain) == 3 else "")
            + "."
        )
        return Result(data=data, narrative=narrative, warnings=tuple(warnings))

    # ── Step 1: where? ───────────────────────────────────────────────────────

    def _geocode(self, place: str) -> tuple[str, float, float]:
        url = NOMINATIM_SEARCH + "?" + urlencode({
            "q": place, "format": "jsonv2", "limit": 1, "countrycodes": "gb",
            "viewbox": LONDON_VIEWBOX, "bounded": 1,
        })
        data = self._get_json(url, "Nominatim", GEOCODE_TIMEOUT)
        if not data:  # 200 with []
            error = PlaceNotFound(f"no London match for {place!r}")
            error.user_message = f"I could not find anywhere in London called {place}."
            raise error
        top = data[0]
        label = top.get("name") or top.get("display_name", place).split(",")[0]
        return label or place, float(top["lat"]), float(top["lon"])

    # ── Step 2a: places from OpenStreetMap ───────────────────────────────────

    def _places(self, lat: float, lon: float, radius: int,
                deadline: float | None = None) -> list[Spot]:
        around = f"(around:{radius},{lat:.6f},{lon:.6f})"
        food = f"(around:{min(radius, FOOD_RADIUS_M)},{lat:.6f},{lon:.6f})"
        query = (
            "[out:json][timeout:8];"
            f'(nwr{food}["amenity"~"^(cafe|restaurant)$"]["name"];)->.food;'
            f'(nwr{around}["tourism"~"^(attraction|viewpoint|museum)$"]["name"];'
            f'nwr{around}["historic"]["name"];)->.sights;'
            ".food out center tags 80;"
            ".sights out center tags 150;"
        )
        data = self._overpass(query, deadline)
        spots: list[Spot] = []
        seen: set[str] = set()
        for element in (data or {}).get("elements", []):
            spot = _spot_from_osm(element, lat, lon)
            if spot is None or spot.name.lower() in seen:
                continue
            seen.add(spot.name.lower())
            spots.append(spot)
        spots.sort(key=lambda s: s.distance_m)
        return spots

    def _overpass(self, query: str, deadline: float | None):
        """Ask the main Overpass server, and the mirror if it is busy or down.

        The mirror is only tried if there is time left for it.
        """
        last_error: AgentAppError | None = None
        for index, server in enumerate(OVERPASS_SERVERS):
            timeout = OVERPASS_TIMEOUT
            if index > 0:
                left = (deadline - time.monotonic() - 1.5) if deadline else OVERPASS_TIMEOUT
                if left < 2:
                    break
                timeout = min(OVERPASS_TIMEOUT, left)
            url = server + "?" + urlencode({"data": query})
            try:
                return self._get_json(url, "the Overpass API", timeout)
            except (ServiceUnavailableError, RateLimitedError) as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    # ── Step 2b: events from Ticketmaster ────────────────────────────────────

    def _ticketmaster_events(self, lat: float, lon: float, now: datetime, key: str) -> list[Spot]:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        url = TICKETMASTER_EVENTS + "?" + urlencode({
            "apikey": key,
            "latlong": f"{lat:.5f},{lon:.5f}",
            "radius": EVENT_RADIUS_KM,
            "unit": "km",
            "startDateTime": now.strftime(fmt),
            "endDateTime": (now + timedelta(days=7)).strftime(fmt),
            "sort": "date,asc",
            "size": 30,
        })
        data = self._get_json(url, "Ticketmaster", EVENTS_TIMEOUT)
        found = ((data or {}).get("_embedded") or {}).get("events") or []
        events: list[Spot] = []
        seen: set[str] = set()
        for item in found:
            spot = _spot_from_event(item, lat, lon, now)
            if spot is None or spot.name.lower() in seen:
                continue
            seen.add(spot.name.lower())
            events.append(spot)
        return events

    # ── Step 2c: events from Skiddle ─────────────────────────────────────────

    def _skiddle_events(self, lat: float, lon: float, now: datetime, key: str) -> list[Spot]:
        today = now.astimezone(UK_TIME).date()
        url = SKIDDLE_EVENTS + "?" + urlencode({
            "api_key": key,
            "latitude": f"{lat:.5f}",
            "longitude": f"{lon:.5f}",
            "radius": max(1, round(EVENT_RADIUS_KM / 1.609)),  # Skiddle uses miles
            "country": "GB",
            "minDate": today.isoformat(),
            "maxDate": (today + timedelta(days=7)).isoformat(),
            "order": "distance",
            "getdistance": 1,
            "limit": 30,
        })
        data = self._get_json(url, "Skiddle", EVENTS_TIMEOUT)
        if not isinstance(data, dict):
            raise ServiceUnavailableError("Skiddle sent an unexpected reply")
        if str(data.get("error", "0")) not in ("0", "", "False", "false"):
            # A bad key arrives as a 200 with an error flag. Its message is
            # never passed on: providers sometimes echo the request in it.
            raise ServiceUnavailableError("Skiddle reported an error")
        events: list[Spot] = []
        for item in data.get("results") or []:
            spot = _spot_from_skiddle(item, lat, lon, now)
            if spot is not None:
                events.append(spot)
        return events

    # ── Step 3: lore from Wikipedia ──────────────────────────────────────────

    def _add_lore(self, legends: list[Spot], deadline: float | None = None) -> None:
        timeout = LORE_TIMEOUT
        if deadline is not None:
            # Lore may use a little of the safety margin, but never all of it.
            timeout = min(LORE_TIMEOUT, deadline + 1.5 - time.monotonic())

        def lookup(spot: Spot) -> None:
            title = _wiki_title(spot.wikipedia)
            if title and timeout >= 0.8:
                url = WIKIPEDIA_SUMMARY + quote(title.replace(" ", "_"), safe="")
                try:
                    data = self._get_json(url, "Wikipedia", timeout)
                    sentence = _first_sentence((data or {}).get("extract", ""))
                    if sentence:
                        spot.lore = sentence
                        return
                except AgentAppError:
                    pass  # lore is a garnish; fall back to the map's own tags
            spot.lore = _lore_from_tags(spot)

        if not legends:
            return
        with ThreadPoolExecutor(max_workers=len(legends)) as pool:
            list(pool.map(lookup, legends))


# ── Pure helpers: no network, easy to test ───────────────────────────────────


def _distance_m(lat1, lon1, lat2, lon2) -> int:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return int(round(2 * r * math.asin(math.sqrt(h))))


def _radius_for(minutes: int | None) -> int:
    """How far to look: 1 km, or less if there is little time."""
    if minutes is None:
        return SEARCH_RADIUS_M
    one_way = (minutes - 15) / 2 * WALK_M_PER_MIN  # leave 15 min to enjoy it
    return int(max(300, min(SEARCH_RADIUS_M, one_way)))


def _fits(spot: Spot, minutes: int) -> bool:
    walk = 2 * spot.distance_m / WALK_M_PER_MIN
    return walk + DWELL_MIN.get(spot.rarity, 0) <= minutes


def _spot_from_osm(element: dict, lat0: float, lon0: float) -> Spot | None:
    tags = element.get("tags") or {}
    name = (tags.get("name") or "").strip()
    if not name:
        return None
    lat = element.get("lat") or (element.get("center") or {}).get("lat")
    lon = element.get("lon") or (element.get("center") or {}).get("lon")
    if lat is None or lon is None:
        return None
    if tags.get("memorial") in NOT_QUESTS or tags.get("historic") in NOT_QUESTS:
        return None
    famous = bool(tags.get("wikipedia") or tags.get("wikidata"))
    amenity, tourism, historic = tags.get("amenity"), tags.get("tourism"), tags.get("historic")

    if tourism == "museum":
        kind, rarity = "museum", LEGENDARY
    elif amenity in ("cafe", "restaurant"):
        kind, rarity = amenity, COMMON
    elif tourism == "viewpoint":
        kind, rarity = "viewpoint", LEGENDARY if famous else RARE
    elif tourism == "attraction":
        kind, rarity = "attraction", LEGENDARY if famous else RARE
    elif historic:
        kind, rarity = "historic", LEGENDARY if famous else RARE
    else:
        return None

    return Spot(
        name=name, kind=kind, rarity=rarity,
        distance_m=_distance_m(lat0, lon0, float(lat), float(lon)),
        lat=float(lat), lon=float(lon),
        wikipedia=tags.get("wikipedia", ""),
        extra={k: tags[k] for k in ("historic", "start_date", "description",
                                    "inscription", "heritage", "opening_hours")
               if k in tags},
    )


def _spot_from_event(item: dict, lat0: float, lon0: float, now: datetime) -> Spot | None:
    name = (item.get("name") or "").strip()
    dates = item.get("dates") or {}
    if not name or (dates.get("status") or {}).get("code") in ("cancelled", "postponed"):
        return None

    venues = (item.get("_embedded") or {}).get("venues") or [{}]
    venue = venues[0]
    location = venue.get("location") or {}
    try:
        lat, lon = float(location["latitude"]), float(location["longitude"])
        distance = _distance_m(lat0, lon0, lat, lon)
    except (KeyError, TypeError, ValueError):
        lat = lon = None
        distance = None

    start = _parse_utc((dates.get("start") or {}).get("dateTime"))
    end = _parse_utc((dates.get("end") or {}).get("dateTime"))
    if end is not None and end < now:
        return None
    if end is not None:
        when = f"ends {_clock(end)}"
    elif start is not None:
        when = f"starts {_clock(start)}, one showing only"
    else:
        local_date = (dates.get("start") or {}).get("localDate", "")
        when = f"on {local_date}" if local_date else "date to be confirmed"

    return Spot(name=name, kind="event", rarity=LIMITED, distance_m=distance,
                lat=lat, lon=lon, when=when, start=start, end=end,
                extra={"venue": venue.get("name", "")})


def _spot_from_skiddle(item: dict, lat0: float, lon0: float, now: datetime) -> Spot | None:
    """One Skiddle event. Read cautiously: Skiddle does not document its reply."""
    if not isinstance(item, dict):
        return None
    name = (item.get("eventname") or item.get("name") or "").strip()
    if not name or str(item.get("cancelled", "0")) in ("1", "true", "True"):
        return None

    venue = item.get("venue") if isinstance(item.get("venue"), dict) else {}
    try:
        lat, lon = float(venue["latitude"]), float(venue["longitude"])
        distance = _distance_m(lat0, lon0, lat, lon)
    except (KeyError, TypeError, ValueError):
        lat = lon = None
        distance = None

    start = _parse_utc(item.get("startdate"))
    end = _parse_utc(item.get("enddate"))
    if end is not None and end < now:
        return None
    if end is not None:
        when = f"ends {_clock(end)}"
    elif start is not None:
        when = f"starts {_clock(start)}, one showing only"
    elif item.get("date"):
        when = f"on {item['date']}"
    else:
        when = "date to be confirmed"

    return Spot(name=name, kind="event", rarity=LIMITED, distance_m=distance,
                lat=lat, lon=lon, when=when, start=start, end=end,
                extra={"venue": venue.get("name", "")})


def _merge_events(events: list[Spot]) -> list[Spot]:
    """Both services, one list: soonest first, each event once.

    The same show is often on both. Treat two events as one when their names
    match once case, punctuation and "the" are ignored, and they start within
    an hour of each other (or either has no start time).
    """
    far = datetime.max.replace(tzinfo=timezone.utc)
    ordered = sorted(events, key=lambda e: (
        e.start or far, e.distance_m if e.distance_m is not None else 10**9))
    kept: list[Spot] = []
    for event in ordered:
        key = _name_key(event.name)
        duplicate = any(
            _name_key(other.name) == key and (
                event.start is None or other.start is None
                or abs((event.start - other.start).total_seconds()) <= 3600)
            for other in kept
        )
        if not duplicate:
            kept.append(event)
    return kept


def _name_key(name: str) -> str:
    words = re.findall(r"[a-z0-9]+", name.lower())
    return " ".join(w for w in words if w != "the")


def _parse_utc(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _clock(moment: datetime) -> str:
    local = moment.astimezone(UK_TIME)
    return f"{local:%a} {local.day} {local:%b}, {local:%H:%M}"


def _pick_board(spots: list[Spot], events: list[Spot]) -> list[Spot]:
    """Up to six quests: one of each tier first, then round-robin to six."""
    cafes = [s for s in spots if s.kind == "cafe"]
    food = [s for s in spots if s.kind == "restaurant"]
    commons: list[Spot] = []
    for i in range(max(len(cafes), len(food))):  # alternate for variety
        commons += cafes[i:i + 1] + food[i:i + 1]
    tiers = {
        LEGENDARY: [s for s in spots if s.rarity == LEGENDARY],
        RARE: [s for s in spots if s.rarity == RARE],
        COMMON: commons,
        LIMITED: list(events),
    }

    board: list[Spot] = []
    order = [LEGENDARY, RARE, COMMON, LIMITED]
    while len(board) < MAX_QUESTS and any(tiers[t] for t in order):
        for tier in order:
            if tiers[tier] and len(board) < MAX_QUESTS:
                board.append(tiers[tier].pop(0))
    rank = {LEGENDARY: 0, LIMITED: 1, RARE: 2, COMMON: 3}
    board.sort(key=lambda s: (rank[s.rarity], s.distance_m if s.distance_m is not None else 10**9))
    return board


def _pick_chain(spots: list[Spot], board: list[Spot]) -> list[Spot]:
    """Three places, each a short walk from the last.

    Prefers places not already on the board, starting from the nearest Rare
    or Legendary, and always stepping to the closest place not yet used.
    """
    pool = [s for s in spots if s not in board and s.lat is not None]
    if len(pool) < 3:
        pool = [s for s in spots if s.lat is not None]
    if len(pool) < 3:
        return []
    starts = [s for s in pool if s.rarity in (RARE, LEGENDARY)] or pool
    chain = [min(starts, key=lambda s: s.distance_m)]
    while len(chain) < 3:
        last = chain[-1]
        rest = [s for s in pool if s not in chain]
        chain.append(min(rest, key=lambda s: _distance_m(last.lat, last.lon, s.lat, s.lon)))
    return chain


def _chain_lines(chain: list[Spot]) -> tuple[list[str], int]:
    lines, walked = [], 0
    for i, spot in enumerate(chain):
        if i == 0:
            hop = f"{_distance_text(spot)} from you"
        else:
            prev = chain[i - 1]
            step = _distance_m(prev.lat, prev.lon, spot.lat, spot.lon)
            walked += step
            hop = f"then ~{step} m on"
        line = f"{i + 1}. {BADGE[spot.rarity]}: {spot.name} ({hop}). {_flavour(spot)}"
        if spot.lore:
            line += f" Lore: {spot.lore}"
        lines.append(line)
    return lines, walked


def _quest_line(spot: Spot) -> str:
    line = (f"{BADGE[spot.rarity]} | {spot.name} | {_distance_text(spot)} | "
            f"{spot.xp} XP | {_flavour(spot)}")
    if spot.rarity == LEGENDARY and spot.lore:
        line += f" Lore: {spot.lore}"
    if spot.rarity == LIMITED:
        venue = spot.extra.get("venue")
        line += (f" At {venue}," if venue else "") + f" {spot.when}."
    return line


def _flavour(spot: Spot) -> str:
    """A stable, playful one-liner. crc32, not hash(): the same every call."""
    options = FLAVOUR.get(spot.kind) or FLAVOUR["attraction"]
    return options[zlib.crc32(spot.name.encode("utf-8")) % len(options)].format(n=spot.name)


def _wiki_title(tag: str) -> str:
    """OSM wikipedia tags look like "en:Tower of London". English only."""
    if not tag:
        return ""
    if ":" in tag:
        lang, title = tag.split(":", 1)
        return title.strip() if lang.strip().lower() == "en" else ""
    return tag.strip()


def _first_sentence(text: str, limit: int = 180) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    match = re.match(r"(.{25,}?[.!?])(\s+[A-Z\"]|$)", text)  # skip "St." and co.
    sentence = match.group(1) if match else text
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rsplit(" ", 1)[0] + "…"
    return sentence


def _lore_from_tags(spot: Spot) -> str:
    """Lore from OpenStreetMap's own tags, when Wikipedia has nothing."""
    tags = spot.extra
    if tags.get("description"):
        return _first_sentence(tags["description"], 140)
    if tags.get("start_date"):
        return f"It has stood here since {tags['start_date']}."
    if tags.get("inscription"):
        return _first_sentence(f'Its inscription reads: "{tags["inscription"]}"', 140)
    if tags.get("heritage"):
        return "It is a protected heritage site."
    return ""  # no fact to hand: better no lore than made-up lore


def _distance_text(spot: Spot) -> str:
    """Straight-line distance, marked as approximate: walking is further."""
    if spot.distance_m is None:
        return "nearby"
    return f"~{spot.distance_m} m"


def _events_in_window_first(events: list[Spot], now: datetime, minutes: int) -> list[Spot]:
    """With a time limit, events on now or starting within it come first."""
    until = now + timedelta(minutes=minutes)

    def in_window(event: Spot) -> bool:
        if event.start is None:
            return False
        if event.start <= now:
            return event.end is None or event.end >= now
        return event.start <= until

    return sorted(events, key=lambda e: 0 if in_window(e) else 1)  # stable


# ── Opening hours ─────────────────────────────────────────────────────────────

_DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
_DAY_LIST = re.compile(r"^([A-Z][a-z](?:-[A-Z][a-z])?(?:,[A-Z][a-z](?:-[A-Z][a-z])?)*)\s+(.+)$")
_RANGE = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})\+?$")


def _open_now(spec: str | None, now: datetime) -> bool | None:
    """Is a place open at `now` (UK time), going by its OSM opening_hours?

    True or False when the tag is in the common simple form
    ("Mo-Fr 09:00-17:30; Sa 10:00-16:00; Su off", "24/7", "10:00-22:00",
    overnight ranges such as "Fr,Sa 18:00-02:00"). None when there is no tag
    or it uses anything fancier (months, holidays, sunset, comments): then
    the place is kept, because unknown is not the same as closed.
    """
    if not spec or not spec.strip():
        return None
    spec = spec.strip()
    if spec == "24/7":
        return True

    week: dict[int, list[tuple[int, int]]] = {d: [] for d in range(7)}
    try:
        for rule in spec.split(";"):
            rule = rule.strip()
            if not rule:
                continue
            if rule.startswith(("PH", "SH")):
                continue  # public and school holidays: ignore those rules
            match = _DAY_LIST.match(rule)
            if match:
                days, times = _parse_days(match.group(1)), match.group(2).strip()
            else:
                days, times = list(range(7)), rule
            if times in ("off", "closed"):
                ranges: list[tuple[int, int]] = []
            elif times == "24/7":
                ranges = [(0, 1440)]
            else:
                ranges = [_parse_range(part.strip()) for part in times.split(",")]
            for day in days:  # a later rule replaces an earlier one for its days
                week[day] = ranges
    except (ValueError, KeyError):
        return None

    today, minute = now.weekday(), now.hour * 60 + now.minute
    for start, end in week[today]:
        if start < end and start <= minute < end:
            return True
        if end <= start and minute >= start:  # runs past midnight
            return True
    for start, end in week[(today - 1) % 7]:
        if end <= start and minute < end:  # yesterday's late opening
            return True
    return False


def _parse_days(text: str) -> list[int]:
    days: list[int] = []
    for piece in text.split(","):
        if "-" in piece:
            first, last = (_DAYS.index(p) for p in piece.split("-"))
            span = (last - first) % 7
            days += [(first + i) % 7 for i in range(span + 1)]
        else:
            days.append(_DAYS.index(piece))
    return days


def _parse_range(text: str) -> tuple[int, int]:
    match = _RANGE.match(text)
    if not match:
        raise ValueError(text)
    h1, m1, h2, m2 = (int(g) for g in match.groups())
    if h1 > 24 or h2 > 24 or m1 > 59 or m2 > 59:
        raise ValueError(text)
    return h1 * 60 + m1, h2 * 60 + m2

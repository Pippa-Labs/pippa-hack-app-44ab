"""London Side Quests: a quest board of things to do near a London location.

One call, four sources, no state:

1. Nominatim turns the place name into coordinates (kept inside London).
2. In parallel:
   - the OpenStreetMap Overpass API lists named cafés, restaurants,
     attractions, viewpoints, historic sites and museums within ~1 km;
   - the Ticketmaster Discovery API lists events nearby in the next 7 days
     (only if TICKETMASTER_KEY is set; if it fails, the board still comes back
     without Limited-time quests).
3. Wikipedia gives a one-line lore fact for the Legendary quests.

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
OVERPASS = "https://overpass-api.de/api/interpreter"
TICKETMASTER_EVENTS = "https://app.ticketmaster.com/discovery/v2/events.json"
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

#: Greater London, as Nominatim's viewbox: west, north, east, south.
LONDON_VIEWBOX = "-0.5104,51.6919,0.3340,51.2868"

UK_TIME = ZoneInfo("Europe/London")

SEARCH_RADIUS_M = 1000
EVENT_RADIUS_KM = 2
WALK_M_PER_MIN = 80

#: Timeouts, sharing one 20 second budget: geocode, then Overpass and
#: Ticketmaster side by side, then Wikipedia lookups side by side.
GEOCODE_TIMEOUT = 5.0
OVERPASS_TIMEOUT = 9.0
EVENTS_TIMEOUT = 6.0
LORE_TIMEOUT = 3.0

MAX_QUESTS = 6
COMMON, RARE, LEGENDARY, LIMITED = "Common", "Rare", "Legendary", "Limited"
BADGE = {COMMON: "⚪ Common", RARE: "🔷 Rare", LEGENDARY: "🌟 Legendary",
         LIMITED: "⚡ Limited-time"}
BASE_XP = {COMMON: 50, RARE: 120, LEGENDARY: 250, LIMITED: 200}
#: Minutes spent at a place, for fitting quests into the time available.
DWELL_MIN = {COMMON: 25, RARE: 15, LEGENDARY: 45}

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
    """Refuse redirects, so the Ticketmaster key never follows one anywhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


@dataclass(eq=False)
class Spot:
    name: str
    kind: str  # cafe, restaurant, viewpoint, museum, historic, attraction, event
    rarity: str
    distance_m: int
    lat: float | None = None
    lon: float | None = None
    wikipedia: str = ""
    lore: str = ""
    when: str = ""  # events only: when it ends
    extra: dict = field(default_factory=dict)

    @property
    def xp(self) -> int:
        bonus = 5 * round(self.distance_m / 50)  # a little more for walking further
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
        URL carries the key.
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

        name, lat, lon = self._geocode(place)
        radius = _radius_for(minutes)

        warnings: list[str] = []
        events: list[Spot] = []
        events_ok = False
        key = os.environ.get("TICKETMASTER_KEY", "").strip()
        with ThreadPoolExecutor(max_workers=2) as pool:
            places_job = pool.submit(self._places, lat, lon, radius)
            events_job = pool.submit(self._events, lat, lon, now, key) if key else None
            if events_job is not None:
                try:
                    events = events_job.result()
                    events_ok = True
                except Exception:  # noqa: BLE001 — events are optional by design
                    warnings.append(
                        "Live events could not be loaded, so there are no "
                        "Limited-time quests this time.")
            else:
                warnings.append(
                    "No Ticketmaster key is set up yet, so there are no "
                    "Limited-time quests.")
            spots = places_job.result()

        if minutes is not None:
            spots = [s for s in spots if _fits(s, minutes)]

        board = _pick_board(spots, events)
        if not board:
            error = NothingNearby(f"no quests within {radius} m of {name!r}")
            error.user_message = (
                f"I could not find anything to explore near {name}"
                + (f" in {minutes} minutes." if minutes else ".")
            )
            raise error

        chain = _pick_chain(spots, board)
        self._add_lore([s for s in board + chain if s.rarity == LEGENDARY][:4])

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

    def _places(self, lat: float, lon: float, radius: int) -> list[Spot]:
        around = f"(around:{radius},{lat:.6f},{lon:.6f})"
        query = (
            "[out:json][timeout:8];"
            f'(nwr{around}["amenity"~"^(cafe|restaurant)$"]["name"];)->.food;'
            f'(nwr{around}["tourism"~"^(attraction|viewpoint|museum)$"]["name"];'
            f'nwr{around}["historic"]["name"];)->.sights;'
            ".food out center tags 80;"
            ".sights out center tags 150;"
        )
        url = OVERPASS + "?" + urlencode({"data": query})
        data = self._get_json(url, "the Overpass API", OVERPASS_TIMEOUT)
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

    # ── Step 2b: events from Ticketmaster ────────────────────────────────────

    def _events(self, lat: float, lon: float, now: datetime, key: str) -> list[Spot]:
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

    # ── Step 3: lore from Wikipedia ──────────────────────────────────────────

    def _add_lore(self, legends: list[Spot]) -> None:
        def lookup(spot: Spot) -> None:
            title = _wiki_title(spot.wikipedia)
            if title:
                url = WIKIPEDIA_SUMMARY + quote(title.replace(" ", "_"), safe="")
                try:
                    data = self._get_json(url, "Wikipedia", LORE_TIMEOUT)
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
                                    "inscription", "heritage") if k in tags},
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
        distance = 0

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
                lat=lat, lon=lon, when=when,
                extra={"venue": venue.get("name", "")})


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
    board.sort(key=lambda s: (rank[s.rarity], s.distance_m))
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
            hop = f"{spot.distance_m} m from you"
        else:
            prev = chain[i - 1]
            step = _distance_m(prev.lat, prev.lon, spot.lat, spot.lon)
            walked += step
            hop = f"then {step} m on"
        line = f"{i + 1}. {BADGE[spot.rarity]}: {spot.name} ({hop}). {_flavour(spot)}"
        if spot.lore:
            line += f" Lore: {spot.lore}"
        lines.append(line)
    return lines, walked


def _quest_line(spot: Spot) -> str:
    line = (f"{BADGE[spot.rarity]} | {spot.name} | {spot.distance_m} m | "
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

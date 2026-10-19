"""Polite HTTP for the training-corpus fetcher: one User-Agent, per-host rate limits, robots.txt,
cached listings and capped, checked PDF downloads.

- **User-Agent** is fixed (:data:`USER_AGENT`) and carries no personal data.
- **Rate limits** are per bucket: every host maps to a bucket and a minimum interval between
  requests (:func:`host_policy`), raised to the host's robots.txt ``Crawl-delay``. All SEC hosts
  share one bucket (at most 5 requests/s); all arXiv hosts share one (OAI-PMH 1 per 3 s, PDFs 1 per
  15 s, the arxiv.org crawl delay); ``DEMO_KEY`` requests to api.data.gov APIs share one (10 per
  hour, the limit its response headers report).
- **robots.txt** is fetched once per web host and obeyed (``*`` and ``$`` patterns, longest match
  wins). When robots.txt cannot be read (401/403/429, a server or network error), nothing on that
  host is fetched this run, and the failure is *transient* (:class:`HostBlocked` or
  :class:`FetchError`), so the next run asks again; only an explicit disallow is permanent.
  Documented API hosts are machine interfaces and are exempt.
- **Redirects** are followed by this module, never by urllib, in a bounded loop: at most
  :data:`MAX_REDIRECTS` hops, http(s) only, never from https down to http, and each hop checked
  like a first request (the target host's block list, robots.txt and rate limit). A robots.txt
  fetch follows its own redirects without robots vetting, so vetting never re-enters itself.
- **No aggressive retries.** A 401, 403 or 429 blocks the host for the rest of the run; a network
  error or 5xx fails that one request (the next run retries it).
- **Downloads are untrusted data.** A PDF is streamed to ``<dest>.part``, refused above
  :data:`MAX_PDF_BYTES` (from Content-Length or while reading), refused unless it starts with the
  ``%PDF-`` magic, and renamed into place only when whole. Nothing downloaded is executed.
- **Listings** (API and index pages, never PDFs) are cached by URL with any ``api_key`` removed,
  so a resumed or repeated run lists the same candidates without asking again.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import re
import string
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

USER_AGENT = "docrooter-research/0.5 (research; contact via repository)"
ROBOTS_TOKEN = "docrooter-research"
MAX_PDF_BYTES = 50 * 1024 * 1024
PDF_MAGIC = b"%PDF-"
TIMEOUT_SECONDS = 120.0
DEFAULT_INTERVAL = 1.0
DEMO_KEY = "DEMO_KEY"
DEMO_KEY_INTERVAL = 360.0  # 10 requests/hour, as api.data.gov reported for DEMO_KEY
# Own api.data.gov keys: far inside govinfo (40/s) and congress.gov (5,000/h).
API_KEY_INTERVAL = 1.0

_SEC_HOSTS = frozenset({"www.sec.gov", "efts.sec.gov", "data.sec.gov"})
_API_DATA_GOV_HOSTS = frozenset({"api.govinfo.gov", "api.congress.gov"})
# Documented machine interfaces: robots.txt is for crawlers, these are APIs meant for programs.
_API_HOSTS = frozenset({"efts.sec.gov", "data.sec.gov", "oaipmh.arxiv.org"}) | _API_DATA_GOV_HOSTS
_BLOCKING_STATUSES = frozenset({401, 403, 429})
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MAX_REDIRECTS = 5


class FetchError(Exception):
    """A request failed in a way the next run may not repeat (network error, 5xx)."""


class HostBlocked(FetchError):
    """The host refused us (401/403/429, robots.txt, an exhausted DEMO_KEY): stop asking it."""


class Rejected(Exception):
    """The response is not an acceptable PDF (404, wrong magic, over the size cap): permanent."""


@dataclass(frozen=True)
class HostPolicy:
    bucket: str
    interval: float
    robots: bool


def host_policy(host: str, *, demo_key: bool = False) -> HostPolicy:
    """The rate-limit bucket, minimum interval (seconds) and robots rule for ``host``."""
    host = host.lower()
    if host in _SEC_HOSTS:
        return HostPolicy("sec.gov", 0.2, robots=host not in _API_HOSTS)
    if host == "oaipmh.arxiv.org":
        return HostPolicy("arxiv", 3.0, robots=False)
    if host in ("arxiv.org", "export.arxiv.org"):
        return HostPolicy("arxiv", 15.0, robots=True)
    if host in _API_DATA_GOV_HOSTS:
        if demo_key:
            return HostPolicy("api.data.gov DEMO_KEY", DEMO_KEY_INTERVAL, robots=False)
        return HostPolicy(host, API_KEY_INTERVAL, robots=False)
    return HostPolicy(host, DEFAULT_INTERVAL, robots=True)


class RateLimiter:
    """Minimum spacing between the starts of requests in the same bucket."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.clock = clock
        self.sleep = sleep
        self._last: dict[str, float] = {}

    def wait(self, bucket: str, interval: float) -> None:
        last = self._last.get(bucket)
        if last is not None:
            remaining = last + interval - self.clock()
            if remaining > 0:
                self.sleep(remaining)
        self._last[bucket] = self.clock()


# --- robots.txt ---------------------------------------------------------------------------


@dataclass
class Robots:
    """A parsed robots.txt group for our agent: allow/disallow rules and the crawl delay."""

    rules: list[tuple[bool, str]]
    crawl_delay: float | None = None
    status: str = "ok"
    error: str | None = None  # robots.txt could not be read: nothing is fetched this run
    refused: bool = False  # the error was a 401/403/429 (the host refused us)

    def allows(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        best: tuple[int, bool] | None = None
        for allow, pattern in self.rules:
            if not pattern:
                continue  # "Disallow:" with no path allows everything
            if _robots_match(pattern, path):
                key = (len(pattern), allow)  # longest match wins; on a tie, allow wins
                if best is None or key > best:
                    best = key
        return True if best is None else best[1]


def _robots_match(pattern: str, path: str) -> bool:
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    regex = "".join(".*" if ch == "*" else re.escape(ch) for ch in body)
    return re.match(regex + ("$" if anchored else ""), path) is not None


def parse_robots(text: str, token: str = ROBOTS_TOKEN) -> Robots:
    """Our group of a robots.txt: the group naming ``token`` if any, else the ``*`` group."""
    groups: list[tuple[list[str], list[tuple[bool, str]], list[float]]] = []
    agents: list[str] = []
    rules: list[tuple[bool, str]] = []
    delays: list[float] = []
    in_rules = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        field, value = (part.strip() for part in line.split(":", 1))
        field = field.lower()
        if field == "user-agent":
            if in_rules:
                groups.append((agents, rules, delays))
                agents, rules, delays = [], [], []
                in_rules = False
            agents.append(value.lower())
        elif field in ("allow", "disallow"):
            in_rules = True
            rules.append((field == "allow", value))
        elif field == "crawl-delay":
            in_rules = True
            try:
                delays.append(float(value))
            except ValueError:
                pass
    if agents:
        groups.append((agents, rules, delays))
    token = token.lower()
    chosen = next((g for g in groups if any(a != "*" and a in token for a in g[0])), None)
    if chosen is None:
        chosen = next((g for g in groups if "*" in g[0]), None)
    if chosen is None:
        return Robots(rules=[])
    return Robots(rules=chosen[1], crawl_delay=max(chosen[2]) if chosen[2] else None)


# --- the client ---------------------------------------------------------------------------

Transport = Callable[[urllib.request.Request, float], Any]


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Hands every 3xx back as an ``HTTPError``: :meth:`Net._open` follows redirects itself,
    iteratively and bounded, so urllib never recurses into the fetcher."""

    def redirect_request(
        self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> urllib.request.Request | None:
        return None


def redirect_target(base_url: str, location: str) -> str:
    """The absolute, percent-quoted URL a ``Location`` header points to, built as urllib's own
    ``HTTPRedirectHandler`` builds it: an empty path becomes ``/``, the value is quoted with
    ISO-8859-1 (http.client decodes header bytes that way, so UTF-8 bytes on the wire become
    their own ``%XX`` escapes) keeping punctuation, and it is joined to the current URL."""
    parts = urllib.parse.urlparse(location)
    if not parts.path and parts.netloc:
        parts = parts._replace(path="/")
    quoted = urllib.parse.quote(
        urllib.parse.urlunparse(parts), encoding="iso-8859-1", safe=string.punctuation
    )
    return urllib.parse.urljoin(base_url, quoted)


def _netloc(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.lower()


def _strip_key(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query) if k.lower() != "api_key"]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def redact(url: str) -> str:
    """``url`` with any ``api_key`` value hidden, for logs."""
    return re.sub(r"(?i)(api_key=)[^&]+", r"\1***", url)


def _uses_demo_key(url: str) -> bool:
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    return DEMO_KEY in query.get("api_key", [])


class Net:
    """The fetcher's only way to the network (see the module docstring)."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        archive_dir: Path | None = None,
        transport: Transport | None = None,
        limiter: RateLimiter | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.cache_dir = cache_dir
        self.archive_dir = archive_dir  # where sources keep downloaded dataset archives
        self.opener = urllib.request.build_opener(_NoRedirects())
        self.transport: Transport = transport or (
            lambda request, timeout: self.opener.open(request, timeout=timeout)
        )
        self.limiter = limiter or RateLimiter()
        self.log = log
        self._robots: dict[str, Robots] = {}  # by netloc: robots.txt is per host and port
        self._blocked: dict[str, str] = {}
        self.requests = 0

    # -- policy

    def _policy(self, url: str) -> HostPolicy:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        policy = host_policy(host, demo_key=_uses_demo_key(url))
        robots = self._robots.get(_netloc(url))
        if robots is not None and robots.crawl_delay:
            policy = HostPolicy(
                policy.bucket, max(policy.interval, robots.crawl_delay), policy.robots
            )
        return policy

    def blocked(self, url: str) -> str | None:
        """Why requests to ``url``'s host (or its rate bucket) stopped this run, if they did."""
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        return self._blocked.get(host) or self._blocked.get(self._policy(url).bucket)

    def _open(self, request: urllib.request.Request, *, robots_fetch: bool = False) -> Any:
        """Send ``request`` and follow up to :data:`MAX_REDIRECTS` redirects in a loop. Every
        hop passes the block list and the rate limit; every redirect target is vetted by
        :meth:`check_redirect`, except within a robots.txt fetch (``robots_fetch``), which must
        not trigger robots vetting itself."""
        for hop in range(MAX_REDIRECTS + 1):
            response, status, location = self._send(request)
            if location is None:
                return response
            if hop == MAX_REDIRECTS:
                break
            new_url = redirect_target(request.full_url, location)
            self.check_redirect(request.full_url, new_url, vet_robots=not robots_fetch)
            request = self._follow(request, new_url, status)
        raise FetchError(f"more than {MAX_REDIRECTS} redirects from {redact(request.full_url)}")

    def _follow(
        self, request: urllib.request.Request, new_url: str, status: int
    ) -> urllib.request.Request:
        """The next request of a redirect: 307/308 keep the method and body, others GET."""
        if status in (307, 308) and request.data is not None:
            return self._request(new_url, request.data)  # type: ignore[arg-type]
        return self._request(new_url)

    def _send(self, request: urllib.request.Request) -> tuple[Any, int, str | None]:
        """One HTTP exchange: ``(response, status, None)``, or ``(None, status, location)``
        for a redirect."""
        url = request.full_url
        reason = self.blocked(url)
        if reason:
            raise HostBlocked(reason)
        policy = self._policy(url)
        self.limiter.wait(policy.bucket, policy.interval)
        self.requests += 1
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        try:
            response = self.transport(request, TIMEOUT_SECONDS)
        except urllib.error.HTTPError as exc:
            location = exc.headers.get("Location") if exc.headers is not None else None
            exc.close()
            if exc.code in _REDIRECT_STATUSES and location:
                return None, exc.code, str(location)
            if exc.code in _BLOCKING_STATUSES:
                why = f"{host} answered HTTP {exc.code}; not asking it again this run"
                self._blocked[host] = why
                raise HostBlocked(why) from None
            if exc.code in (404, 410):
                raise Rejected(f"HTTP {exc.code}") from None
            raise FetchError(f"HTTP {exc.code} from {redact(url)}") from None
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
            raise FetchError(f"{type(exc).__name__}: {exc} ({redact(url)})") from None
        except RecursionError:  # defensive: never let one URL take the run down
            raise FetchError(f"RecursionError while fetching {redact(url)}") from None
        if policy.bucket.endswith(DEMO_KEY):
            remaining = (response.headers.get("X-RateLimit-Remaining") or "").strip()
            if remaining == "0":
                self._blocked[policy.bucket] = "DEMO_KEY hourly limit reached; listings stay cached"
        return response, int(getattr(response, "status", 200) or 200), None

    def _request(self, url: str, body: bytes | None = None) -> urllib.request.Request:
        headers = {"User-Agent": USER_AGENT}
        if body is not None:
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(
            url, data=body, headers=headers, method="POST" if body is not None else "GET"
        )

    # -- robots.txt

    def allowed(self, url: str) -> bool:
        """Whether robots.txt lets us fetch ``url`` (API hosts are exempt). One robots.txt
        request per host per run. Raises :class:`HostBlocked` or :class:`FetchError` when
        robots.txt could not be read: that is transient, never a reason to drop for good."""
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
        if not host_policy(host).robots:
            return True
        key = parts.netloc.lower()
        if key not in self._robots:
            self._robots[key] = self._fetch_robots(f"{parts.scheme}://{parts.netloc}/robots.txt")
        robots = self._robots[key]
        if robots.error is not None:
            raise (HostBlocked if robots.refused else FetchError)(robots.error)
        return robots.allows(url)

    def check_redirect(self, old_url: str, new_url: str, *, vet_robots: bool = True) -> None:
        """Vet one redirect hop as if it were a first request: http(s) only and no https ->
        http downgrade (:class:`Rejected`), the target host not blocked, and (unless inside a
        robots.txt fetch) robots.txt allowing it. The hop's rate limit is applied when it is
        sent. Not recursive: a robots.txt fetch it triggers follows redirects unvetted."""
        old, new = urllib.parse.urlsplit(old_url), urllib.parse.urlsplit(new_url)
        if new.scheme not in ("http", "https") or (old.scheme == "https" and new.scheme != "https"):
            raise Rejected(f"redirect from {old.scheme} to {new.scheme} refused")
        reason = self.blocked(new_url)
        if reason:
            raise HostBlocked(reason)
        if vet_robots and not self.allowed(new_url):
            raise Rejected("robots.txt disallows the redirect target")

    def robots_status(self, url: str) -> str:
        robots = self._robots.get(_netloc(url))
        return robots.status if robots else "not fetched"

    def _fetch_robots(self, robots_url: str) -> Robots:
        try:
            with self._open(self._request(robots_url), robots_fetch=True) as response:
                text = response.read(512 * 1024).decode("utf-8", "replace")
        except Rejected as exc:  # 404/410: no robots.txt, nothing is disallowed
            return Robots(rules=[], status=f"no robots.txt ({exc})")
        except HostBlocked as exc:
            why = f"robots.txt refused: {exc}"
            return Robots(rules=[], status=why, error=why, refused=True)
        except FetchError as exc:
            why = f"robots.txt unavailable: {exc}"
            return Robots(rules=[], status=why, error=why)
        robots = parse_robots(text)
        robots.status = "ok"
        return robots

    # -- listings

    def listing(self, url: str, body: bytes | None = None) -> bytes:
        """The body of a listing request (GET, or POST of ``body``), from the cache when this
        exact request (``api_key`` aside) was answered before."""
        cache_path = None
        if self.cache_dir is not None:
            key = hashlib.sha1(_strip_key(url).encode() + b"\0" + (body or b"")).hexdigest()
            cache_path = self.cache_dir / f"{key}.bin"
            if cache_path.exists():
                return cache_path.read_bytes()
        if not self.allowed(url):
            raise HostBlocked(f"robots.txt disallows {redact(url)}")
        try:
            with self._open(self._request(url, body)) as response:
                data: bytes = response.read()
        except (http.client.HTTPException, OSError) as exc:  # a broken read, not a refusal
            raise FetchError(f"{type(exc).__name__}: {exc} ({redact(url)})") from None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cache_path)
        return data

    # -- PDFs

    def download_pdf(self, url: str, dest: Path, max_bytes: int = MAX_PDF_BYTES) -> int:
        """Stream ``url`` to ``dest`` if it is a PDF of at most ``max_bytes``; the byte count.
        :class:`Rejected` for a non-PDF, oversized or missing file; nothing is left behind."""
        return self.download(url, dest, max_bytes=max_bytes, magic=PDF_MAGIC)

    def download(self, url: str, dest: Path, *, max_bytes: int, magic: bytes) -> int:
        """Stream ``url`` to ``dest`` if it starts with ``magic`` and has at most ``max_bytes``."""
        if not self.allowed(url):
            raise Rejected("robots.txt disallows this URL")
        part = dest.with_name(dest.name + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._open(self._request(url)) as response:
                length = (response.headers.get("Content-Length") or "").strip()
                if length.isdigit() and int(length) > max_bytes:
                    raise Rejected(f"Content-Length {length} is over the {max_bytes}-byte cap")
                head = response.read(len(magic))
                if not head.startswith(magic):
                    raise Rejected(f"wrong file type (no {magic!r} magic)")
                size = len(head)
                with part.open("wb") as out:
                    out.write(head)
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        size += len(block)
                        if size > max_bytes:
                            raise Rejected(f"over the {max_bytes}-byte cap")
                        out.write(block)
                    out.flush()
                    os.fsync(out.fileno())
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            part.unlink(missing_ok=True)
            raise FetchError(f"{type(exc).__name__}: {exc} ({redact(url)})") from None
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, dest)
        return size

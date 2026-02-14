"""Check GitHub for newer tscribe releases."""

from __future__ import annotations

import json
import urllib.request

_RELEASES_URL = "https://api.github.com/repos/johnrobinsn/tscribe/releases/latest"
_TIMEOUT = 3  # seconds


def get_latest_release() -> tuple[str, str] | None:
    """Query GitHub releases API for the latest release.

    Returns ``(version, release_url)`` or ``None`` on any failure.
    The version string has a leading ``v`` stripped if present.
    """
    try:
        req = urllib.request.Request(
            _RELEASES_URL, headers={"Accept": "application/vnd.github+json"}
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
        tag = data.get("tag_name", "")
        url = data.get("html_url", "")
        if not tag:
            return None
        version = tag.lstrip("v")
        return version, url
    except Exception:
        return None


def is_newer(latest: str, current: str) -> bool:
    """Return True if *latest* is a higher version than *current*."""
    try:
        lat = tuple(int(x) for x in latest.split("."))
        cur = tuple(int(x) for x in current.split("."))
        return lat > cur
    except (ValueError, TypeError):
        return False

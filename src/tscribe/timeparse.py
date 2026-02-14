"""Duration and time parsing for scheduled recording."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_DURATION_RE = re.compile(
    r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", re.IGNORECASE
)

_CLOCK_12_RE = re.compile(
    r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$", re.IGNORECASE
)

_CLOCK_24_RE = re.compile(r"^(\d{1,2}):(\d{2})$")

_DATE_TIME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|\d{1,2}:\d{2})$",
    re.IGNORECASE,
)


def parse_duration(s: str) -> float:
    """Parse a duration string like '5m', '1h30m', '2h30m30s' into seconds.

    Raises ValueError on invalid input.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty duration string")

    m = _DURATION_RE.match(s)
    if not m or not any(m.groups()):
        raise ValueError(f"Invalid duration: {s!r}")

    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(m.group(3) or 0)
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError(f"Duration must be positive: {s!r}")
    return float(total)


def _parse_clock(s: str) -> tuple[int, int]:
    """Parse a clock time string, return (hour24, minute)."""
    m = _CLOCK_12_RE.match(s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3).lower()
        if hour < 1 or hour > 12 or minute > 59:
            raise ValueError(f"Invalid time: {s!r}")
        if ampm == "am" and hour == 12:
            hour = 0
        elif ampm == "pm" and hour != 12:
            hour += 12
        return hour, minute

    m = _CLOCK_24_RE.match(s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        if hour > 23 or minute > 59:
            raise ValueError(f"Invalid time: {s!r}")
        return hour, minute

    raise ValueError(f"Invalid time format: {s!r}")


def parse_time(s: str) -> datetime:
    """Parse a time string into a local datetime.

    Supported formats:
      - Clock: '14:30', '9am', '2:30pm'
      - Tomorrow: 'tomorrow 9am', 'tomorrow 14:30'
      - Date+time: '2026-02-15 14:30'

    Clock times in the past today are interpreted as tomorrow.
    Raises ValueError on invalid input.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty time string")

    # Date+time: '2026-02-15 14:30' or '2026-02-15 9am'
    m = _DATE_TIME_RE.match(s)
    if m:
        date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        hour, minute = _parse_clock(m.group(2))
        return datetime(date.year, date.month, date.day, hour, minute)

    # Tomorrow prefix
    lower = s.lower()
    if lower.startswith("tomorrow"):
        rest = s[8:].strip()
        if not rest:
            raise ValueError("Expected time after 'tomorrow'")
        hour, minute = _parse_clock(rest)
        tomorrow = datetime.now().date() + timedelta(days=1)
        return datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour, minute)

    # Plain clock time — today or tomorrow if past
    hour, minute = _parse_clock(s)
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target

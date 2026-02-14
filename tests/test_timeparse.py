"""Tests for the timeparse module."""

from datetime import datetime, timedelta

import pytest

from tscribe.timeparse import parse_duration, parse_time


# ──── parse_duration ────


class TestParseDuration:
    def test_seconds(self):
        assert parse_duration("30s") == 30.0

    def test_minutes(self):
        assert parse_duration("5m") == 300.0

    def test_hours(self):
        assert parse_duration("1h") == 3600.0

    def test_hours_minutes(self):
        assert parse_duration("1h30m") == 5400.0

    def test_hours_minutes_seconds(self):
        assert parse_duration("2h30m30s") == 9030.0

    def test_large_minutes(self):
        assert parse_duration("90m") == 5400.0

    def test_whitespace_stripped(self):
        assert parse_duration("  5m  ") == 300.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            parse_duration("")

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("abc")

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("30x")

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            parse_duration("0s")


# ──── parse_time ────


class TestParseTime:
    def test_24h_clock(self):
        result = parse_time("14:30")
        assert result.hour == 14
        assert result.minute == 30
        # Should be today or tomorrow
        assert result.date() >= datetime.now().date()

    def test_12h_am(self):
        result = parse_time("9am")
        assert result.hour == 9
        assert result.minute == 0

    def test_12h_pm(self):
        result = parse_time("2:30pm")
        assert result.hour == 14
        assert result.minute == 30

    def test_12am_is_midnight(self):
        result = parse_time("12am")
        assert result.hour == 0

    def test_12pm_is_noon(self):
        result = parse_time("12pm")
        assert result.hour == 12

    def test_clock_in_past_wraps_to_tomorrow(self):
        """A clock time that's already passed today should be tomorrow."""
        # Use a time that's definitely in the past (1 hour ago)
        past = datetime.now() - timedelta(hours=1)
        time_str = f"{past.hour}:{past.minute:02d}"
        result = parse_time(time_str)
        assert result > datetime.now()
        assert result.date() == datetime.now().date() + timedelta(days=1)

    def test_tomorrow_9am(self):
        result = parse_time("tomorrow 9am")
        tomorrow = datetime.now().date() + timedelta(days=1)
        assert result.date() == tomorrow
        assert result.hour == 9
        assert result.minute == 0

    def test_tomorrow_24h(self):
        result = parse_time("tomorrow 14:30")
        tomorrow = datetime.now().date() + timedelta(days=1)
        assert result.date() == tomorrow
        assert result.hour == 14
        assert result.minute == 30

    def test_date_time(self):
        result = parse_time("2026-02-15 14:30")
        assert result == datetime(2026, 2, 15, 14, 30)

    def test_date_time_12h(self):
        result = parse_time("2026-02-15 9am")
        assert result == datetime(2026, 2, 15, 9, 0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            parse_time("")

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            parse_time("not-a-time")

    def test_tomorrow_no_time_raises(self):
        with pytest.raises(ValueError, match="Expected time"):
            parse_time("tomorrow")

    def test_invalid_hour(self):
        with pytest.raises(ValueError, match="Invalid"):
            parse_time("25:00")

    def test_invalid_minute(self):
        with pytest.raises(ValueError, match="Invalid"):
            parse_time("14:60")

    def test_whitespace_stripped(self):
        result = parse_time("  9am  ")
        assert result.hour == 9

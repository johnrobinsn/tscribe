"""Tests for update_check module."""

import json
from unittest.mock import MagicMock, patch

from tscribe.update_check import get_latest_release, is_newer


class TestIsNewer:
    def test_newer_patch(self):
        assert is_newer("1.0.1", "1.0.0") is True

    def test_newer_minor(self):
        assert is_newer("1.1.0", "1.0.0") is True

    def test_newer_major(self):
        assert is_newer("2.0.0", "1.0.0") is True

    def test_same(self):
        assert is_newer("1.0.0", "1.0.0") is False

    def test_older(self):
        assert is_newer("1.0.0", "1.1.0") is False

    def test_two_part_version(self):
        assert is_newer("1.1", "1.0") is True

    def test_invalid_version(self):
        assert is_newer("abc", "1.0.0") is False

    def test_empty(self):
        assert is_newer("", "1.0.0") is False


class TestGetLatestRelease:
    def test_success(self):
        body = json.dumps({
            "tag_name": "v1.2.0",
            "html_url": "https://github.com/johnrobinsn/tscribe/releases/tag/v1.2.0",
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tscribe.update_check.urllib.request.urlopen", return_value=mock_resp):
            result = get_latest_release()

        assert result == (
            "1.2.0",
            "https://github.com/johnrobinsn/tscribe/releases/tag/v1.2.0",
        )

    def test_strips_v_prefix(self):
        body = json.dumps({"tag_name": "v2.0", "html_url": "https://example.com"}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tscribe.update_check.urllib.request.urlopen", return_value=mock_resp):
            result = get_latest_release()

        assert result is not None
        assert result[0] == "2.0"

    def test_network_error(self):
        with patch(
            "tscribe.update_check.urllib.request.urlopen",
            side_effect=OSError("Connection refused"),
        ):
            assert get_latest_release() is None

    def test_bad_json(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tscribe.update_check.urllib.request.urlopen", return_value=mock_resp):
            assert get_latest_release() is None

    def test_missing_tag(self):
        body = json.dumps({"html_url": "https://example.com"}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("tscribe.update_check.urllib.request.urlopen", return_value=mock_resp):
            assert get_latest_release() is None

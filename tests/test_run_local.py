import pytest

from run_local import configured_port


def test_configured_port_defaults_and_validates(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    assert configured_port() == 8000

    monkeypatch.setenv("PORT", "8001")
    assert configured_port() == 8001

    monkeypatch.setenv("PORT", "invalid")
    with pytest.raises(SystemExit, match="integer"):
        configured_port()

    monkeypatch.setenv("PORT", "70000")
    with pytest.raises(SystemExit, match="between"):
        configured_port()

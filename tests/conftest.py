"""Shared pytest fixtures for the Customer Operations Agent test suite."""

import pytest


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test if any code under test attempts to open a socket connection."""

    def fail_on_network(*args, **kwargs):
        raise AssertionError("Unexpected network access during test.")

    monkeypatch.setattr("socket.socket.connect", fail_on_network)

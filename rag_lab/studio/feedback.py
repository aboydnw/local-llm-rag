"""Tier-2 feedback event logging; no-ops when the recorder isn't installed."""

try:
    from streamlit_testing_feedback import instrument, log_event
except ImportError:
    from contextlib import contextmanager

    def log_event(type: str, **payload) -> None:
        """No-op stand-in when streamlit-testing-feedback is absent."""

    @contextmanager
    def instrument(name: str, **payload):
        """No-op stand-in when streamlit-testing-feedback is absent."""
        yield


__all__ = ["log_event", "instrument"]

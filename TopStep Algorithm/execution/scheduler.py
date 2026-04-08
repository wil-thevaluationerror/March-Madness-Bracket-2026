from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from config import SessionConfig, SessionWindow


class SessionScheduler:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.tz = ZoneInfo(config.timezone)

    def localize(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self.tz)
        return dt.astimezone(self.tz)

    def _tod(self, dt: datetime) -> time:
        return self.localize(dt).timetz().replace(tzinfo=None)

    @staticmethod
    def _minutes(value: time) -> int:
        return (value.hour * 60) + value.minute

    def _windows(self) -> tuple[SessionWindow, ...]:
        if self.config.session_windows:
            return self.config.session_windows
        return (
            SessionWindow(
                label="default",
                market_open=self.config.market_open,
                no_new_trades_after=self.config.no_new_trades_after,
                force_flatten_at=self.config.force_flatten_at,
                exchange_close=self.config.exchange_close,
            ),
        )

    def _window_offset(self, tod: time, window: SessionWindow) -> int:
        start_minutes = self._minutes(window.market_open)
        current_minutes = self._minutes(tod)
        if current_minutes < start_minutes:
            current_minutes += 24 * 60
        return current_minutes - start_minutes

    def _window_length(self, window: SessionWindow) -> int:
        return self._window_offset(window.exchange_close, window)

    def _active_window(self, dt: datetime) -> SessionWindow | None:
        tod = self._tod(dt)
        for window in self._windows():
            offset = self._window_offset(tod, window)
            if 0 <= offset < self._window_length(window):
                return window
        return None

    def is_trading_session(self, dt: datetime) -> bool:
        return self._active_window(dt) is not None

    def is_past_new_trade_cutoff(self, dt: datetime) -> bool:
        window = self._active_window(dt)
        if window is None:
            return False
        tod = self._tod(dt)
        return self._window_offset(tod, window) >= self._window_offset(window.no_new_trades_after, window)

    def should_force_flatten(self, dt: datetime) -> bool:
        window = self._active_window(dt)
        if window is None:
            return False
        tod = self._tod(dt)
        return self._window_offset(tod, window) >= self._window_offset(window.force_flatten_at, window)

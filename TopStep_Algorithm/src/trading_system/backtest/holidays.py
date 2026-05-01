"""
US Futures Market Holiday Calendar
====================================

Generates the set of dates on which CME Group MES futures have a full-day
trading halt, plus a small number of structurally thin-market sessions that
consistently produce adverse backtest results (e.g. Black Friday, Christmas Eve).

The goal is NOT to exclude every low-liquidity day — the ATR floor in profiles.py
handles most of that.  This calendar handles:
  1. Official full-day CME closures (market simply does not trade)
  2. Extreme thin-market days with documented adverse WFO performance
     (Black Friday, Christmas Eve — these are early-close days with near-zero
     MES volume in the NY morning session that the algo targets)

Usage
-----
    from backtest.holidays import us_futures_skip_dates
    skip = us_futures_skip_dates(2024, 2026)   # dates in [2024, 2026] inclusive
    config.session.skip_dates = skip
"""
from __future__ import annotations

from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Fixed-date CME full-session closures
# ---------------------------------------------------------------------------
_FIXED_HOLIDAYS: list[tuple[int, int]] = [
    (1, 1),    # New Year's Day
    (6, 19),   # Juneteenth National Independence Day (since 2022)
    (7, 4),    # Independence Day
    (12, 25),  # Christmas Day
]

# Thin-market sessions added beyond official closures.
# These are early-close / very-low-volume days that fall within the
# 08:30–12:30 CT trading window the strategy uses.
_EXTRA_THIN_MARKET: list[tuple[int, int]] = [
    (12, 24),  # Christmas Eve — early close, near-zero NY morning volume
    (11, 11),  # Veterans Day — CME is OPEN but equity/bond markets closed → erratic MES flow
    (12, 26),  # Day after Christmas — if weekday, volume is severely impaired
    (1, 2),    # Day after New Year's Day — thin carry-over session
]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th occurrence (1-based) of a given weekday in a month."""
    first = date(year, month, 1)
    diff = (weekday - first.weekday()) % 7
    return first + timedelta(days=diff + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of a given weekday in a month."""
    # Start from the last day of the month and walk back.
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    diff = (last.weekday() - weekday) % 7
    return last - timedelta(days=diff)


def _good_friday(year: int) -> date:
    """Compute Good Friday using the Anonymous Gregorian algorithm for Easter."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    easter = date(year, month, day)
    return easter - timedelta(days=2)


def _variable_holidays(year: int) -> list[date]:
    """Return variable-date CME holiday closures for a given year."""
    return [
        _nth_weekday(year, 1, 0, 3),   # MLK Day: 3rd Monday of January
        _nth_weekday(year, 2, 0, 3),   # Presidents Day: 3rd Monday of February
        _good_friday(year),            # Good Friday
        _last_weekday(year, 5, 0),     # Memorial Day: last Monday of May
        _nth_weekday(year, 9, 0, 1),   # Labor Day: 1st Monday of September
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving: 4th Thursday of November
    ]


def _observed_holidays(year: int) -> list[date]:
    """
    When a fixed holiday falls on Saturday, US markets observe on Friday.
    When it falls on Sunday, markets observe on Monday.
    """
    observed = []
    for month, day in _FIXED_HOLIDAYS:
        try:
            d = date(year, month, day)
        except ValueError:
            continue
        wd = d.weekday()
        if wd == 5:   # Saturday → observe Friday
            observed.append(d - timedelta(days=1))
        elif wd == 6:  # Sunday → observe Monday
            observed.append(d + timedelta(days=1))
        else:
            observed.append(d)
    return observed


def _day_after_thanksgiving(year: int) -> date:
    """Black Friday (day after Thanksgiving) — thin market, early close."""
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    return thanksgiving + timedelta(days=1)


def _pre_christmas_thin_days(year: int) -> list[date]:
    """
    Return the weekday trading days in the 10 calendar days before Dec 25 (Dec 15–24).

    Institutional desks reduce risk over the final two weeks before Christmas:
    volume dries up, ATR compresses, and breakout signals reverse more frequently.
    W9 (Nov 30 – Dec 22, 2025) shows 17.6% WR driven largely by Dec 15–22.
    The existing ADX≥25 + ATR-floor filters do not fully capture this structural
    regime shift — a calendar exclusion is a cleaner fix.
    """
    christmas = date(year, 12, 25)
    thin = []
    for offset in range(1, 11):  # Dec 15–24
        d = christmas - timedelta(days=offset)
        if d.weekday() < 5:  # weekday only
            thin.append(d)
    return thin


def _pre_new_year_thin_days(year: int) -> list[date]:
    """
    Return the weekday trading days in the 3 calendar days before Jan 1 of year+1.

    Dec 29–31 see similar year-end thin-market conditions as the Christmas week.
    """
    new_year = date(year + 1, 1, 1)
    thin = []
    for offset in range(1, 4):  # Dec 29–31
        d = new_year - timedelta(days=offset)
        if d.weekday() < 5:
            thin.append(d)
    return thin


def us_futures_skip_dates(start_year: int, end_year: int) -> tuple[date, ...]:
    """
    Return a sorted tuple of dates in [start_year, end_year] (inclusive)
    on which the strategy should not enter new trades.

    Includes:
    - All CME full-day holiday closures (fixed + variable + observed weekends)
    - Black Friday (day after Thanksgiving) — early close / thin
    - Veterans Day (Nov 11) — CME open but equity/bond closure → erratic MES flow
    - Pre-Christmas thin period (Dec 20–24 weekdays)
    - Pre-New-Year thin period (Dec 29–31 weekdays)
    - Christmas Eve (Dec 24), Dec 26, Jan 2 (structural thin-market carry-over days)
    """
    dates: set[date] = set()
    for year in range(start_year, end_year + 1):
        # Official full-day closures
        dates.update(_observed_holidays(year))
        dates.update(_variable_holidays(year))
        # Black Friday (early close / thin)
        dates.add(_day_after_thanksgiving(year))
        # Pre-holiday thin periods
        dates.update(_pre_christmas_thin_days(year))
        dates.update(_pre_new_year_thin_days(year))
        # Extra thin-market sessions
        for month, day in _EXTRA_THIN_MARKET:
            try:
                d = date(year, month, day)
                if d.weekday() < 5:  # only add if weekday
                    dates.add(d)
            except ValueError:
                pass

    return tuple(sorted(dates))

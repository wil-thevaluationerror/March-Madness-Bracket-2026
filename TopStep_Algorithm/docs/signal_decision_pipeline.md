# London Sweep Signal Decision Pipeline

This document defines the observable decision path from raw London-session bars
to accepted trades. The raw setup ledger starts at a deliberately narrow
threshold: a 5-minute London-session bar with enough local history for ATR-14
evaluation. It does not log off-session bars or early warmup bars, because those
are not actionable raw setup candidates.

| Stage | Data Used | Entry-Time Only | Rejection Reason | Category |
| --- | --- | --- | --- | --- |
| Session/time window eligibility | SessionDay London indices, 08:30-13:30 UTC | Yes | `outside_session` | market-structure rejection |
| Minimum 5m history | Current session 5m bars up to decision bar | Yes | `data_quality_failure` | data-quality rejection |
| ATR availability | Current/prior 5m OHLC true ranges | Yes | `missing_atr` | data-quality rejection |
| ATR compression floor | ATR-14 and rolling mean ATR-40 | Yes | `atr_below_threshold` | market-structure rejection |
| Asian range construction | 01:00-07:00 UTC 5m bars for session date | Yes | `missing_asian_range` | data-quality rejection |
| Sweep detection | Current 5m bar, Asian high/low, ATR-14 | Yes | `no_sweep_detected` | market-structure rejection |
| Sweep direction validation | Active sweep state | Yes | `invalid_sweep_direction` | market-structure rejection |
| Sweep depth | Wick pierce depth versus Asian range and ATR | Yes | `insufficient_sweep_depth` | market-structure rejection |
| ADX availability | 5m OHLC up to decision bar | Yes | `missing_adx` | data-quality rejection |
| ADX threshold | ADX-14 and configured minimum | Yes | `adx_below_threshold` | market-structure rejection |
| EMA trend/regime | Closed 1h bars strictly before current 5m timestamp | Yes | `ema_trend_filter_failed` | market-structure rejection |
| VWAP context | Same-session bars up to decision bar | Yes | `vwap_filter_failed` | market-structure rejection |
| FVG detection | Last 20 5m bars up to decision bar | Yes | `no_fvg_detected` | market-structure rejection |
| OB detection | Last 20 5m bars up to decision bar | Yes | `no_ob_detected` | market-structure rejection |
| Confluence classification | OB/FVG result for sweep direction | Yes | `confluence_failed` | market-structure rejection |
| Confluence allowlist | Configured allowed confluence types | Yes | `confluence_failed` | market-structure rejection |
| Entry trigger formation | Current close rounded to instrument tick | Yes | `invalid_entry_price` | data-quality rejection |
| Stop placement validation | ATR stop, instrument tick, direction | Yes | `invalid_stop_price` | risk rejection |
| Target placement validation | ATR target multiples, instrument tick, direction | Yes | `invalid_target_price` | risk rejection |
| Risk/reward validation | Entry, stop, target relation | Yes | `risk_reward_failed` | risk rejection |
| Duplicate/cooldown rules | Strategy state and existing signal state | Yes | `duplicate_signal`, `cooldown_active` | market-structure rejection |
| Position overlap gate | Open trade state in SessionSimulator/live feed | Yes | `position_overlap` | account rejection |
| Max position/trade cap | Session trade count and profile max trades | Yes | `max_positions_reached` | account rejection |
| TopStep daily loss guard | Same-day portfolio P&L and candidate initial risk | Yes | `topstep_daily_loss_guard` | account rejection |
| TopStep max loss guard | Account trailing drawdown floor and candidate initial risk | Yes | `topstep_max_loss_guard` | account rejection |
| Flatten window | Time-based live execution guard, if enabled | Yes | `flatten_window` | account rejection |

## Current Raw Setup Candidate Definition

A raw setup candidate is a London-session 5-minute decision bar that reaches the
signal engine with at least 15 5-minute bars available. The event records the
first gate that rejects the setup, or records `final_candidate` when all
market-structure gates pass and a bracket can be formed.

The ledger intentionally excludes MFE, MAE, realized P&L, exit reason, and
holding time from entry-time feature columns. Those fields may be populated only
after simulation as outcome diagnostics and must remain excluded from model
preprocessing.

## Known Observability Boundaries

The backtester currently avoids evaluating the signal engine while a trade is
already open or when the session trade cap has been reached, because doing so
would mutate the sweep detector and alter strategy behavior. Those gates remain
documented account/position gates, but raw market-structure events during
active-position periods are not emitted until a side-effect-free signal probe is
introduced.

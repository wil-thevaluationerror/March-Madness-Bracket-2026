# Handoff: Order Role Misclassification Bug — 2026-05-01

## Session Summary

Live trading session (NY, `topstep-50k-express`, live mode) was shut down cleanly at end of day.
A bug in the realtime order mapping was identified and patched during the post-session review.

---

## What Was Observed in the Logs

```
2026-05-01T12:11:02 INFO data_pipeline.live_feed  live_feed_signals count=1 new_bar_ts=2026-05-01 11:10:00-05:00
2026-05-01T12:11:02 INFO execution.topstep_live_adapter  topstep_realtime_order_learned order_id=entry-b5c7b65186 broker_order_id=2916363474 status=PENDING
2026-05-01T12:11:02 INFO execution.topstep_live_adapter  topstep_realtime_order_learned order_id=flatten-c3ae2412de broker_order_id=2916363528 status=PENDING
```

Followed by 12+ minutes of frozen account state with no balance change:

```
2026-05-01T12:11:53  account_state refreshed balance=49271.68 cushion_to_max_loss_limit=0.00
...
2026-05-01T12:23:39  account_state refreshed balance=49271.68 cushion_to_max_loss_limit=0.00
```

### Signals That Flagged the Issue

1. Both an entry and a flatten order appeared as `topstep_realtime_order_learned` at the exact same second a signal fired.
2. The balance never moved despite PENDING orders at the broker — no position was open.
3. `cushion_to_max_loss_limit=0.00` throughout (explained separately below).

---

## Root Cause

### Primary Bug — `_map_remote_order` role fallback

**File:** `execution/topstep_live_adapter.py`, `_map_remote_order()`

When the adapter encounters an order it has not seen before (`previous is None`), it was assigning `role="entry"` unconditionally:

```python
# BEFORE (line ~787)
role=previous.role if previous else "entry",
```

This fires in two real scenarios:

| Scenario | What happens |
|---|---|
| **Realtime race** | Broker pushes `GatewayUserOrder` via WebSocket before the HTTP `place_order` response stores the order locally. The realtime handler finds `previous=None` and mislabels the order. |
| **Stale session orders** | PENDING orders surviving from a previous session are re-delivered by the realtime stream on reconnect. The fresh session has no memory of them, so again `previous=None`. |

In both cases a `flatten-` or `stop-` order is stamped with `role="entry"`. When the fill eventually arrives, `on_fill` processes it as an entry — recording the wrong position direction and calling `_submit_or_update_children` instead of the exit path.

### Secondary (Non-Bug) — `cushion_to_max_loss_limit=0.00`

The TopStep API's `maximumDrawdown` field returns `null` for this account type. The adapter maps that to `0.0` via:

```python
cushion = float(account.get("maximumDrawdown") or 0.0)
```

This looks alarming in the logs but is **not a risk enforcement failure**. The cushion check in `execution_checks.py` already gates on `cushion > 0.0`:

```python
if not risk_engine.is_backtest_mode and cushion > 0.0 and estimated_risk > cushion:
    return ExecutionDecision(False, "max_loss_buffer_exceeded", 0, None)
```

Zero is treated as "field not populated — skip broker cushion check." The actual daily loss gate is `internal_daily_loss_budget_exceeded`, which uses internal P&L tracking.

**No action required on the cushion — it is working as designed.**

---

## Fix Applied

**File:** `execution/topstep_live_adapter.py`

Added a module-level helper that recovers the role from the order_id prefix, which already encodes intent:

| Order ID prefix | Role |
|---|---|
| `entry-{hex}` | `"entry"` |
| `stop-{hex}` | `"stop"` |
| `target-{hex}` | `"target"` |
| `flatten-{hex}` | `"flatten"` |

```python
# Added near module-level constants
_ROLE_PREFIXES = ("flatten", "stop", "target")

def _infer_role_from_order_id(order_id: str) -> str:
    for prefix in _ROLE_PREFIXES:
        if order_id.startswith(f"{prefix}-"):
            return prefix
    return "entry"
```

```python
# In _map_remote_order — AFTER
role=previous.role if previous else _infer_role_from_order_id(order_id),
```

### Verification

- All 77 existing execution tests pass after the change.
- Module imports cleanly.

---

## Open Questions / Follow-up

1. **Why did the PENDING orders never fill?** The orders sat for 12+ minutes without a fill or cancel event visible in the provided logs. Worth checking the full log around 12:11–12:23 for `order_rejected`, `order_canceled`, or missing fill events. Could be a limit price that was never reached, or the broker rejected the stale orders silently.

2. **Stale order cancellation at startup.** The current startup flow loads open broker orders via `_refresh_open_orders()` but does not cancel orders that have no matching internal state (i.e., no `customTag` recognized by the engine). Consider adding a startup step to cancel any `broker-{id}`-prefixed orders (orders learned with no custom tag = orders not placed by this codebase) before proceeding to reconciliation.

3. **Realtime race window.** The `topstep_realtime_order_learned` log for a freshly submitted order means the WebSocket beat the HTTP response. This is now handled correctly by role inference, but the "learned" log is still emitted — which can be confusing. Consider suppressing or downgrading the log when the order_id prefix is recognized as a local ID format.

---

## Session Shutdown

Trader was shut down via `SIGINT` to PID `50135` (screen session `topstep_ny`).
`safe_shutdown()` was invoked — any open position would have been flattened before exit.
Lock file: `runtime_logs/run_trader_live_topstep-50k-express.lock`

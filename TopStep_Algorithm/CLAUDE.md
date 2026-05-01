# CLAUDE.md — TopStep Algorithm: Invariants & Architecture Reference

This file is the authoritative reference for Claude Code and Codex contributors.
Read it before touching any file in this repository.
It is updated at every phase tag — never let it fall behind the code.

**Current phase:** `v0.1-stable-flat` (Phase 0 complete)
**Next phase:** Phase 1 — `src/trading_system/` package layout

---

## 1. Non-Negotiable Safety Invariants

These rules may never be broken by any PR, refactor, or feature addition:

1. **Strategies never touch the broker.** Models emit `OrderIntent` objects only.
   No strategy file may import from `execution/` or call any broker method directly.

2. **Every order flows through this exact pipeline — no shortcuts:**
   ```
   Strategy.generate_intents()
     → ExecutionEngine (intent validation, dedup, session check)
       → RiskEngine.validate_intent() / AccountGuard.check()   [Phase 4+]
         → OrderManager (construct BrokerOrder)
           → BrokerAdapter.place_order()
             → PaperBroker  OR  LiveTopstepAdapter
   ```

3. **Live mode requires explicit multi-factor activation:**
   - CLI flag `--mode live` (or `mode: live` in config)
   - Environment variable `LIVE_TRADING_CONFIRMED=true` in `.env`
   - Interactive typed confirmation token at startup (Phase 6+)
   - If any factor is absent, startup raises and exits.

4. **Kill switch is always checked.** The in-memory `RiskState.kill_switch` is checked
   before every order submission in `ExecutionEngine`. In Phase 5+, a file-backed
   `KillSwitch` at `runtime_logs/KILL_SWITCH` is also checked every tick.
   Operators can halt trading without restarting: `touch runtime_logs/KILL_SWITCH`.

5. **StateStore writes are atomic.** All writes go through `tempfile.mkstemp` +
   `Path.replace()` in `execution/state_store.py`. Never write state directly to the
   target path.

6. **No credentials in source code or YAML files.** All secrets are injected at runtime
   from a `.env` file located OUTSIDE the repository root.

7. **One active model per instrument per account (v1).** Two models may not trade the
   same symbol simultaneously. Enforced by `AccountGuard` instrument lock in Phase 4+.

8. **Default mode is safe.** All scripts and configs default to `backtest` or `paper`.
   Live mode cannot be triggered accidentally.

---

## 2. Current Architecture (v0.1-stable-flat)

### Module map

```
TopStep_Algorithm/              ← repo root; also Python path root (sys.path hack in scripts/)
│
├── config.py                   ← re-exports TraderConfig, build_config, profile constants
│                                  (delegates to backtest/config.py for dataclasses)
├── profiles.py                 ← imperative profile builders; mutates TraderConfig instances
│
├── models/
│   ├── orders.py               ← ALL domain types: OrderIntent, BrokerOrder, Fill,
│   │                              PositionSnapshot, Side, Regime, TradingMode,
│   │                              KillSwitchState, OrderState, ExecutionReport, …
│   └── instruments.py          ← InstrumentSpec, resolve_instrument()
│
├── execution/
│   ├── engine.py               ← ExecutionEngine — central coordinator; calls RiskEngine,
│   │                              OrderManager, BrokerAdapter, StateStore, Reconciler
│   ├── order_manager.py        ← ManagedOrderChain, OrderManager — order lifecycle
│   ├── broker.py               ← BrokerAdapter (abstract protocol)
│   ├── topstepx_adapter.py     ← TopstepXAdapter (HTTP REST, used in paper mode)
│   ├── topstep_live_adapter.py ← LiveTopstepAdapter (HTTP + SignalR WebSocket, live mode)
│   │                              CONTAINS: _infer_role_from_order_id() [fix 2026-05-01]
│   ├── topstepx_adapter.py     ← TopstepXAdapter (paper HTTP adapter)
│   ├── reconciler.py           ← Reconciler — broker vs local state check at startup
│   ├── state_store.py          ← StateStore — JSON-backed, atomic writes via tempfile
│   ├── scheduler.py            ← SessionScheduler — session window checks, flatten time
│   └── logging.py              ← EventLogger — structured JSONL event log
│
├── risk/
│   ├── engine.py               ← RiskEngine + RiskState — stateful quantitative risk checks
│   └── execution_checks.py     ← validate_intent() — 280-line free function; checks
│                                  stacking, cluster risk, cooldown, loss limits, etc.
│
├── strategy/
│   ├── rules.py                ← generate_intents(), build_order_intent(), SignalInput
│   ├── signal.py               ← SignalEngine — sweep/reentry signal generation
│   ├── sweep_detector.py       ← SweepDetector — liquidity sweep detection
│   ├── asian_range.py          ← AsianRangeCalculator — overnight range computation
│   ├── confluence.py           ← ConfluenceScorer — multi-factor signal scoring
│   ├── intent_bridge.py        ← IntentBridge — signal-to-intent conversion for Model B
│   └── diagnostics.py          ← StrategyDiagnosticsLogger
│
├── features/
│   └── indicators.py           ← EMA, VWAP, ATR, ADX, volume metrics (pure functions)
│
├── backtest/
│   ├── config.py               ← AUTHORITATIVE source of all config dataclasses
│   │                              (TraderConfig, SessionConfig, RiskLimits,
│   │                               ExecutionConfig, StrategyConfig, TopstepConnectionConfig)
│   ├── engine.py               ← BacktestEngine — historical bar simulation
│   ├── simulator.py            ← BarSimulator — tick-by-tick simulation loop
│   ├── metrics.py              ← BacktestMetrics — Sharpe, drawdown, win rate, etc.
│   ├── walk_forward.py         ← WalkForwardRunner — IS/OOS window validation
│   ├── data_loader.py          ← DataLoader — Databento DBN file reader
│   ├── signal_ledger.py        ← SignalLedger — per-bar signal audit trail
│   ├── raw_setup_ledger.py     ← RawSetupLedger — raw filter state per bar
│   ├── reporter.py             ← BacktestReporter — HTML dashboard + JSON summary
│   ├── feature_importance.py   ← FeatureImportanceAnalyzer
│   ├── run_backtest.py         ← CLI entry point for backtests
│   └── dashboard.py            ← build_dashboard_payload()
│
├── data_pipeline/
│   ├── live_feed.py            ← TopstepLiveFeed — Databento live market data (Model A)
│   ├── sweep_live_feed.py      ← SweepLiveFeed — live feed for London sweep Model B
│   ├── loader.py               ← historical bar loader
│   └── preprocess.py           ← bar preprocessing utilities
│
├── api/
│   └── market_data.py          ← market data helpers
│
├── scripts/
│   └── run_trader.py           ← CURRENT LIVE ENTRY POINT
│                                  Adds PROJECT_ROOT to sys.path (temporary hack)
│                                  Handles --profile, --mode, --enforce-live-rules flags
│                                  Creates per-profile lock file in runtime_logs/
│                                  Dispatches to TopstepLiveFeed or SweepLiveFeed
│                                  Calls ExecutionEngine.on_bar() in the main loop
│
└── tests/
    ├── test_execution.py
    ├── test_gaps_1_4.py
    ├── test_intent_bridge.py
    ├── test_ml_diagnostics.py
    ├── test_strategy_diagnostics.py
    └── test_walk_forward_p1.py
```

### Current config inheritance chain

```
backtest/config.py          ← defines all frozen dataclasses
    ↑ re-exported by
config.py (root)            ← adds PROFILE_* constants, available_profiles(), build_config()
    ↑ imported by
profiles.py                 ← build_config() + _apply_*() mutators for each profile
    ↑ imported by
scripts/run_trader.py       ← calls build_config(profile) to get a TraderConfig
```

**In Phase 2**, `backtest/config.py` and root `config.py` will merge into
`trading_system/config.py` (Pydantic v2) and profile logic will move to YAML overlays.

---

## 3. Current Profiles

| Profile constant | Symbol | Session | Mode |
|---|---|---|---|
| `PROFILE_TOPSTEP_50K_EXPRESS` | MES | 09:30–13:30 ET | live |
| `PROFILE_TOPSTEP_50K_EXPRESS_LONDON` | MES | 09:30–13:30 ET | live (legacy) |
| `PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6B_PAPER` | 6B | 02:00–07:00 ET | paper only |
| `PROFILE_TOPSTEP_50K_EXPRESS_LONDON_6E_PAPER` | 6E | 02:00–07:00 ET | paper only |

6B and 6E are blocked from live mode via `_LIVE_BLOCKED_SYMBOLS` in
`execution/topstep_live_adapter.py`. Do not remove this guard.

---

## 4. Key Runtime Files — Handle with Care

| File | Why it is sensitive |
|---|---|
| `execution/topstep_live_adapter.py` | Talks to live broker; contains `_infer_role_from_order_id` fix (2026-05-01) |
| `execution/engine.py` | Central coordinator; any change here affects all models |
| `risk/execution_checks.py` | `validate_intent()` — all quantitative risk gates live here |
| `risk/engine.py` | `RiskState` is the in-memory risk ledger; resets on restart |
| `execution/state_store.py` | Atomic JSON writes; do not change write pattern |
| `execution/reconciler.py` | Startup broker/local state reconciliation |
| `scripts/run_trader.py` | Live entry point; always keep it runnable |

---

## 5. What Is Safe to Add Without Breaking Anything

- New files under `strategy/` — signal generators, filters
- New files under `features/` — indicator functions (pure, no side effects)
- New files under `backtest/` — analysis tools, reporters
- New test files under `tests/`
- New YAML configs (after Phase 2)
- New model directories under `trading_system/models/` (after Phase 3)

---

## 6. What Requires a Full Test Run Before Merging

Any change to:
- `execution/engine.py`
- `execution/topstep_live_adapter.py`
- `risk/engine.py`
- `risk/execution_checks.py`
- `execution/order_manager.py`
- `execution/reconciler.py`
- `config.py` / `backtest/config.py` / `profiles.py`

Run: `python -m pytest tests/ -x` before committing.

---

## 7. Phase Roadmap

| Tag | Phase | What changes |
|---|---|---|
| `v0.1-stable-flat` | 0 — Pre-flight | `pyproject.toml`, `CLAUDE.md`, `pip install -e .` |
| `v0.2-src-layout` | 1 — Package layout | `src/trading_system/`, `core/` replaces `models/` |
| `v0.3-pydantic-config` | 2 — Pydantic configs | YAML config files, `config_loader.py`, Pydantic v2 |
| `v0.4-strategy-registry` | 3 — Strategy base | `Strategy` ABC, `registry.py`, model_a/model_b plug-ins |
| `v0.5-account-guard` | 4 — AccountGuard | 12-check single choke point wired into ExecutionEngine |
| `v0.6-killswitch-watchdog` | 5 — KillSwitch/Watchdog | File-backed kill switch, periodic watchdog |
| `v0.7-session-router` | 6 — SessionRouter | Router as sole live entry point, ModelRunner per model |
| `v0.8-ci` | 7+8 — CI | import-linter contracts, GitHub Actions |
| `v0.9-regression-baselines` | 9 — Regression | Committed baseline JSON, regression test job |

---

## 8. How to Add a New Trading Model (Phase 3+ steady state)

1. `cp config/models/model_a.yaml config/models/model_c.yaml` — fill in session, instrument, params
2. Create `src/trading_system/models/model_c/` with `__init__.py` and `strategy.py`
3. Implement `class ModelC(Strategy)` and decorate with `@register("model_c")`
4. Run `python scripts/generate_regression_baseline.py --model model_c` — commit baseline JSON
5. Run `lint-imports` — confirm no cross-model imports
6. Run `pytest tests/` — all green
7. PR → `develop` → CI → merge

The `ExecutionEngine`, `AccountGuard`, `SessionRouter`, `RiskEngine`, and all shared
infrastructure require zero changes to add a new model.

---

## 9. Secrets and Credentials

**Never commit:**
- `.env` (actual secrets)
- `*.key`, `*.pem`
- Any file containing `TOPSTEP_PASSWORD`, `TOPSTEP_USERNAME`, `TOPSTEP_API_KEY`

**Always use:**
- `.env.example` (key names only, no values) — committed
- `python-dotenv` loads `.env` from PROJECT_ROOT at startup

**Required env vars** (see `.env.example`):
```
TOPSTEP_USERNAME
TOPSTEP_PASSWORD
TOPSTEP_API_BASE_URL
TOPSTEP_WEBSOCKET_URL
TOPSTEP_ACCOUNT_ID
LIVE_TRADING_CONFIRMED        # must be "true" to enable live mode
```

---

## 10. Known Issues / Open Items (as of 2026-05-01)

1. **`run_trader.py` uses sys.path hack** — `PROJECT_ROOT` added to `sys.path` at line 13.
   Fixed in Phase 1 when `pip install -e .` makes it unnecessary.

2. **`backtest/config.py` is the true source of config dataclasses**, not `config.py` at root.
   Root `config.py` re-exports from it. Unified in Phase 2.

3. **`RiskState.kill_switch` is in-memory only** — resets on restart.
   File-backed `KillSwitch` added in Phase 5.

4. **`StateStore` is JSON, not SQLite** — aspirational SQLite upgrade is post-Phase 6.

5. **Order role bug (fixed 2026-05-01)** — `_infer_role_from_order_id()` added to
   `topstep_live_adapter.py`. See `docs/handoff_order_role_bug_2026-05-01.md`.

6. **Stale order cancellation at startup** — open items in handoff doc section 2.
   PENDING orders with no matching internal state are not yet auto-cancelled.

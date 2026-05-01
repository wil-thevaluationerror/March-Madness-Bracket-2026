# TopStep Model Context Report

## Overview

The TopStep system is an automated futures trading research and execution project built around TopStep/TopstepX account constraints. Its purpose is to evaluate, backtest, and run rule-based trading models that generate bracketed trade intents, route them through a risk engine, and submit them through mock, paper, or live execution adapters.

The project is not just a signal script. It includes market-data ingestion, strategy logic, walk-forward backtesting, risk management, order lifecycle handling, TopstepX connectivity, reconciliation, diagnostics, and runtime safeguards.

## Trading Profiles

The active configuration is driven by named profiles in `profiles.py`.

- `topstep-50k-express`: MES-focused New York session profile.
- `topstep-50k-express-london`: London-session variant using the shared TopStep 50K Express tuning.
- `topstep-50k-express-london-6b-paper`: paper-mode British Pound futures profile.
- `topstep-50k-express-london-6e-paper`: paper-mode Euro FX futures profile.

The shared TopStep 50K Express profile is tuned around the account structure: a $2,000 trailing max drawdown and a $3,000 profit target. The MES version uses larger size with drawdown-based risk tiers, while the 6B and 6E London profiles are limited to one contract and are explicitly blocked from live execution until verified.

## Strategy Logic

There are two main strategy paths.

The NY/MES path uses an EMA/VWAP breakout model. It looks for price breaking above or below recent structure while aligned with VWAP and EMA trend. It filters out weak or extended entries using ATR, ADX, EMA persistence, volume, signal score, and duplicate/cooldown rules before producing an `OrderIntent`.

The London profiles use a liquidity-sweep model. This model builds an Asian range from overnight price action, then waits during the London window for price to sweep the Asian high or low. A valid trade requires:

- enough 5-minute and 1-hour bar history;
- ATR availability and sufficient volatility;
- a detected Asian-range sweep;
- ADX momentum above the configured threshold;
- 1-hour EMA trend alignment and persistence;
- order-block or fair-value-gap confluence;
- valid bracket prices with stop and target placement.

Accepted London sweep trades are converted into bracket-style trade intents with an entry, stop, and profit target. The current profile defaults to a 2:1 target multiple and disables breakeven stops because walk-forward testing showed breakeven behavior harmed win rate.

## Risk Controls

The risk engine is designed around funded-account survival first. It tracks realized P&L, equity peak, open positions, trade count, consecutive losses, cooldowns, active intents, API errors, and kill-switch state.

Important safeguards include:

- one active position by default;
- maximum trades per day;
- daily loss lockout;
- consecutive-loss cooldown;
- trailing drawdown kill switch with a buffer before the TopStep limit;
- drawdown-based position-size reduction;
- slippage checks;
- stale order cancellation;
- reconciliation between broker state and internal state;
- force-flatten behavior near session end;
- API-error kill switch.

For the TopStep 50K Express calibration, the internal daily loss limit is set below the platform drawdown limit, and risk tiers reduce size as drawdown deepens.

## Runtime Flow

The main runtime entry point is `scripts/run_trader.py`.

At startup, the system builds the active profile, injects TopstepX credentials from environment variables, creates the risk engine, adapter, order manager, event logger, and execution engine, then reconciles internal state against broker state. In paper or live mode, it creates either a standard Topstep live feed or the London sweep feed depending on the profile.

During the loop, the feed polls for new market data, emits trade intents when the strategy passes all gates, and sends those intents to the execution engine. The execution engine validates each intent against account state and risk limits, submits the order plan, listens for fills or rejects, manages child stop/target orders, reconciles state, and persists runtime state to logs.

## Backtesting And Diagnostics

The backtest stack supports walk-forward testing using historical 1-minute and 1-hour data. It can load profile-specific strategy settings so research results match the intended live or paper runtime configuration.

The project also includes diagnostics around signal rejection stages. The London sweep pipeline records why a candidate did or did not become a final trade candidate, separating data-quality rejections, market-structure rejections, risk rejections, and account-level rejections. This makes it easier to understand whether the model is failing because of poor market structure, missing data, risk controls, or session/account constraints.

## Current Practical Role

In practical terms, this system is a TopStep-oriented futures trading framework. It attempts to find high-quality intraday breakout or liquidity-sweep setups, size and filter them according to funded-account constraints, execute them through a controlled order engine, and provide enough diagnostics to continue improving the model without guessing.

The London 6B and 6E variants currently look like paper/research profiles, while MES is the primary TopStep Express profile with more complete sizing and risk calibration.

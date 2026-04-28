"""
Connectivity diagnostic — tests auth, account lookup, and contract resolution.
Run with: python scripts/check_connection.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from backtest.config import TopstepConnectionConfig
from execution.topstep_live_adapter import LiveTopstepAdapter, UrlLibTopstepTransport
from models.orders import TradingMode


def build_config() -> TopstepConnectionConfig:
    cfg = TopstepConnectionConfig()
    mapping = {
        "TOPSTEP_USERNAME": "username",
        "TOPSTEP_API_KEY": "api_key",
        "TOPSTEP_ACCOUNT_ID": "account_id",
        "TOPSTEP_ENVIRONMENT": "environment",
        "TOPSTEP_API_BASE_URL": "api_base_url",
        "TOPSTEP_WEBSOCKET_URL": "websocket_url",
    }
    for env_var, field in mapping.items():
        value = os.environ.get(env_var, "").strip()
        if value:
            setattr(cfg, field, value)
    return cfg


def main() -> None:
    cfg = build_config()

    missing = cfg.missing_required_fields()
    if missing:
        print(f"[FAIL] Missing credentials: {', '.join(missing)}")
        print("       Copy .env.example to .env and fill in the values.")
        sys.exit(1)

    print(f"[INFO] Username   : {cfg.username}")
    print(f"[INFO] Account ID : {cfg.account_id}")
    print(f"[INFO] API URL    : {cfg.api_base_url}")
    print(f"[INFO] WS URL     : {cfg.websocket_url}")
    print(f"[INFO] Environment: {cfg.environment}")
    print()

    transport = UrlLibTopstepTransport()

    # Step 1: Authenticate
    print("[STEP 1] Authenticating...")
    try:
        response = transport.post(
            cfg.api_base_url,
            "/api/Auth/loginKey",
            {"userName": cfg.username, "apiKey": cfg.api_key},
            timeout_seconds=cfg.request_timeout_seconds,
        )
    except RuntimeError as exc:
        print(f"[FAIL] Auth request failed: {exc}")
        sys.exit(1)

    token = response.get("token") or response.get("jwtToken")
    if not token:
        print(f"[FAIL] Auth succeeded but no token in response.")
        print(f"       Full response: {response}")
        sys.exit(1)
    print(f"[OK]   Token received (first 20 chars): {str(token)[:20]}...")

    # Step 2: List accounts
    print("\n[STEP 2] Fetching accounts...")
    try:
        acct_response = transport.post(
            cfg.api_base_url,
            "/api/Account/search",
            {"onlyActiveAccounts": True},
            bearer_token=token,
            timeout_seconds=cfg.request_timeout_seconds,
        )
    except RuntimeError as exc:
        print(f"[FAIL] Account fetch failed: {exc}")
        sys.exit(1)

    accounts = list(acct_response.get("accounts") or [])
    if not accounts:
        print(f"[WARN] No accounts returned. Full response: {acct_response}")
    else:
        print(f"[OK]   Found {len(accounts)} account(s):")
        for a in accounts:
            can_trade = a.get("canTrade", "?")
            print(f"       id={a.get('id')}  name={a.get('name')}  balance={a.get('balance')}  canTrade={can_trade}")

    # Step 3: Verify our account_id matches one of them
    print(f"\n[STEP 3] Matching configured account_id='{cfg.account_id}'...")
    desired = str(cfg.account_id).strip()
    matched = None
    for a in accounts:
        if str(a.get("id")) == desired or str(a.get("name") or "").strip() == desired:
            matched = a
            break
    if matched is None:
        print(f"[WARN]  No account matched '{desired}'. Check TOPSTEP_ACCOUNT_ID in .env.")
    else:
        print(f"[OK]   Matched: id={matched.get('id')}  name={matched.get('name')}  canTrade={matched.get('canTrade')}")

    # Step 4: Contract resolution for MES
    print("\n[STEP 4] Resolving MES contract...")
    try:
        contract_response = transport.post(
            cfg.api_base_url,
            "/api/Contract/search",
            {"live": cfg.environment == "live", "searchText": "MES"},
            bearer_token=token,
            timeout_seconds=cfg.request_timeout_seconds,
        )
    except RuntimeError as exc:
        print(f"[FAIL] Contract search failed: {exc}")
        sys.exit(1)

    contracts = list(contract_response.get("contracts") or [])
    if not contracts:
        print(f"[WARN] No MES contracts found. Full response: {contract_response}")
    else:
        active = [c for c in contracts if c.get("activeContract", True)]
        print(f"[OK]   Found {len(contracts)} MES contract(s), {len(active)} active:")
        for c in active[:3]:
            print(f"       id={c.get('id')}  name={c.get('name')}  active={c.get('activeContract')}")

    account_id = matched.get("id") if matched is not None else cfg.account_id

    # Step 5: Verify open positions
    print("\n[STEP 5] Checking open positions...")
    try:
        position_response = transport.post(
            cfg.api_base_url,
            "/api/Position/searchOpen",
            {"accountId": int(account_id)},
            bearer_token=token,
            timeout_seconds=cfg.request_timeout_seconds,
        )
    except RuntimeError as exc:
        print(f"[FAIL] Position fetch failed: {exc}")
        sys.exit(1)

    positions = list(position_response.get("positions") or [])
    if not positions:
        print("[OK]   No open positions.")
    else:
        print(f"[WARN] Found {len(positions)} open position(s):")
        for p in positions:
            print(f"       contractId={p.get('contractId')}  type={p.get('type')}  size={p.get('size')}")

    # Step 6: Verify open orders
    print("\n[STEP 6] Checking open orders...")
    try:
        order_response = transport.post(
            cfg.api_base_url,
            "/api/Order/searchOpen",
            {"accountId": int(account_id)},
            bearer_token=token,
            timeout_seconds=cfg.request_timeout_seconds,
        )
    except RuntimeError as exc:
        print(f"[FAIL] Open order fetch failed: {exc}")
        sys.exit(1)

    orders = list(order_response.get("orders") or [])
    if not orders:
        print("[OK]   No open orders.")
    else:
        print(f"[WARN] Found {len(orders)} open order(s):")
        for o in orders:
            print(
                f"       id={o.get('id')}  contractId={o.get('contractId')}  "
                f"type={o.get('type')}  side={o.get('side')}  size={o.get('size')}"
            )

    print("\n[DONE] Connectivity check complete.")


if __name__ == "__main__":
    main()

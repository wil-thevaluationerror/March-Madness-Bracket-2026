"""
Emergency open-order cancel utility for TopstepX.

Run with: python scripts/cancel_open_orders.py
"""
from __future__ import annotations

import os
import sys
import time
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
from execution.topstep_live_adapter import UrlLibTopstepTransport


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


def post(transport, cfg, path, payload, token):
    return transport.post(
        cfg.api_base_url,
        path,
        payload,
        bearer_token=token,
        timeout_seconds=cfg.request_timeout_seconds,
    )


def main() -> None:
    cfg = build_config()
    missing = cfg.missing_required_fields()
    if missing:
        raise SystemExit(f"Missing credentials: {', '.join(missing)}")

    transport = UrlLibTopstepTransport()
    auth = transport.post(
        cfg.api_base_url,
        "/api/Auth/loginKey",
        {"userName": cfg.username, "apiKey": cfg.api_key},
        timeout_seconds=cfg.request_timeout_seconds,
    )
    token = auth.get("token") or auth.get("jwtToken")
    if not token:
        raise SystemExit(f"Auth succeeded without token: {auth}")

    accounts_response = post(transport, cfg, "/api/Account/search", {"onlyActiveAccounts": True}, token)
    desired = str(cfg.account_id).strip()
    account = None
    for candidate in list(accounts_response.get("accounts") or []):
        if str(candidate.get("id")) == desired or str(candidate.get("name") or "").strip() == desired:
            account = candidate
            break
    if account is None:
        raise SystemExit(f"Account not found: {cfg.account_id}")
    if account.get("canTrade") is False:
        raise SystemExit(f"Account cannot trade: {account.get('id')}")

    account_id = int(account["id"])
    open_response = post(transport, cfg, "/api/Order/searchOpen", {"accountId": account_id}, token)
    orders = list(open_response.get("orders") or [])
    if not orders:
        print("[OK] No open orders to cancel.")
        return

    print(f"[INFO] Canceling {len(orders)} open order(s) on account {account_id}...")
    for order in orders:
        order_id = int(order["id"])
        response = post(
            transport,
            cfg,
            "/api/Order/cancel",
            {"accountId": account_id, "orderId": order_id},
            token,
        )
        print(
            f"[OK] Cancel requested id={order_id} contractId={order.get('contractId')} "
            f"type={order.get('type')} side={order.get('side')} size={order.get('size')} response={response}"
        )

    time.sleep(1.0)
    verify = post(transport, cfg, "/api/Order/searchOpen", {"accountId": account_id}, token)
    remaining = list(verify.get("orders") or [])
    if remaining:
        print(f"[WARN] Orders still open after cancel attempt: {remaining}")
        raise SystemExit(2)
    print("[OK] No open orders remain.")


if __name__ == "__main__":
    main()

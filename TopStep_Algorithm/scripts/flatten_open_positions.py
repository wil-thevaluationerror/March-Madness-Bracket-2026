"""
Emergency flatten utility for TopstepX.

Run with: python scripts/flatten_open_positions.py
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
    positions_response = post(transport, cfg, "/api/Position/searchOpen", {"accountId": account_id}, token)
    positions = list(positions_response.get("positions") or [])
    if not positions:
        print("[OK] No open positions to flatten.")
        return

    print(f"[INFO] Flattening {len(positions)} open position(s) on account {account_id}...")
    for position in positions:
        contract_id = str(position["contractId"])
        size = int(position.get("size") or 0)
        position_type = int(position.get("type") or 0)
        if size <= 0:
            continue
        # Topstep position type: 1 = long, 2 = short. Order side: 0 = buy, 1 = sell.
        side = 1 if position_type == 1 else 0
        response = post(
            transport,
            cfg,
            "/api/Order/place",
            {
                "accountId": account_id,
                "contractId": contract_id,
                "type": 2,
                "side": side,
                "size": size,
                "customTag": f"emergency-flatten-{int(time.time())}",
            },
            token,
        )
        print(f"[OK] Submitted flatten order contractId={contract_id} size={size} response={response}")

    time.sleep(1.0)
    verify = post(transport, cfg, "/api/Position/searchOpen", {"accountId": account_id}, token)
    remaining = list(verify.get("positions") or [])
    if remaining:
        print(f"[WARN] Positions still open after flatten attempt: {remaining}")
        raise SystemExit(2)
    print("[OK] Account is flat.")


if __name__ == "__main__":
    main()

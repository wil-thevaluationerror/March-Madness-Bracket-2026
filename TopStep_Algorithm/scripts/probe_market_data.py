"""
Probe TopstepX API for bar history endpoints.
Run: python scripts/probe_market_data.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime, timedelta
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


def build_cfg() -> TopstepConnectionConfig:
    cfg = TopstepConnectionConfig()
    for env_var, field in {
        "TOPSTEP_USERNAME": "username",
        "TOPSTEP_API_KEY": "api_key",
        "TOPSTEP_ACCOUNT_ID": "account_id",
        "TOPSTEP_ENVIRONMENT": "environment",
        "TOPSTEP_API_BASE_URL": "api_base_url",
        "TOPSTEP_WEBSOCKET_URL": "websocket_url",
    }.items():
        value = os.environ.get(env_var, "").strip()
        if value:
            setattr(cfg, field, value)
    return cfg


def main() -> None:
    cfg = build_cfg()
    transport = UrlLibTopstepTransport()

    # Auth
    print("[STEP 1] Authenticating...")
    resp = transport.post(cfg.api_base_url, "/api/Auth/loginKey",
                          {"userName": cfg.username, "apiKey": cfg.api_key})
    token = resp.get("token") or resp.get("jwtToken")
    if not token:
        print(f"[FAIL] Auth: {resp}")
        sys.exit(1)
    print(f"[OK] Token: {str(token)[:20]}...")

    # Contract ID
    contract_id = "CON.F.US.MES.M26"
    now = datetime.now(UTC)
    start = now - timedelta(hours=2)

    # Probe candidate bar-history endpoints
    candidates = [
        ("/api/History/retrieveBars", {
            "contractId": contract_id,
            "live": False,
            "startTime": start.isoformat(),
            "endTime": now.isoformat(),
            "unit": 2,       # 2 = minute (common ProjectX convention)
            "unitNumber": 1,
            "limit": 10,
            "includePartialBar": False,
        }),
        ("/api/History/retrieveBars", {
            "contractId": contract_id,
            "live": False,
            "startTime": start.isoformat(),
            "endTime": now.isoformat(),
            "barType": "Minute",
            "barInterval": 1,
            "limit": 10,
        }),
        ("/api/MarketData/getBars", {
            "contractId": contract_id,
            "startTime": start.isoformat(),
            "endTime": now.isoformat(),
            "barType": 2,
            "barInterval": 1,
            "limit": 10,
        }),
        ("/api/MarketData/bars", {
            "contractId": contract_id,
            "startTime": start.isoformat(),
            "endTime": now.isoformat(),
            "resolution": "1m",
            "limit": 10,
        }),
    ]

    for path, payload in candidates:
        print(f"\n[PROBE] POST {path}")
        print(f"        payload: {json.dumps(payload, default=str)}")
        try:
            result = transport.post(cfg.api_base_url, path, payload, bearer_token=token)
            print(f"        response keys: {list(result.keys())}")
            # Find the bars list in the response
            for key, val in result.items():
                if isinstance(val, list) and val:
                    print(f"        '{key}' has {len(val)} items. First item keys: {list(val[0].keys()) if isinstance(val[0], dict) else type(val[0])}")
                    print(f"        First bar: {json.dumps(val[0], default=str)}")
                    break
            else:
                print(f"        full response: {json.dumps(result, default=str)[:500]}")
        except RuntimeError as exc:
            print(f"        [ERROR] {exc}")


if __name__ == "__main__":
    main()

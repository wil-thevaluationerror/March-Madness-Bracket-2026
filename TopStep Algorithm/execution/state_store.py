from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class StateStore:
    def __init__(self, base_dir: str, filename: str = "engine_state.json") -> None:
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.path = self.base_path / filename

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, payload: dict[str, Any]) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
        tmp_path.replace(self.path)

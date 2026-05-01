from __future__ import annotations

import json
import os
import tempfile
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
        fd, tmp_name = tempfile.mkstemp(
            dir=self.base_path,
            prefix=f"{self.path.stem}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, indent=2)
            tmp_path.replace(self.path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

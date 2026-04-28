from __future__ import annotations

from pathlib import Path

import databento as db
import pandas as pd


def load_dbn(path: str | Path) -> pd.DataFrame:
    data = db.read_dbn(str(path))
    return data.to_df().reset_index()

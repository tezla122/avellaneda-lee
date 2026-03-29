from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

def expanding_folds(index: pd.DatetimeIndex, *, min_train_days: int, test_days: int, step_days: int) -> list[WalkForwardFold]:
    idx = pd.DatetimeIndex(index).sort_values()
    folds: list[WalkForwardFold] = []
    train_end = min_train_days - 1
    while True:
        test_start = train_end + 1
        test_end = test_start + test_days - 1
        if test_end >= len(idx):
            break
        folds.append(WalkForwardFold(train_start=idx[0], train_end=idx[train_end], test_start=idx[test_start], test_end=idx[test_end]))
        train_end += step_days
    return folds

def concat_oos_pnl(series_list: list[pd.Series]) -> pd.Series:
    return pd.concat(series_list, axis=0).sort_index()

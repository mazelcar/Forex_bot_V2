import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    d = df.copy()
    d['H-L'] = d['high'] - d['low']
    d['H-C'] = abs(d['high'] - d['close'].shift(1))
    d['L-C'] = abs(d['low'] - d['close'].shift(1))
    d['TR'] = d[['H-L','H-C','L-C']].max(axis=1)
    return d['TR'].ewm(span=period, adjust=False).mean()


"""Shared CSV validation for batch scoring and notebook model analysis."""

from io import StringIO
import pandas as pd
from app.domain.schema import FEATURES, Transaction


def load_csv(text, labelled=False):
    if len(text) > 1_000_000:
        raise ValueError("CSV exceeds 1 MB")
    try:
        frame = pd.read_csv(StringIO(text))
    except Exception as exc:
        raise ValueError("Invalid CSV") from exc
    if not 1 <= len(frame) <= 10000:
        raise ValueError("CSV must contain 1 to 10,000 rows")
    required = FEATURES + (["fraud"] if labelled else [])
    if not set(required) <= set(frame.columns):
        raise ValueError("Required columns: " + ", ".join(required))
    records = []
    for index, row in frame.iterrows():
        try:
            records.append(Transaction(**row[FEATURES].to_dict()).model_dump())
        except Exception as exc:
            raise ValueError(f"Invalid transaction at row {index+2}") from exc
    x = pd.DataFrame(records, columns=FEATURES)
    if not labelled:
        return x
    y = frame.fraud
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError(
            "fraud must contain both binary classes, with no missing values"
        )
    if y.value_counts().min() < 10:
        raise ValueError(
            "At least 10 samples per class are required for stratified splits"
        )
    return x, y.astype(int)

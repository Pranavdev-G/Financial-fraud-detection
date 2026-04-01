import hashlib
from datetime import datetime

import numpy as np
import pandas as pd

_df: pd.DataFrame | None = None
_raw_columns: list[str] = []
_column_map: dict[str, str] = {}

CITY_POOL = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune",
    "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
]

COLUMN_KEYWORDS = {
    "amount": ["amount", "amt", "value", "transaction_amount", "txn_amount", "money", "total"],
    "fraud": ["fraud", "is_fraud", "label", "target", "class", "fraud_flag"],
    "sender": ["sender", "nameorig", "from", "payer", "customer", "user", "account_id", "from_account"],
    "receiver": ["receiver", "namedest", "to", "payee", "merchant", "beneficiary", "to_account"],
    "payment_method": ["payment_method", "type", "transaction_type", "method", "channel", "mode"],
    "balance_before": ["oldbalanceorg", "old_balance", "balance_before", "prev_balance", "opening_balance"],
    "balance_after": ["newbalanceorig", "new_balance", "balance_after", "closing_balance"],
    "timestamp": ["timestamp", "time", "datetime", "transaction_time", "date", "step"],
    "location": ["location", "city", "state", "branch", "merchant_city", "geo", "ip_location"],
}


def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _find_column(df_cols, keyword_list):
    lower_map = {c.lower().replace(" ", "_").replace("-", "_"): c for c in df_cols}
    for kw in keyword_list:
        kw_norm = kw.lower().replace(" ", "_").replace("-", "_")
        if kw_norm in lower_map:
            return lower_map[kw_norm]
        for lc, orig in lower_map.items():
            if kw_norm in lc or lc in kw_norm:
                return orig
    return None


def _auto_map_columns(df):
    mapping = {}
    used = set()
    for std_name, keywords in COLUMN_KEYWORDS.items():
        found = _find_column(df.columns, keywords)
        if found and found not in used:
            mapping[found] = std_name
            used.add(found)
    return mapping


def _ensure_timestamp(df):
    df = df.copy()
    if "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed.notna().sum() > 0:
            fallback = pd.date_range("2025-01-01 08:00:00", periods=len(df), freq="20min")
            df["timestamp"] = parsed.fillna(pd.Series(fallback, index=df.index))
            return df

        numeric = pd.to_numeric(df["timestamp"], errors="coerce")
        if numeric.notna().sum() > 0:
            base = pd.Timestamp("2025-01-01 08:00:00")
            df["timestamp"] = base + pd.to_timedelta(numeric.fillna(0), unit="h")
            return df

    base_time = pd.Timestamp("2025-01-01 08:00:00")
    offsets = [(idx * 17) + (_stable_hash(str(sender)) % 7) for idx, sender in enumerate(df.get("sender", pd.Series(range(len(df)))))]
    df["timestamp"] = [base_time + pd.Timedelta(minutes=int(offset)) for offset in offsets]
    return df


def _ensure_location(df):
    df = df.copy()
    if "location" in df.columns:
        df["location"] = df["location"].astype(str).replace({"nan": "UNKNOWN"}).fillna("UNKNOWN")
        return df

    locations = []
    for idx, row in df.iterrows():
        sender = str(row.get("sender", "UNKNOWN"))
        receiver = str(row.get("receiver", "UNKNOWN"))
        seed = _stable_hash(f"{sender}-{receiver}-{idx}")
        base_city = CITY_POOL[seed % len(CITY_POOL)]
        if idx % 9 == 0:
            base_city = CITY_POOL[(seed + 3) % len(CITY_POOL)]
        locations.append(base_city)
    df["location"] = locations
    return df


def _infer_fraud_label(df):
    df = df.copy()
    if "amount" not in df.columns:
        df["fraud"] = 0
        return df
    q3 = df["amount"].quantile(0.75)
    iqr = q3 - df["amount"].quantile(0.25)
    upper = q3 + 3 * iqr
    z = np.abs((df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-9))
    df["fraud"] = ((df["amount"] > upper) | (z > 3)).astype(int)
    return df


def _txn_count_last_24h_vectorized(group: pd.DataFrame) -> pd.Series:
    times = group["timestamp"].values.astype('datetime64[ns]')
    txn_counts = np.zeros(len(group), dtype=int)
    for i in range(len(group)):
        mask = np.abs(times - times[i]) <= np.timedelta64(24, 'h')
        txn_counts[i] = np.sum(mask)
    return pd.Series(txn_counts, index=group.index)


def _engineer_features(df):
    df = df.copy().sort_values(["sender", "timestamp", "receiver"]).reset_index(drop=True)
    global_mean = float(df["amount"].mean()) if len(df) else 0.0

    df["prev_txn_count"] = df.groupby("sender").cumcount()
    sender_cumsum = df.groupby("sender")["amount"].cumsum() - df["amount"]
    df["avg_amount_user"] = sender_cumsum / df["prev_txn_count"].replace(0, np.nan)
    df["avg_amount_user"] = df["avg_amount_user"].fillna(global_mean).round(2)

    sender_cummax = df.groupby("sender")["amount"].cummax()
    df["historical_max_amount"] = sender_cummax.groupby(df["sender"]).shift(1).fillna(df["avg_amount_user"])
    df["historical_max_flag"] = (df["amount"] > df["historical_max_amount"]).astype(int)
    df["amount_deviation"] = ((df["amount"] - df["avg_amount_user"]).abs() / (df["avg_amount_user"].abs() + 1.0)).round(4)

    receiver_counts = df.groupby(["sender", "receiver"]).cumcount()
    df["is_new_receiver"] = (receiver_counts == 0).astype(int)

    df["time_gap"] = (
        df.groupby("sender")["timestamp"].diff().dt.total_seconds().div(3600).fillna(24.0).clip(lower=0.05)
    ).round(3)
    df["txn_count_24h"] = df.groupby("sender").apply(_txn_count_last_24h_vectorized).reset_index(level=0, drop=True).astype(int)
    df["transaction_velocity"] = (df["amount"] / df["time_gap"].replace(0, 0.05)).round(2)
    df["hour_of_day"] = df["timestamp"].dt.hour.astype(int)
    df["unusual_time_flag"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5, 23]).astype(int)

    previous_location = df.groupby("sender")["location"].shift(1).fillna(df["location"])
    df["location_change_flag"] = (df["location"] != previous_location).astype(int)

    df["sender_tx_count"] = df.groupby("sender")["sender"].transform("count")
    df["receiver_tx_count"] = df.groupby("receiver")["receiver"].transform("count")
    df["sender_avg_amount"] = df.groupby("sender")["amount"].transform("mean").round(2)
    df["sender_max_amount"] = df.groupby("sender")["amount"].transform("max").round(2)
    df["amount_log"] = np.log1p(df["amount"]).round(5)
    df["amount_zscore"] = ((df["amount"] - global_mean) / (df["amount"].std() + 1e-9)).round(4)
    df["is_large_amount"] = (df["amount"] > global_mean + 2 * df["amount"].std()).astype(int)
    df["amount_round"] = (df["amount"] % 1 == 0).astype(int)
    df["amount_bin"] = pd.qcut(df["amount"], q=min(10, len(df)), labels=False, duplicates="drop")
    df["amount_bin"] = df["amount_bin"].fillna(0).astype(int)

    if "balance_before" in df.columns and "balance_after" in df.columns:
        df["balance_diff"] = (df["balance_before"] - df["balance_after"]).astype(float)
        df["balance_ratio"] = (df["balance_diff"] / (df["balance_before"] + 1e-9)).round(4)
        df["emptied_account"] = ((df["balance_after"] == 0) & (df["balance_before"] > 0)).astype(int)
        df["balance_mismatch"] = np.abs(df["balance_diff"] - df["amount"]).round(2)
    else:
        df["balance_diff"] = 0.0
        df["balance_ratio"] = 0.0
        df["emptied_account"] = 0
        df["balance_mismatch"] = 0.0

    return df


def load_dataset(filepath: str) -> dict:
    global _df, _raw_columns, _column_map

    raw = pd.read_csv(filepath)
    raw.columns = [c.strip() for c in raw.columns]
    _raw_columns = list(raw.columns)
    raw.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in raw.columns]

    _column_map = _auto_map_columns(raw)
    df = raw.rename(columns=_column_map).dropna(axis=1, how="all")

    if "amount" not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric column found to use as transaction amount.")
        best = max(numeric_cols, key=lambda c: df[c].std())
        df = df.rename(columns={best: "amount"})

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).abs()

    if "fraud" not in df.columns:
        df = _infer_fraud_label(df)
    else:
        df["fraud"] = pd.to_numeric(df["fraud"], errors="coerce").fillna(0).astype(int)
        if df["fraud"].nunique() < 2:
            df = _infer_fraud_label(df)

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "sender" not in df.columns:
        df["sender"] = df[cat_cols[0]].astype(str) if cat_cols else ("ACC_" + df.index.astype(str))
    if "receiver" not in df.columns:
        remaining = [c for c in cat_cols if c != "sender"]
        df["receiver"] = df[remaining[0]].astype(str) if remaining else ("MERCH_" + (df.index % 500).astype(str))
    if "payment_method" not in df.columns:
        remaining = [c for c in cat_cols if c not in ("sender", "receiver")]
        df["payment_method"] = df[remaining[0]].astype(str) if remaining else "TRANSACTION"

    df["sender"] = df["sender"].astype(str).fillna("UNKNOWN")
    df["receiver"] = df["receiver"].astype(str).fillna("UNKNOWN")
    df["payment_method"] = df["payment_method"].astype(str).fillna("UNKNOWN")

    df = _ensure_timestamp(df)
    df = _ensure_location(df)
    df = _engineer_features(df)

    _df = df

    # Reset derived caches; they warm up asynchronously after upload.
    from ..caches import caches
    caches.invalidate()

    return get_dataset_info()


def get_df() -> pd.DataFrame | None:
    return _df


def is_loaded() -> bool:
    return _df is not None


def get_dataset_info() -> dict:
    if _df is None:
        return {"error": "No dataset loaded."}

    fraud_count = int(_df["fraud"].sum())
    total = len(_df)
    amount_stats = {
        "min": round(float(_df["amount"].min()), 2),
        "max": round(float(_df["amount"].max()), 2),
        "mean": round(float(_df["amount"].mean()), 2),
        "median": round(float(_df["amount"].median()), 2),
        "std": round(float(_df["amount"].std()), 2),
    }

    return {
        "loaded_at": datetime.utcnow().isoformat(),
        "total_transactions": total,
        "fraud_transactions": fraud_count,
        "legitimate_transactions": total - fraud_count,
        "fraud_rate": round(fraud_count / total * 100, 2) if total else 0,
        "unique_senders": int(_df["sender"].nunique()),
        "unique_receivers": int(_df["receiver"].nunique()),
        "payment_methods": _df["payment_method"].value_counts().head(10).to_dict(),
        "amount_stats": amount_stats,
        "columns_detected": _column_map,
        "original_columns": _raw_columns,
        "engineered_features": [
            "avg_amount_user",
            "txn_count_24h",
            "amount_deviation",
            "is_new_receiver",
            "time_gap",
            "transaction_velocity",
            "unusual_time_flag",
            "location_change_flag",
        ],
    }


def get_column_map() -> dict:
    return _column_map

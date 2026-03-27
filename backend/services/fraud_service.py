# services/fraud_service.py
# Smart dataset loader — accepts ANY financial fraud CSV.
# Auto-maps columns, engineers features, infers fraud labels if missing.

import pandas as pd
import numpy as np

_df: pd.DataFrame = None
_raw_columns: list = []
_column_map: dict = {}

# ── Column keyword maps ────────────────────────────────────────────────
COLUMN_KEYWORDS = {
    "amount": [
        "amount", "amt", "value", "transaction_amount", "trans_amount",
        "payment_amount", "sum", "price", "total", "money", "cash",
        "transaction_value", "txn_amount"
    ],
    "fraud": [
        "fraud", "is_fraud", "isfraud", "label", "class", "target",
        "fraudulent", "fraud_flag", "fraud_label", "anomaly", "suspicious",
        "isFraud", "is_fraudulent", "fraud_ind", "fraud_indicator"
    ],
    "sender": [
        "sender", "nameorig", "name_orig", "from", "source", "payer",
        "account_from", "acc_from", "customer", "user", "account_id",
        "card_number", "cardholder", "customer_id", "userid",
        "from_account", "originator", "account_number", "acc_no",
        "account_no", "card_no", "v1", "accountnumber"
    ],
    "receiver": [
        "receiver", "namedest", "name_dest", "to", "destination", "payee",
        "account_to", "acc_to", "merchant", "beneficiary", "merchant_id",
        "to_account", "dest", "recipient", "terminal_id", "pos_terminal"
    ],
    "payment_method": [
        "payment_method", "type", "transaction_type", "trans_type",
        "method", "category", "payment_type", "channel", "mode",
        "payment_channel", "transaction_method", "txn_type", "tx_type"
    ],
    "balance_before": [
        "oldbalanceorg", "old_balance", "balance_before", "prev_balance",
        "before_balance", "opening_balance", "balance_orig",
        "oldbalanceorig", "balance_old"
    ],
    "balance_after": [
        "newbalanceorig", "new_balance", "balance_after", "after_balance",
        "closing_balance", "balance_dest", "newbalancedest",
        "newbalanceorg", "balance_new"
    ],
}


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


def _engineer_features(df):
    df = df.copy()

    if "amount" in df.columns:
        mean_amt = df["amount"].mean()
        std_amt  = df["amount"].std() + 1e-9
        df["amount_log"]      = np.log1p(df["amount"])
        df["amount_zscore"]   = (df["amount"] - mean_amt) / std_amt
        df["is_large_amount"] = (df["amount"] > mean_amt + 2 * std_amt).astype(int)
        df["amount_round"]    = (df["amount"] % 1 == 0).astype(int)
        df["amount_bin"]      = pd.qcut(df["amount"], q=10, labels=False, duplicates="drop")

    if "balance_before" in df.columns and "balance_after" in df.columns:
        df["balance_diff"]    = df["balance_before"] - df["balance_after"]
        df["balance_ratio"]   = df["balance_diff"] / (df["balance_before"] + 1e-9)
        df["emptied_account"] = ((df["balance_after"] == 0) & (df["balance_before"] > 0)).astype(int)
        df["balance_mismatch"] = np.abs(df["balance_diff"] - df.get("amount", 0)).astype(float)

    if "sender" in df.columns:
        sc = df["sender"].value_counts()
        df["sender_tx_count"] = df["sender"].map(sc)
        if "amount" in df.columns:
            df["sender_avg_amount"] = df.groupby("sender")["amount"].transform("mean")
            df["sender_max_amount"] = df.groupby("sender")["amount"].transform("max")

    if "receiver" in df.columns:
        rc = df["receiver"].value_counts()
        df["receiver_tx_count"] = df["receiver"].map(rc)

    return df


def _infer_fraud_label(df):
    df = df.copy()
    if "amount" not in df.columns:
        df["fraud"] = 0
        return df
    Q3  = df["amount"].quantile(0.75)
    IQR = Q3 - df["amount"].quantile(0.25)
    upper = Q3 + 3 * IQR
    z = np.abs((df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-9))
    df["fraud"] = ((df["amount"] > upper) | (z > 3)).astype(int)
    return df


def load_dataset(filepath: str) -> dict:
    global _df, _raw_columns, _column_map

    raw = pd.read_csv(filepath)
    raw.columns = [c.strip() for c in raw.columns]
    _raw_columns = list(raw.columns)
    raw.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in raw.columns]

    _column_map = _auto_map_columns(raw)
    df = raw.rename(columns=_column_map)
    df = df.dropna(axis=1, how="all")

    # ── Amount ─────────────────────────────────────────────────────────
    if "amount" not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            # pick column with highest variance as likely amount
            best = max(num_cols, key=lambda c: df[c].std())
            df = df.rename(columns={best: "amount"})
        else:
            raise ValueError("No numeric column found to use as transaction amount.")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).abs()

    # ── Fraud label ────────────────────────────────────────────────────
    if "fraud" not in df.columns:
        df = _infer_fraud_label(df)
    else:
        df["fraud"] = pd.to_numeric(df["fraud"], errors="coerce").fillna(0).astype(int)
        if df["fraud"].sum() == 0 or df["fraud"].nunique() < 2:
            df = _infer_fraud_label(df)

    # ── Sender ─────────────────────────────────────────────────────────
    if "sender" not in df.columns:
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            df["sender"] = df[cat_cols[0]].astype(str)
        else:
            df["sender"] = "ACCT_" + df.index.astype(str)

    # ── Receiver ───────────────────────────────────────────────────────
    if "receiver" not in df.columns:
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        remaining = [c for c in cat_cols if c not in ("sender",)]
        if remaining:
            df["receiver"] = df[remaining[0]].astype(str)
        else:
            df["receiver"] = "MERCH_" + (df.index % 500).astype(str)

    # ── Payment method ─────────────────────────────────────────────────
    if "payment_method" not in df.columns:
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        remaining = [c for c in cat_cols if c not in ("sender", "receiver")]
        if remaining:
            df["payment_method"] = df[remaining[0]].astype(str)
        else:
            df["payment_method"] = "TRANSACTION"

    df["sender"]         = df["sender"].astype(str).fillna("UNKNOWN")
    df["receiver"]       = df["receiver"].astype(str).fillna("UNKNOWN")
    df["payment_method"] = df["payment_method"].astype(str).fillna("UNKNOWN")

    # Feature engineering
    df = _engineer_features(df)

    _df = df
    return get_dataset_info()


def get_df() -> pd.DataFrame:
    return _df


def is_loaded() -> bool:
    return _df is not None


def get_dataset_info() -> dict:
    if _df is None:
        return {"error": "No dataset loaded."}

    fraud_count = int(_df["fraud"].sum())
    total = len(_df)

    return {
        "total_transactions": total,
        "fraud_transactions": fraud_count,
        "legitimate_transactions": total - fraud_count,
        "fraud_rate": round(fraud_count / total * 100, 2) if total else 0,
        "unique_senders":   int(_df["sender"].nunique()),
        "unique_receivers": int(_df["receiver"].nunique()),
        "payment_methods": _df["payment_method"].value_counts().head(10).to_dict(),
        "amount_stats": {
            "min":    round(float(_df["amount"].min()),    2),
            "max":    round(float(_df["amount"].max()),    2),
            "mean":   round(float(_df["amount"].mean()),   2),
            "median": round(float(_df["amount"].median()), 2),
            "std":    round(float(_df["amount"].std()),    2),
        },
        "columns_detected": _column_map,
        "original_columns": _raw_columns,
    }


def get_column_map() -> dict:
    return _column_map

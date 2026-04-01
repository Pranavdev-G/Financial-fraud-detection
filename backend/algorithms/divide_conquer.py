import pandas as pd


def merge_sort(arr):
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    return _merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))


def _merge(left, right):
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def iterative_max(values):
    return max(values) if values else 0.0


def iterative_extreme_deviation(values, baseline):
    return max((abs(float(v) - baseline) for v in values), default=0.0)


def compute_outlier_scores(df: pd.DataFrame) -> pd.Series:
    baseline = pd.to_numeric(df.get("avg_amount_user", 0.0), errors="coerce").fillna(0.0)
    amount = pd.to_numeric(df.get("amount", 0.0), errors="coerce").fillna(0.0)
    hist_max = pd.to_numeric(df.get("historical_max_amount", baseline), errors="coerce").fillna(baseline)

    deviation_ratio = (amount - baseline).abs() / (baseline.abs() + 1.0)
    scores = (deviation_ratio * 22).clip(upper=100)
    scores = (scores + (amount > hist_max).astype(float) * 25).clip(upper=100)
    return scores.round(2)


def run_divide_conquer(df, sample=300):
    df_s = df.iloc[:sample].copy().reset_index(drop=True)
    amounts = [round(float(value), 2) for value in df_s["amount"].tolist()]
    sorted_amounts = merge_sort(amounts)
    global_avg = float(df_s["amount"].mean()) if len(df_s) else 0.0
    max_amount = iterative_max(amounts)
    max_deviation = iterative_extreme_deviation(amounts, global_avg)

    if "outlier_score" not in df_s.columns:
        df_s["outlier_score"] = compute_outlier_scores(df_s)
    flagged = df_s.sort_values("outlier_score", ascending=False).head(12)
    outliers = [
        {
            "sender": str(row.sender),
            "receiver": str(row.receiver),
            "amount": round(float(row.amount), 2),
            "payment_method": str(row.payment_method),
            "outlier_score": round(float(row.outlier_score), 2),
            "historical_max_amount": round(float(row.historical_max_amount), 2),
            "fraud_flag": int(row.fraud),
        }
        for row in flagged.itertuples(index=False)
    ]

    return {
        "total_processed": len(df_s),
        "global_average": round(global_avg, 2),
        "max_transaction": round(max_amount, 2),
        "max_deviation": round(max_deviation, 2),
        "merge_sorted_sample": sorted_amounts[:25],
        "outlier_transactions": outliers,
    }

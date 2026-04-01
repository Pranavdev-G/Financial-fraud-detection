import numpy as np
import pandas as pd


def compute_pattern_scores(df: pd.DataFrame) -> pd.Series:
    sorted_df = df.sort_values(["sender", "timestamp"]).reset_index()
    scored_values = np.zeros(len(sorted_df), dtype=float)

    for _, group in sorted_df.groupby("sender", sort=False):
        ratios = group["amount_deviation"].to_numpy(dtype=float, copy=False) + 1.0
        historical_flags = group["historical_max_flag"].to_numpy(dtype=int, copy=False)
        positions = group.index.to_numpy(dtype=int, copy=False)
        quiet_streak = 0
        spike_streak = 0
        best_score = 0.0
        for pos, ratio, historical_flag in zip(positions, ratios, historical_flags):
            is_quiet = ratio <= 1.1
            is_spike = ratio >= 1.8 or historical_flag == 1

            if is_quiet:
                quiet_streak = min(quiet_streak + 1, 4)
                spike_streak = max(spike_streak - 1, 0)
                current_score = max(5.0, best_score * 0.35)
            elif is_spike:
                spike_streak += 1
                current_score = min(100.0, 18 + quiet_streak * 12 + spike_streak * 18 + ratio * 8)
                best_score = max(best_score, current_score)
            else:
                quiet_streak = max(quiet_streak - 1, 0)
                spike_streak = max(spike_streak - 1, 0)
                current_score = max(best_score * 0.45, 10.0 + ratio * 4)
                best_score = max(best_score * 0.8, current_score)

            scored_values[pos] = round(float(current_score), 2)

    scored = pd.Series(scored_values, index=sorted_df["index"])
    return scored.reindex(df.index).fillna(0.0)


def run_dynamic_programming(df, sample=500):
    df_s = df.iloc[:sample].copy().reset_index(drop=True)
    if "pattern_score" not in df_s.columns:
        df_s["pattern_score"] = compute_pattern_scores(df_s)
    hot = df_s.sort_values("pattern_score", ascending=False).head(12)
    suspicious_chain = [
        {
            "sender": str(row.sender),
            "receiver": str(row.receiver),
            "amount": round(float(row.amount), 2),
            "payment_method": str(row.payment_method),
            "location": str(row.location),
            "timestamp": str(row.timestamp),
            "txn_count_24h": int(row.txn_count_24h),
            "pattern_score": round(float(row.pattern_score), 2),
            "fraud_flag": int(row.fraud),
        }
        for row in hot.itertuples(index=False)
    ]

    sender_patterns = (
        df_s.groupby("sender")["pattern_score"]
        .max()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "sample_size": len(df_s),
        "max_pattern_score": round(float(df_s["pattern_score"].max()), 2) if len(df_s) else 0,
        "avg_pattern_score": round(float(df_s["pattern_score"].mean()), 2) if len(df_s) else 0,
        "high_pattern_transactions": suspicious_chain,
        "pattern_by_sender": [
            {"sender": row["sender"], "pattern_score": round(float(row["pattern_score"]), 2)}
            for row in sender_patterns
        ],
    }

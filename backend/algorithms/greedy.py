# algorithms/greedy.py
# Greedy suspicious activity detection.
# Applies multiple fraud rules greedily, ranked by risk score.

import numpy as np
import pandas as pd


def detect_suspicious_greedy(df, top_n=50):
    df = df.copy()
    amounts  = df["amount"].astype(float)
    mean_val = amounts.mean()
    std_val  = amounts.std()
    threshold = mean_val + 2 * std_val

    parts = []

    # Rule 1: High-value anomaly (amount > mean + 2σ)
    r1 = df[df["amount"] > threshold].copy()
    r1["risk_reason"] = "High-value anomaly (> mean + 2σ)"
    r1["risk_score"]  = ((r1["amount"] - mean_val) / (std_val + 1e-9)).round(2)
    parts.append(r1)

    # Rule 2: Account emptied (balance → 0)
    if "balance_before" in df.columns and "balance_after" in df.columns:
        r2 = df[(df["balance_after"] == 0) & (df["balance_before"] > 0)].copy()
        r2["risk_reason"] = "Account emptied (balance → 0)"
        r2["risk_score"]  = 9.5
        if len(r2): parts.append(r2)

    # Rule 3: Balance mismatch — only flag if mismatch > 80% AND large amount
    if "balance_before" in df.columns and "balance_after" in df.columns:
        bal_diff = (df["balance_before"] - df["balance_after"]).abs()
        mismatch = (bal_diff - df["amount"]).abs()
        # Strict: mismatch > 80% of amount AND amount > threshold
        mask = (mismatch > df["amount"] * 0.8) & (df["amount"] > mean_val + std_val)
        r3 = df[mask].copy()
        r3["risk_reason"] = "Balance mismatch + high amount"
        r3["risk_score"]  = 8.0
        if len(r3): parts.append(r3)

    # Rule 4: High-frequency sender (top 2% — stricter)
    sender_counts = df["sender"].value_counts()
    q98 = sender_counts.quantile(0.98)
    if q98 > 1:
        high_freq = sender_counts[sender_counts > q98].index
        r4 = df[df["sender"].isin(high_freq)].copy()
        r4["risk_reason"] = "High-frequency sender (top 2%)"
        r4["risk_score"]  = 7.0
        if len(r4): parts.append(r4)

    # Rule 5: Large round-number amounts (multiples of 1000 above mean)
    r5 = df[(df["amount"] % 1000 == 0) & (df["amount"] > threshold)].copy()
    r5["risk_reason"] = "Suspicious round-number large amount"
    r5["risk_score"]  = 6.5
    if len(r5): parts.append(r5)

    if not parts:
        return {"threshold": round(threshold,2), "mean": round(mean_val,2),
                "std_dev": round(std_val,2), "suspicious_count": 0,
                "rules_applied": 5, "top_suspicious": []}

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["sender","receiver","amount"], keep="first")
    combined = combined.sort_values(["risk_score","amount"], ascending=[False,False])

    results = []
    running_total = 0.0
    for i, (_, row) in enumerate(combined.head(top_n).iterrows()):
        running_total += float(row["amount"])
        results.append({
            "_i":            i + 1,
            "sender":        str(row.get("sender","N/A")),
            "receiver":      str(row.get("receiver","N/A")),
            "amount":        round(float(row["amount"]),2),
            "payment_method":str(row.get("payment_method","N/A")),
            "fraud_flag":    int(row.get("fraud",0)),
            "risk_reason":   str(row.get("risk_reason","")),
            "risk_score":    round(float(row.get("risk_score",0)),2),
            "running_total": round(running_total,2),
        })

    return {
        "threshold":        round(threshold,2),
        "mean":             round(mean_val,2),
        "std_dev":          round(std_val,2),
        "suspicious_count": len(combined),
        "rules_applied":    5,
        "top_suspicious":   results,
    }

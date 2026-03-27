# algorithms/dynamic_programming.py
# DP-based suspicious activity detection.
# Kadane's algorithm: finds peak burst fraud window (max-sum subarray).
# LIS: finds longest escalating transaction chain — gradual fraud escalation.
# Both return actual transaction details, not just amounts.

import bisect
import numpy as np


def max_subarray_sum(amounts):
    """Kadane's algorithm. Returns (max_sum, start_idx, end_idx)."""
    if not amounts:
        return 0, 0, 0
    best_sum = current_sum = amounts[0]
    best_start = start = 0
    best_end = 0
    for i in range(1, len(amounts)):
        if current_sum + amounts[i] < amounts[i]:
            current_sum = amounts[i]
            start = i
        else:
            current_sum += amounts[i]
        if current_sum > best_sum:
            best_sum    = current_sum
            best_start  = start
            best_end    = i
    return round(best_sum, 2), best_start, best_end


def longest_increasing_subsequence_indices(amounts):
    """
    O(n log n) LIS using patience sorting.
    Returns list of indices (into amounts) that form the LIS.
    """
    if not amounts:
        return []
    n     = len(amounts)
    tails = []        # tails[i] = index of smallest tail for IS of length i+1
    preds = [-1] * n  # predecessor for path reconstruction
    pos   = []        # actual index stored at each tails position

    for i, a in enumerate(amounts):
        idx = bisect.bisect_left([amounts[t] for t in tails], a)
        if idx == len(tails):
            tails.append(i)
            pos.append(i)
        else:
            tails[idx] = i
            pos[idx]   = i
        preds[i] = tails[idx - 1] if idx > 0 else -1

    # Reconstruct path
    path = []
    k = tails[-1]
    while k != -1:
        path.append(k)
        k = preds[k]
    path.reverse()
    return path


def run_dynamic_programming(df, sample=500):
    amounts = [round(float(x), 2) for x in df["amount"].tolist()[:sample]]
    df_s    = df.iloc[:sample].reset_index(drop=True)

    # ── Kadane's: burst window ─────────────────────────────────────────
    max_sum, s_idx, e_idx = max_subarray_sum(amounts)
    window_df = df_s.iloc[s_idx: e_idx + 1]

    window_transactions = []
    for _, row in window_df.iterrows():
        window_transactions.append({
            "sender":         str(row.get("sender", "N/A")),
            "receiver":       str(row.get("receiver", "N/A")),
            "amount":         round(float(row["amount"]), 2),
            "payment_method": str(row.get("payment_method", "N/A")),
            "fraud_flag":     int(row.get("fraud", 0)),
        })

    fraud_in_window = sum(1 for t in window_transactions if t["fraud_flag"] == 1)

    # ── LIS: escalating chain ──────────────────────────────────────────
    lis_indices = longest_increasing_subsequence_indices(amounts)

    lis_transactions = []
    for idx in lis_indices:
        row = df_s.iloc[idx]
        lis_transactions.append({
            "index":          int(idx),
            "sender":         str(row.get("sender", "N/A")),
            "receiver":       str(row.get("receiver", "N/A")),
            "amount":         round(float(row["amount"]), 2),
            "payment_method": str(row.get("payment_method", "N/A")),
            "fraud_flag":     int(row.get("fraud", 0)),
        })

    fraud_in_lis = sum(1 for t in lis_transactions if t["fraud_flag"] == 1)

    return {
        "sample_size": len(amounts),
        "max_subarray": {
            "sum":              max_sum,
            "start_index":      s_idx,
            "end_index":        e_idx,
            "window_length":    e_idx - s_idx + 1,
            "fraud_in_window":  fraud_in_window,
            "transactions":     window_transactions,
        },
        "longest_increasing_subsequence": {
            "length":          len(lis_indices),
            "fraud_in_lis":    fraud_in_lis,
            "transactions":    lis_transactions,
        },
    }

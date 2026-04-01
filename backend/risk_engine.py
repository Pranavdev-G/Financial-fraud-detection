from typing import Any

import numpy as np
import pandas as pd

from .ai_model import get_model
from .algorithms.divide_conquer import compute_outlier_scores, run_divide_conquer
from .algorithms.dynamic_programming import compute_pattern_scores, run_dynamic_programming
from .algorithms.hashing_implementation import build_user_profile_hash, run_hashing_analysis
from .caches import caches


def _risk_level(score: float) -> str:
    if score > 75:
        return "HIGH"
    if score > 45:
        return "MEDIUM"
    return "LOW"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _rule_score(row: pd.Series) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    deviation = _safe_float(row.get("amount_deviation"))
    if deviation >= 2.5:
        score += 35
        reasons.append("High deviation from user average")
    elif deviation >= 1.5:
        score += 20
        reasons.append("Amount is elevated versus sender history")

    if int(row.get("is_new_receiver", 0)) == 1:
        score += 12
        reasons.append("New receiver for this sender")

    if int(row.get("txn_count_24h", 0)) >= 3:
        score += 10
        reasons.append("High transaction count in the last 24 hours")

    if int(row.get("unusual_time_flag", 0)) == 1:
        score += 8
        reasons.append("Transaction happened at an unusual hour")

    if int(row.get("location_change_flag", 0)) == 1:
        score += 8
        reasons.append("Location changed since the previous transaction")

    if int(row.get("historical_max_flag", 0)) == 1:
        score += 18
        reasons.append("Current amount exceeds historical maximum")

    if _safe_float(row.get("transaction_velocity")) > _safe_float(row.get("avg_amount_user")) * 2:
        score += 9
        reasons.append("Transaction velocity is unusually high")

    return min(score, 100.0), reasons


def _build_reasons(row: pd.Series, ml_probability: float) -> list[str]:
    _, reasons = _rule_score(row)
    pattern_score = _safe_float(row.get("pattern_score"))
    outlier_score = _safe_float(row.get("outlier_score"))

    if pattern_score >= 55:
        reasons.append("Abnormal sequential pattern detected")
    if outlier_score >= 50:
        reasons.append("Extreme deviation detected by recursive outlier analysis")
    if ml_probability >= 0.8:
        reasons.append(f"ML confidence high: {ml_probability:.2f}")
    elif ml_probability >= 0.55:
        reasons.append(f"ML confidence moderate: {ml_probability:.2f}")

    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped[:6]


def _finalize_response(row: pd.Series, ml_probability: float) -> dict[str, Any]:
    rule_score, _ = _rule_score(row)
    pattern_score = _safe_float(row.get("pattern_score"))
    outlier_score = _safe_float(row.get("outlier_score"))
    ml_score = round(ml_probability * 100, 2)
    final_score = (
        rule_score * 0.4 +
        pattern_score * 0.2 +
        outlier_score * 0.1 +
        ml_score * 0.3
    )
    final_score = round(min(final_score, 100.0), 2)

    return {
        "risk_score": final_score,
        "risk_level": _risk_level(final_score),
        "confidence": round(ml_probability, 4),
        "reasons": _build_reasons(row, ml_probability),
        "components": {
            "rule_score": round(rule_score, 2),
            "pattern_score": round(pattern_score, 2),
            "outlier_score": round(outlier_score, 2),
            "ml_score": ml_score,
        },
    }


def evaluate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if caches.is_fresh() and caches.scored_df is not None and len(caches.scored_df) == len(df):
        return caches.scored_df.copy()

    model = get_model()
    enriched = df.copy().reset_index(drop=True)
    enriched["pattern_score"] = compute_pattern_scores(enriched)
    enriched["outlier_score"] = compute_outlier_scores(enriched)

    if caches.ml_preds is not None and len(caches.ml_preds) == len(enriched):
        ml_probabilities = np.asarray(caches.ml_preds)
    else:
        ml_probabilities = np.asarray(model.score_dataframe(enriched))
        caches.ml_preds = ml_probabilities

    responses = [_finalize_response(enriched.iloc[idx], float(ml_probabilities[idx])) for idx in range(len(enriched))]
    enriched["ml_probability"] = np.round(ml_probabilities, 4)
    enriched["ml_score"] = [item["components"]["ml_score"] for item in responses]
    enriched["rule_score"] = [item["components"]["rule_score"] for item in responses]
    enriched["risk_score"] = [item["risk_score"] for item in responses]
    enriched["risk_level"] = [item["risk_level"] for item in responses]
    enriched["risk_reasons"] = ["; ".join(item["reasons"]) for item in responses]

    enriched = enriched.sort_values("risk_score", ascending=False).reset_index(drop=True)
    caches.scored_df = enriched.copy()
    return enriched


def simulate_transaction(
    df: pd.DataFrame,
    sender: str,
    receiver: str,
    payment_method: str,
    amount: float,
    location: str | None = None,
    hour: int | None = None,
) -> dict[str, Any]:
    profiles = caches.user_hash or build_user_profile_hash(df, sample=len(df))
    profile = profiles.search(sender) if profiles else None
    profile_data = profile[0] if isinstance(profile, list) and profile else {}

    history = df[df["sender"].astype(str) == str(sender)].sort_values("timestamp")
    avg_amount = _safe_float(profile_data.get("avg_amount"), _safe_float(df["amount"].mean()))
    last_location = profile_data.get(
        "last_location",
        df["location"].mode().iloc[0] if "location" in df.columns and not df["location"].mode().empty else "UNKNOWN",
    )
    last_time = history["timestamp"].iloc[-1] if len(history) else df["timestamp"].max()
    receiver_known = receiver in profile_data.get("known_receivers", [])
    txn_count_24h = int((history["timestamp"] >= (last_time - pd.Timedelta(hours=24))).sum()) if len(history) else 0

    if hour is None:
        hour = int(last_time.hour) if pd.notna(last_time) else 12
    current_location = location or last_location
    amount_value = _safe_float(amount)
    simulated_time = pd.Timestamp(last_time.date()) + pd.Timedelta(hours=int(hour)) if pd.notna(last_time) else pd.Timestamp.now()
    time_gap = max((simulated_time - last_time).total_seconds() / 3600, 0.05) if pd.notna(last_time) else 24.0
    historical_max = _safe_float(profile_data.get("max_amount"), avg_amount)

    row = pd.DataFrame(
        [
            {
                "sender": sender,
                "receiver": receiver,
                "payment_method": payment_method,
                "amount": amount_value,
                "fraud": 0,
                "timestamp": simulated_time,
                "location": current_location,
                "avg_amount_user": round(avg_amount, 2),
                "txn_count_24h": txn_count_24h,
                "amount_deviation": round(abs(amount_value - avg_amount) / (abs(avg_amount) + 1.0), 4),
                "is_new_receiver": int(not receiver_known),
                "time_gap": round(time_gap, 3),
                "transaction_velocity": round(amount_value / max(time_gap, 0.05), 2),
                "unusual_time_flag": int(int(hour) <= 5 or int(hour) >= 23),
                "location_change_flag": int(current_location != last_location),
                "hour_of_day": int(hour),
                "historical_max_amount": round(historical_max, 2),
                "historical_max_flag": int(amount_value > historical_max),
                "prev_txn_count": int(profile_data.get("transaction_count", 0)),
                "sender_tx_count": int(profile_data.get("transaction_count", 0)) + 1,
                "receiver_tx_count": int((df["receiver"].astype(str) == str(receiver)).sum()),
                "sender_avg_amount": round(avg_amount, 2),
                "sender_max_amount": round(historical_max, 2),
                "amount_log": round(np.log1p(max(amount_value, 0.0)), 5),
                "amount_zscore": round((amount_value - _safe_float(df["amount"].mean())) / (_safe_float(df["amount"].std(), 1.0) + 1e-9), 4),
                "is_large_amount": int(amount_value > _safe_float(df["amount"].mean()) + 2 * _safe_float(df["amount"].std())),
                "amount_round": int(amount_value % 1 == 0),
                "amount_bin": int(min(9, max(0, round(amount_value / max(_safe_float(df["amount"].quantile(0.9)), 1.0))))),
                "balance_diff": 0.0,
                "balance_ratio": 0.0,
                "emptied_account": 0,
                "balance_mismatch": 0.0,
            }
        ]
    )

    temp = pd.concat([history.tail(6), row], ignore_index=True, sort=False)
    row["pattern_score"] = float(compute_pattern_scores(temp).iloc[-1])
    row["outlier_score"] = float(compute_outlier_scores(row).iloc[0])

    model = get_model()
    prediction = model.predict_from_frame(row)
    response = _finalize_response(row.iloc[0], prediction["fraud_probability"])

    response["sender"] = sender
    response["receiver"] = receiver
    response["payment_method"] = payment_method
    response["amount"] = round(amount_value, 2)
    response["location"] = current_location
    response["timestamp"] = str(simulated_time)
    response["txn_count_24h"] = txn_count_24h
    response["features"] = {
        "avg_amount_user": round(avg_amount, 2),
        "txn_count_24h": txn_count_24h,
        "amount_deviation": round(float(row["amount_deviation"].iloc[0]), 4),
        "is_new_receiver": int(row["is_new_receiver"].iloc[0]),
        "time_gap": round(float(row["time_gap"].iloc[0]), 3),
        "transaction_velocity": round(float(row["transaction_velocity"].iloc[0]), 2),
        "unusual_time_flag": int(row["unusual_time_flag"].iloc[0]),
        "location_change_flag": int(row["location_change_flag"].iloc[0]),
    }
    return response


def build_dashboard_payload(df: pd.DataFrame) -> dict[str, Any]:
    if caches.is_fresh() and caches.dashboard_payload is not None:
        return caches.dashboard_payload.copy()

    scored = evaluate_dataset(df)
    model = get_model()
    top_tx = scored.iloc[0].to_dict() if len(scored) else {}

    user_risk = (
        scored.groupby("sender")["risk_score"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
        .reset_index()
        .rename(columns={"risk_score": "avg_risk_score"})
        .to_dict(orient="records")
    )

    pattern_sender = top_tx.get("sender")
    sender_series = (
        scored[scored["sender"] == pattern_sender].sort_values("timestamp").tail(10)
        if pattern_sender
        else pd.DataFrame()
    )
    pattern_series = [
        {
            "step": index + 1,
            "amount": round(float(row["amount"]), 2),
            "risk_score": round(float(row["risk_score"]), 2),
            "is_anomaly": int(float(row["risk_score"]) > 75),
        }
        for index, (_, row) in enumerate(sender_series.iterrows())
    ]

    suspicious_transactions = []
    for _, row in scored.head(20).iterrows():
        suspicious_transactions.append(
            {
                "sender": str(row.get("sender", "N/A")),
                "receiver": str(row.get("receiver", "N/A")),
                "amount": round(float(row.get("amount", 0)), 2),
                "deviation": round(float(row.get("amount_deviation", 0)), 2),
                "risk_score": round(float(row.get("risk_score", 0)), 2),
                "risk_level": str(row.get("risk_level", "LOW")),
                "reasons": str(row.get("risk_reasons", "")),
                "payment_method": str(row.get("payment_method", "N/A")),
            }
        )

    level_counts = {key: int(value) for key, value in scored["risk_level"].value_counts().to_dict().items()}
    algorithm_roles = {
        "hashing": run_hashing_analysis(df),
        "divide_conquer": run_divide_conquer(df),
        "dynamic_programming": run_dynamic_programming(df),
    }

    result_box = {
        "risk_score": round(float(top_tx.get("risk_score", 0)), 2),
        "risk_level": str(top_tx.get("risk_level", "LOW")),
        "confidence": round(float(top_tx.get("ml_probability", 0)), 4),
        "reasons": str(top_tx.get("risk_reasons", "")).split("; ") if top_tx else [],
        "sender": top_tx.get("sender", "--"),
        "receiver": top_tx.get("receiver", "--"),
        "amount": round(float(top_tx.get("amount", 0)), 2) if top_tx else 0.0,
        "location": top_tx.get("location", "--"),
        "timestamp": str(top_tx.get("timestamp", "--")),
        "txn_count_24h": int(top_tx.get("txn_count_24h", 0)) if top_tx else 0,
    }

    payload = {
        "result_box": result_box,
        "risk_distribution": level_counts,
        "user_risk_bars": user_risk,
        "pattern_series": pattern_series,
        "suspicious_transactions": suspicious_transactions,
        "top_fraud_list": suspicious_transactions[:5],
        "algorithm_roles": algorithm_roles,
        "model_insights": model.get_model_insights(),
    }

    caches.dashboard_payload = payload.copy()
    return payload

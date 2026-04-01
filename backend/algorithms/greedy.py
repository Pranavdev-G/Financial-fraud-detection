def detect_suspicious_greedy(df, top_n=50):
    try:
        from ..risk_engine import evaluate_dataset
    except ImportError:
        from risk_engine import evaluate_dataset

    scored = evaluate_dataset(df).head(top_n).copy()
    top_suspicious = []
    running_total = 0.0

    for i, (_, row) in enumerate(scored.iterrows()):
        running_total += float(row["amount"])
        top_suspicious.append({
            "_i": i + 1,
            "sender": str(row.get("sender", "N/A")),
            "receiver": str(row.get("receiver", "N/A")),
            "amount": round(float(row.get("amount", 0)), 2),
            "payment_method": str(row.get("payment_method", "N/A")),
            "fraud_flag": int(row.get("fraud", 0)),
            "risk_reason": str(row.get("risk_reasons", "")),
            "risk_score": round(float(row.get("risk_score", 0)), 2),
            "running_total": round(running_total, 2),
            "risk_level": str(row.get("risk_level", "LOW")),
        })

    threshold = round(float(scored["risk_score"].quantile(0.75)), 2) if len(scored) else 0.0
    return {
        "threshold": threshold,
        "mean": round(float(scored["risk_score"].mean()), 2) if len(scored) else 0.0,
        "std_dev": round(float(scored["risk_score"].std()), 2) if len(scored) > 1 else 0.0,
        "suspicious_count": len(scored),
        "rules_applied": 4,
        "top_suspicious": top_suspicious,
    }

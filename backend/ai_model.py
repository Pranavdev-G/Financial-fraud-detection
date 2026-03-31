import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

CAT_COLS = ["sender", "receiver", "payment_method"]
EXCLUDE_COLS = {"fraud", "sender", "receiver", "payment_method"}

class FraudDetectionModel:
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.trained = False
        # Fast: 50 trees only
        self.model = RandomForestClassifier(
            n_estimators=50, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1)

    def _encode_categoricals(self, df, fit=False):
        df = df.copy()
        for col in CAT_COLS:
            if col not in df.columns: continue
            if fit:
                le = LabelEncoder()
                df[col+"_enc"] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders.get(col)
                if le:
                    df[col+"_enc"] = df[col].astype(str).map(
                        lambda x, _le=le: int(_le.transform([x])[0]) if x in _le.classes_ else -1)
                else:
                    df[col+"_enc"] = 0
        return df

    def train_model(self, df):
        if "fraud" not in df.columns:
            raise ValueError("No fraud column found.")
        # Use max 10000 rows for speed
        sample = df.sample(min(len(df), 10000), random_state=42)
        df_enc = self._encode_categoricals(sample, fit=True)
        numeric = df_enc.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [c for c in numeric if c not in EXCLUDE_COLS and not c.startswith("Unnamed")]
        X = np.nan_to_num(df_enc[self.feature_cols].values.astype(float))
        y = df_enc["fraud"].astype(int).values
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model.fit(X_train, y_train)
        self.trained = True
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:,1]
        report = classification_report(y_test, y_pred, output_dict=True)
        try: auc = round(roc_auc_score(y_test, y_proba), 4)
        except: auc = None
        return {
            "status": "trained",
            "features_used": len(self.feature_cols),
            "train_accuracy": round(report["accuracy"], 4),
            "fraud_f1": round(report.get("1",{}).get("f1-score",0.0), 4),
            "roc_auc": auc,
        }

    def predict_transaction(self, sender, receiver, payment_method, amount):
        if not self.trained:
            return {"error": "Model not trained. Upload dataset first."}
        try:
            from .services.fraud_service import get_df
        except ImportError:
            from services.fraud_service import get_df
        df = get_df()
        mean_amt = df["amount"].mean() if df is not None else float(amount)
        std_amt = (df["amount"].std()+1e-9) if df is not None else 1.0
        amt = float(amount)
        row = pd.DataFrame([{
            "sender": sender, "receiver": receiver,
            "payment_method": payment_method, "amount": amt,
            "amount_log": np.log1p(amt),
            "amount_zscore": (amt-mean_amt)/std_amt,
            "is_large_amount": int(amt > mean_amt+2*std_amt),
            "amount_round": int(amt%1==0),
        }])
        row_enc = self._encode_categoricals(row, fit=False)
        for c in self.feature_cols:
            if c not in row_enc.columns: row_enc[c] = 0
        X = np.nan_to_num(row_enc[self.feature_cols].values.astype(float))
        X = self.scaler.transform(X)
        prob = self.model.predict_proba(X)[0]
        classes = self.model.classes_.tolist()
        fraud_prob = float(prob[classes.index(1)]) if 1 in classes else float(prob[-1])
        return {
            "sender": sender, "receiver": receiver,
            "payment_method": payment_method, "amount": amt,
            "fraud_probability": round(fraud_prob, 4),
            "legit_probability": round(1-fraud_prob, 4),
            "prediction": 1 if fraud_prob >= 0.5 else 0,
        }

_model_instance = FraudDetectionModel()
def get_model(): return _model_instance

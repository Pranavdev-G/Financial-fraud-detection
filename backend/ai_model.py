import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

warnings.filterwarnings("ignore")

CAT_COLS = ["sender", "receiver", "payment_method", "location"]
EXCLUDE_COLS = {"fraud"}


class FraudDetectionModel:
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.feature_importance = {}
        self.training_summary = {"status": "idle"}
        self.trained = False
        self.model_name = "XGBoost" if XGBClassifier is not None else "RandomForest"
        self.model = (
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            )
            if XGBClassifier is not None
            else RandomForestClassifier(
                n_estimators=120,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        )

    def _encode_categoricals(self, df, fit=False):
        encoded = df.copy()
        for col in CAT_COLS:
            if col not in encoded.columns:
                encoded[col] = "UNKNOWN"
            if fit:
                encoder = LabelEncoder()
                encoded[f"{col}_enc"] = encoder.fit_transform(encoded[col].astype(str))
                self.encoders[col] = encoder
            else:
                encoder = self.encoders.get(col)
                if encoder is None:
                    encoded[f"{col}_enc"] = 0
                else:
                    encoded[f"{col}_enc"] = encoded[col].astype(str).map(
                        lambda value, _encoder=encoder: int(_encoder.transform([value])[0]) if value in _encoder.classes_ else -1
                    )
        return encoded

    def _prepare_numeric_matrix(self, df, fit=False):
        encoded = self._encode_categoricals(df, fit=fit)
        numeric_cols = encoded.select_dtypes(include=[np.number]).columns.tolist()
        if fit:
            self.feature_cols = [col for col in numeric_cols if col not in EXCLUDE_COLS and not col.startswith("unnamed")]
        for col in self.feature_cols:
            if col not in encoded.columns:
                encoded[col] = 0
        matrix = np.nan_to_num(encoded[self.feature_cols].values.astype(float))
        return encoded, matrix

    def train_model(self, df):
        if "fraud" not in df.columns:
            raise ValueError("No fraud column found.")

        self.training_summary = {"status": "training", "model_name": self.model_name}
        sample = df.sample(min(len(df), 4000), random_state=42).copy()
        _, matrix = self._prepare_numeric_matrix(sample, fit=True)
        labels = sample["fraud"].astype(int).values

        self.scaler.fit(matrix)
        matrix = self.scaler.transform(matrix)

        stratify = labels if len(np.unique(labels)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            matrix, labels, test_size=0.2, random_state=42, stratify=stratify
        )

        self.model.fit(X_train, y_train)
        self.trained = True

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        try:
            auc = round(roc_auc_score(y_test, y_proba), 4)
        except Exception:
            auc = None

        self.feature_importance = {
            self.feature_cols[i]: round(float(score), 4)
            for i, score in enumerate(self.model.feature_importances_)
        }
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda item: item[1], reverse=True)
        )

        self.training_summary = {
            "status": "trained",
            "model_name": self.model_name,
            "features_used": len(self.feature_cols),
            "train_accuracy": round(report.get("accuracy", 0.0), 4),
            "fraud_precision": round(report.get("1", {}).get("precision", 0.0), 4),
            "fraud_recall": round(report.get("1", {}).get("recall", 0.0), 4),
            "fraud_f1": round(report.get("1", {}).get("f1-score", 0.0), 4),
            "roc_auc": auc,
            "top_features": list(self.feature_importance.items())[:8],
        }
        return self.training_summary

    def score_dataframe(self, df):
        if not self.trained:
            raise ValueError("Model not trained. Upload dataset first.")
        _, matrix = self._prepare_numeric_matrix(df, fit=False)
        matrix = self.scaler.transform(matrix)
        probabilities = self.model.predict_proba(matrix)
        classes = self.model.classes_.tolist()
        fraud_index = classes.index(1) if 1 in classes else -1
        return probabilities[:, fraud_index]

    def predict_from_frame(self, df):
        if not self.trained:
            return {
                "fraud_probability": 0.5,
                "legit_probability": 0.5,
                "prediction": 0,
                "status": "model_warming_up",
            }
        probability = float(self.score_dataframe(df)[0])
        return {
            "fraud_probability": round(probability, 4),
            "legit_probability": round(1 - probability, 4),
            "prediction": int(probability >= 0.5),
        }

    def get_model_insights(self):
        return {
            "trained": self.trained,
            "training_summary": self.training_summary,
            "feature_importance": [
                {"feature": feature, "importance": importance}
                for feature, importance in list(self.feature_importance.items())[:10]
            ],
        }


_model_instance = FraudDetectionModel()


def get_model():
    return _model_instance

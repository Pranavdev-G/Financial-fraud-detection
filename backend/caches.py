import time
from typing import Dict, Any, Optional
from functools import lru_cache

# Global caches for high-performance access
class FraudCaches:
    def __init__(self):
        self.sender_hash = None
        self.user_hash = None  # from hashing_implementation
        self.pre_features_df = None  # engineered features df
        self.ml_preds = None  # vectorized ML predictions (numpy array)
        self.scored_df = None  # full risk-scored df
        self.dashboard_payload = None
        self.last_preload_time = 0
        self.cache_ttl = 300  # 5min TTL

    def is_fresh(self) -> bool:
        return time.time() - self.last_preload_time < self.cache_ttl

    def invalidate(self):
        self.sender_hash = None
        self.user_hash = None
        self.pre_features_df = None
        self.ml_preds = None
        self.scored_df = None
        self.dashboard_payload = None
        self.last_preload_time = 0

    def preload_all(self, df):
        from .algorithms.hashing_implementation import build_sender_hash_table, build_user_profile_hash
        from .ai_model import get_model
        
        # The uploaded dataset is already engineered during load.
        self.pre_features_df = df.copy()
        
        # Pre-build hash (O(N) once)
        self.sender_hash, _ = build_sender_hash_table(df, sample=len(df))
        self.user_hash = build_user_profile_hash(df)
        
        # Pre-compute ML preds (vectorized)
        model = get_model()
        if model.trained:
            self.ml_preds = model.score_dataframe(self.pre_features_df)
        
        self.last_preload_time = time.time()

caches = FraudCaches()


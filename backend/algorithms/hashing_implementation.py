from dataclasses import dataclass
import pandas as pd
import hashlib
from typing import Dict, List, Any

# ============================================================================
# IMPROVEMENT 1: Better Hash Function using Polynomial Rolling Hash
# Location: Lines 1-45 (NEW - replaces basic hash function)
# ============================================================================

def polynomial_hash(key: str, capacity: int, prime: int = 31) -> int:
    """
    IMPROVEMENT: Better hash distribution using polynomial rolling hash
    
    Why: Python's built-in hash() can have collisions. This ensures
    better distribution across buckets.
    
    Parameters:
        key: The sender ID (string)
        capacity: Number of buckets
        prime: Prime number for polynomial calculation
    
    Returns:
        Bucket index (0 to capacity-1)
    """
    hash_val = 0
    mod = 10**9 + 7
    
    for i, char in enumerate(str(key)):
        hash_val = (hash_val * prime + ord(char)) % mod
    
    return hash_val % capacity


# ============================================================================
# IMPROVEMENT 2: Enhanced HashNode with metadata
# Location: Lines 47-65 (MODIFIED)
# ============================================================================

@dataclass
class HashNode:
    """Enhanced hash node with additional tracking"""
    key: str                                    # Sender ID
    value: list                                 # Transactions list
    next: "HashNode | None" = None
    
    # NEW FIELDS for monitoring
    insertion_time: float = 0.0                # Track when inserted
    access_count: int = 0                      # How many times searched
    collision_count: int = 0                   # Collisions in this chain


# ============================================================================
# IMPROVEMENT 3: Enhanced HashTable with statistics and optimizations
# Location: Lines 67-200 (MODIFIED & EXPANDED)
# ============================================================================

class HashTableImproved:
    """
    Enhanced hash table with:
    - Better hash function (polynomial)
    - Incremental resizing
    - Detailed statistics
    - Collision prevention
    """
    
    def __init__(self, capacity: int = 17):
        """Initialize hash table with capacity estimation"""
        self.capacity = max(5, int(capacity))
        self.table = [None] * self.capacity
        self.size = 0
        self.collisions = 0
        self.total_searches = 0                 # NEW: Search tracking
        self.failed_searches = 0                # NEW: Miss tracking
        self.rehash_count = 0                   # NEW: Resize tracking
        
        # IMPROVEMENT: Incremental resize state (Line 102)
        self.is_resizing = False
        self.new_table = None
        self.rehash_index = 0
    
    # IMPROVEMENT: Better hash function (Line 120)
    def hash_function(self, key: str) -> int:
        """Use polynomial hash instead of built-in hash"""
        return polynomial_hash(str(key), self.capacity)
    
    def _chain_length(self, index: int) -> int:
        """Count chain length at index"""
        count = 0
        current = self.table[index]
        while current:
            count += 1
            current = current.next
        return count
    
    def load_factor(self) -> float:
        """Calculate load factor: (items) / (capacity)"""
        return round(self.size / self.capacity, 4) if self.capacity else 0.0
    
    def insert(self, key: str, value: Any) -> int:
        """
        Insert or update key-value pair
        
        IMPROVEMENT: Incremental resizing
        Instead of all-at-once resize, do it gradually
        """
        
        # IMPROVEMENT: Check if currently resizing and do batch (Line 145)
        if self.is_resizing:
            self._do_incremental_resize_batch(5)
        
        # Standard load factor check (Line 149)
        if self.load_factor() >= 0.75:
            self._start_incremental_resize()
        
        index = self.hash_function(key)
        head = self.table[index]
        current = head
        
        # Check if key exists (Line 156)
        while current:
            if current.key == key:
                if isinstance(current.value, list):
                    if isinstance(value, list):
                        current.value.extend(value)
                    else:
                        current.value.append(value)
                else:
                    current.value = value
                current.access_count += 1      # NEW: Track access
                return index
            current = current.next
        
        # IMPROVEMENT: Track collision (Line 167)
        if head is not None:
            self.collisions += 1
            if head.collision_count is None:
                head.collision_count = 0
            head.collision_count += 1
        
        # Insert new node
        new_value = value if isinstance(value, list) else [value]
        new_node = HashNode(str(key), new_value, head)
        self.table[index] = new_node
        self.size += 1
        return index
    
    def search(self, key: str) -> List[Any] | None:
        """
        Search for key in hash table
        O(1) average, O(n) worst case
        """
        self.total_searches += 1               # NEW: Track searches
        
        index = self.hash_function(key)
        current = self.table[index]
        
        while current:
            if current.key == str(key):
                current.access_count += 1      # NEW: Track hits
                return current.value
            current = current.next
        
        self.failed_searches += 1               # NEW: Track misses
        return None
    
    def delete(self, key: str) -> bool:
        """Delete key from hash table"""
        index = self.hash_function(key)
        current = self.table[index]
        prev = None
        
        while current:
            if current.key == str(key):
                if prev:
                    prev.next = current.next
                else:
                    self.table[index] = current.next
                self.size -= 1
                return True
            prev = current
            current = current.next
        return False
    
    def display(self) -> List[Dict]:
        """Display bucket structure for visualization"""
        rows = []
        for i in range(self.capacity):
            current = self.table[i]
            bucket = []
            while current:
                bucket.append({
                    "key": current.key,
                    "items": len(current.value),
                    "access_count": current.access_count
                })
                current = current.next
            rows.append({
                "bucket_index": i,
                "chain_length": len(bucket),
                "entries": bucket
            })
        return rows
    
    def bucket_stats(self) -> Dict:
        """Get detailed bucket statistics"""
        lengths = [self._chain_length(i) for i in range(self.capacity)]
        used = sum(1 for x in lengths if x > 0)
        
        return {
            "used_buckets": used,
            "empty_buckets": self.capacity - used,
            "max_bucket_depth": max(lengths) if lengths else 0,
            "bucket_lengths": lengths,
            "avg_chain_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "collision_density": round(self.collisions / max(1, self.size), 3),
            "search_efficiency": round(
                (self.total_searches - self.failed_searches) / max(1, self.total_searches),
                3
            )
        }
    
    # IMPROVEMENT 3: Incremental Resizing (Lines 245-280) (NEW)
    def _start_incremental_resize(self):
        """Start incremental resizing without blocking"""
        if self.is_resizing:
            return
        
        self.is_resizing = True
        self.new_table = [None] * (self.capacity * 2 + 1)
        self.rehash_index = 0
        self.rehash_count += 1
    
    def _do_incremental_resize_batch(self, batch_size: int = 5):
        """
        Process batch_size buckets in resize operation
        Called frequently to distribute resize cost
        """
        if not self.is_resizing or self.new_table is None:
            return
        
        processed = 0
        while processed < batch_size and self.rehash_index < self.capacity:
            bucket = self.table[self.rehash_index]
            while bucket:
                # Rehash to new table
                new_index = polynomial_hash(bucket.key, len(self.new_table))
                new_head = self.new_table[new_index]
                new_node = HashNode(bucket.key, list(bucket.value), new_head)
                self.new_table[new_index] = new_node
                bucket = bucket.next
            
            self.rehash_index += 1
            processed += 1
        
        # If resizing complete
        if self.rehash_index >= self.capacity:
            self.table = self.new_table
            self.capacity = len(self.table)
            self.new_table = None
            self.is_resizing = False
    
    def _resize_legacy(self):
        """Legacy: All-at-once resize (kept for reference)"""
        old_entries = []
        for bucket in self.table:
            current = bucket
            while current:
                old_entries.append((current.key, list(current.value)))
                current = current.next
        
        self.capacity = self.capacity * 2 + 1
        self.table = [None] * self.capacity
        self.size = 0
        self.collisions = 0
        
        for key, value in old_entries:
            self.insert(key, value)


# ============================================================================
# IMPROVEMENT 4: Enhanced Transaction Payload
# Location: Lines 290-310 (MODIFIED with extra fields)
# ============================================================================

def _transaction_payload(row: Dict, index: int) -> Dict:
    """Build transaction payload with additional features"""
    return {
        "index": int(index),
        "sender": str(row.get("sender", "N/A")),
        "receiver": str(row.get("receiver", "N/A")),
        "amount": round(float(row.get("amount", 0)), 2),
        "payment_method": str(row.get("payment_method", "N/A")),
        "location": str(row.get("location", "N/A")),
        "timestamp": str(row.get("timestamp", "")),
        "txn_count_24h": int(row.get("txn_count_24h", 0)),
        "fraud_flag": int(row.get("fraud", 0)),
        
        # NEW FIELDS for better analysis
        "time_gap": float(row.get("time_gap", 0)),
        "is_new_receiver": int(row.get("is_new_receiver", 0)),
        "unusual_time_flag": int(row.get("unusual_time_flag", 0)),
        "location_change_flag": int(row.get("location_change_flag", 0)),
    }


def build_sender_hash_table(df: pd.DataFrame, sample: int = 500) -> tuple:
    """Build optimized hash table from transactions"""
    df_s = df.iloc[:sample].reset_index(drop=True)
    
    # IMPROVEMENT: Better capacity estimation (Line 329)
    # Use prime number for better distribution
    primes = [17, 37, 67, 127, 257, 509, 1021]
    estimated = len(df_s) // 6
    capacity = next((p for p in primes if p >= estimated), 1021)
    
    table = HashTableImproved(capacity=capacity)
    
    for index, row in enumerate(df_s.itertuples(index=False)):
        sender = str(getattr(row, "sender", "UNKNOWN"))
        table.insert(
            sender,
            _transaction_payload(
                {
                    "index": index,
                    "sender": sender,
                    "receiver": str(getattr(row, "receiver", "N/A")),
                    "amount": round(float(getattr(row, "amount", 0)), 2),
                    "payment_method": str(getattr(row, "payment_method", "N/A")),
                    "location": str(getattr(row, "location", "N/A")),
                    "timestamp": str(getattr(row, "timestamp", "")),
                    "txn_count_24h": int(getattr(row, "txn_count_24h", 0)),
                    "fraud": int(getattr(row, "fraud", 0)),
                    "time_gap": float(getattr(row, "time_gap", 0)),
                    "is_new_receiver": int(getattr(row, "is_new_receiver", 0)),
                    "unusual_time_flag": int(getattr(row, "unusual_time_flag", 0)),
                    "location_change_flag": int(getattr(row, "location_change_flag", 0)),
                },
                index
            )
        )
    
    return table, df_s


def build_user_profile_hash(df: pd.DataFrame, sample: int = 500) -> HashTableImproved:
    """Build user profile hash table with additional features"""
    table, df_s = build_sender_hash_table(df, sample=sample)
    profile_table = HashTableImproved(capacity=max(17, table.capacity))
    
    if len(df_s) == 0:
        return profile_table
    
    sender_frame = df_s.copy()
    sender_frame["sender"] = sender_frame["sender"].astype(str)
    sender_frame["receiver"] = sender_frame["receiver"].astype(str)
    
    grouped = sender_frame.groupby("sender", sort=False)
    summary = grouped.agg(
        avg_amount=("amount", "mean"),
        transaction_count=("amount", "size"),
        last_location=("location", "last"),
        max_amount=("amount", "max"),
        fraud_count=("fraud", "sum"),                    # NEW
        unusual_time_flag_count=("unusual_time_flag", "sum"),  # NEW
        new_receiver_count=("is_new_receiver", "sum"),   # NEW
    )
    known_receivers = grouped["receiver"].agg(lambda values: sorted(pd.unique(values).tolist()))
    
    for sender, row in summary.iterrows():
        transactions = table.search(sender) or []
        profile_table.insert(
            sender,
            {
                "avg_amount": round(float(row["avg_amount"]), 2),
                "transaction_count": int(row["transaction_count"]),
                "last_transactions": transactions[-5:],
                "last_location": str(row["last_location"]),
                "max_amount": round(float(row["max_amount"]), 2),
                "known_receivers": known_receivers.loc[sender],
                "fraud_count": int(row["fraud_count"]),      # NEW
                "unusual_hours_count": int(row["unusual_time_flag_count"]),  # NEW
                "new_receivers_used": int(row["new_receiver_count"]),  # NEW
                "risk_level": "HIGH" if row["fraud_count"] > 2 else "MEDIUM" if row["fraud_count"] > 0 else "LOW",  # NEW
            }
        )
    
    return profile_table


def get_user_profile(df: pd.DataFrame, sender: str) -> Dict | None:
    """Get user profile by sender"""
    profiles = build_user_profile_hash(df, sample=len(df))
    profile = profiles.search(sender)
    if isinstance(profile, list) and profile:
        return profile[0]
    return None


# ============================================================================
# IMPROVEMENT 5: Enhanced Analysis with More Metrics
# Location: Lines 400-470 (MODIFIED)
# ============================================================================

def run_hashing_analysis(df: pd.DataFrame, sample: int = 500, top_n: int = 12) -> Dict:
    """
    IMPROVED: Run comprehensive hashing analysis with more details
    """
    table, df_s = build_sender_hash_table(df, sample=sample)
    stats = table.bucket_stats()
    
    sender_summaries = []
    if len(df_s):
        sender_frame = df_s.copy()
        sender_frame["sender"] = sender_frame["sender"].astype(str)
        
        payment_methods = sender_frame.groupby("sender", sort=False)["payment_method"].agg(
            lambda values: ", ".join(sorted(pd.unique(values.astype(str)).tolist())[:3]) or "N/A"
        )
        
        summaries = sender_frame.groupby("sender", sort=False).agg(
            transaction_count=("amount", "size"),
            total_amount=("amount", "sum"),
            fraud_count=("fraud", "sum"),
            avg_amount=("amount", "mean"),
            last_location=("location", "last"),
            unusual_hours=("unusual_time_flag", "sum"),      # NEW
            new_receivers=("is_new_receiver", "sum"),        # NEW
            location_changes=("location_change_flag", "sum"), # NEW
        )
        
        summaries["payment_methods"] = payment_methods
        summaries = summaries.reset_index().sort_values(
            ["transaction_count", "total_amount", "sender"],
            ascending=[False, False, True],
        )
        
        sender_summaries = [
            {
                "sender": str(row.sender),
                "transaction_count": int(row.transaction_count),
                "total_amount": round(float(row.total_amount), 2),
                "fraud_count": int(row.fraud_count),
                "payment_methods": str(row.payment_methods),
                "avg_amount": round(float(row.avg_amount), 2),
                "last_location": str(row.last_location),
                "unusual_hours": int(row.unusual_hours),      # NEW
                "new_receivers_used": int(row.new_receivers),  # NEW
                "location_changes": int(row.location_changes),  # NEW
                "risk_score": round(
                    (row.fraud_count * 30 +
                     row.unusual_hours * 15 +
                     row.new_receivers * 20 +
                     row.location_changes * 10) / max(1, row.transaction_count),
                    2
                )  # NEW: Comprehensive risk score
            }
            for row in summaries.head(top_n).itertuples(index=False)
        ]
    
    bucket_view = []
    for bucket in table.display():
        if bucket["chain_length"] > 0:
            bucket_view.append({
                "bucket_index": bucket["bucket_index"],
                "chain_length": bucket["chain_length"],
                "keys": ", ".join(entry["key"] for entry in bucket["entries"][:4]),
            })
    bucket_view.sort(key=lambda item: (-item["chain_length"], item["bucket_index"]))
    
    # IMPROVED Return with more metrics (Line 450)
    return {
        "sample_size": len(df_s),
        "unique_senders": int(df_s["sender"].astype(str).nunique()) if len(df_s) else 0,
        "capacity": table.capacity,
        "size": table.size,
        "load_factor": table.load_factor(),
        "collision_count": table.collisions,
        "used_buckets": stats["used_buckets"],
        "empty_buckets": stats["empty_buckets"],
        "max_bucket_depth": stats["max_bucket_depth"],
        "avg_chain_length": stats["avg_chain_length"],                    # NEW
        "collision_density": stats["collision_density"],                  # NEW
        "search_efficiency": stats["search_efficiency"],                  # NEW
        "total_searches": table.total_searches,                           # NEW
        "failed_searches": table.failed_searches,                         # NEW
        "rehash_count": table.rehash_count,                              # NEW
        "top_sender_groups": sender_summaries[:top_n],
        "bucket_overview": bucket_view[:top_n],
        "table_health": "OPTIMAL" if stats["collision_density"] < 0.3 else "GOOD" if stats["collision_density"] < 0.6 else "NEEDS_RESIZE"  # NEW
    }


def search_sender_in_hash(df: pd.DataFrame, sender: str, sample: int = 500) -> Dict:
    """
    Search for sender in hash table with enriched results
    """
    try:
        from ..caches import caches
    except ImportError:
        caches = None
    
    table = caches.sender_hash if caches and caches.is_fresh() and caches.sender_hash is not None else None
    if table is None:
        table, _ = build_sender_hash_table(df, sample=max(sample, len(df)))
    
    transactions = table.search(sender)
    bucket_index = table.hash_function(sender)
    
    if not transactions:
        return {
            "sender": sender,
            "found": False,
            "bucket_index": bucket_index,
            "transaction_count": 0,
            "total_amount": 0,
            "transactions": [],
            "search_status": "NOT_FOUND"  # NEW
        }
    
    total_amount = round(sum(float(tx["amount"]) for tx in transactions), 2)
    fraud_count = sum(1 for tx in transactions if tx["fraud_flag"] == 1)
    
    # NEW: Enriched response
    return {
        "sender": sender,
        "found": True,
        "bucket_index": bucket_index,
        "transaction_count": len(transactions),
        "total_amount": total_amount,
        "fraud_count": fraud_count,                    # NEW
        "fraud_rate": round(fraud_count / len(transactions) * 100, 2) if transactions else 0,  # NEW
        "transactions": transactions[:20],
        "search_status": "FOUND",                      # NEW
        "risk_assessment": "HIGH" if fraud_count > 2 else "MEDIUM" if fraud_count > 0 else "LOW"  # NEW
    }
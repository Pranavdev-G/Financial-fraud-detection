from dataclasses import dataclass

import pandas as pd


@dataclass
class HashNode:
    key: str
    value: list
    next: "HashNode | None" = None


class HashTable:
    def __init__(self, capacity=17):
        self.capacity = max(5, int(capacity))
        self.table = [None] * self.capacity
        self.size = 0
        self.collisions = 0

    def hash_function(self, key):
        return hash(str(key)) % self.capacity

    def _chain_length(self, index):
        count = 0
        current = self.table[index]
        while current:
            count += 1
            current = current.next
        return count

    def load_factor(self):
        return round(self.size / self.capacity, 4) if self.capacity else 0.0

    def insert(self, key, value):
        if self.load_factor() >= 0.75:
            self._resize()

        index = self.hash_function(key)
        head = self.table[index]
        current = head

        while current:
            if current.key == key:
                if isinstance(current.value, list):
                    if isinstance(value, list):
                        current.value.extend(value)
                    else:
                        current.value.append(value)
                else:
                    current.value = value
                return index
            current = current.next

        if head is not None:
            self.collisions += 1

        new_value = value if isinstance(value, list) else [value]
        new_node = HashNode(str(key), new_value, head)
        self.table[index] = new_node
        self.size += 1
        return index

    def search(self, key):
        index = self.hash_function(key)
        current = self.table[index]

        while current:
            if current.key == str(key):
                return current.value
            current = current.next
        return None

    def delete(self, key):
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

    def display(self):
        rows = []
        for i in range(self.capacity):
            current = self.table[i]
            bucket = []
            while current:
                bucket.append({"key": current.key, "items": len(current.value)})
                current = current.next
            rows.append({"bucket_index": i, "chain_length": len(bucket), "entries": bucket})
        return rows

    def bucket_stats(self):
        lengths = [self._chain_length(i) for i in range(self.capacity)]
        used = sum(1 for x in lengths if x > 0)
        return {
            "used_buckets": used,
            "empty_buckets": self.capacity - used,
            "max_bucket_depth": max(lengths) if lengths else 0,
            "bucket_lengths": lengths,
        }

    def _resize(self):
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


def _transaction_payload(row, index):
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
    }


def build_sender_hash_table(df, sample=500):
    df_s = df.iloc[:sample].reset_index(drop=True)
    estimated = max(17, len(df_s) // 8 if len(df_s) else 17)
    table = HashTable(capacity=estimated)

    for index, row in enumerate(df_s.itertuples(index=False)):
        sender = str(getattr(row, "sender", "UNKNOWN"))
        table.insert(
            sender,
            {
                "index": int(index),
                "sender": sender,
                "receiver": str(getattr(row, "receiver", "N/A")),
                "amount": round(float(getattr(row, "amount", 0)), 2),
                "payment_method": str(getattr(row, "payment_method", "N/A")),
                "location": str(getattr(row, "location", "N/A")),
                "timestamp": str(getattr(row, "timestamp", "")),
                "txn_count_24h": int(getattr(row, "txn_count_24h", 0)),
                "fraud_flag": int(getattr(row, "fraud", 0)),
            },
        )

    return table, df_s


def build_user_profile_hash(df, sample=500):
    table, df_s = build_sender_hash_table(df, sample=sample)
    profile_table = HashTable(capacity=max(17, table.capacity))

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
            },
        )

    return profile_table


def get_user_profile(df, sender):
    profiles = build_user_profile_hash(df, sample=len(df))
    profile = profiles.search(sender)
    if isinstance(profile, list) and profile:
        return profile[0]
    return None


def run_hashing_analysis(df, sample=500, top_n=12):
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
        "top_sender_groups": sender_summaries[:top_n],
        "bucket_overview": bucket_view[:top_n],
    }


def search_sender_in_hash(df, sender, sample=500):
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
        }

    total_amount = round(sum(float(tx["amount"]) for tx in transactions), 2)
    return {
        "sender": sender,
        "found": True,
        "bucket_index": bucket_index,
        "transaction_count": len(transactions),
        "total_amount": total_amount,
        "transactions": transactions[:20],
    }

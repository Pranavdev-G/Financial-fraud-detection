# algorithms/divide_conquer.py
import numpy as np

def merge_sort(arr):
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    return _merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))

def _merge(left, right):
    result = []; i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]: result.append(left[i]); i += 1
        else: result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result

def quick_sort(arr):
    data = arr[:]
    _qs(data, 0, len(data)-1)
    return data

def _qs(arr, low, high):
    if low < high:
        p = _partition(arr, low, high)
        _qs(arr, low, p-1); _qs(arr, p+1, high)

def _partition(arr, low, high):
    pivot = arr[high]; i = low-1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1; arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return i+1

def binary_search(sorted_arr, target):
    lo, hi = 0, len(sorted_arr)-1
    while lo <= hi:
        mid = (lo+hi)//2
        if sorted_arr[mid] == target:
            while mid > 0 and sorted_arr[mid-1] == target: mid -= 1
            return mid
        elif sorted_arr[mid] < target: lo = mid+1
        else: hi = mid-1
    return -1

def run_divide_conquer(df, sample=200):
    amounts = [round(a, 2) for a in df["amount"].tolist()[:sample]]
    ms = merge_sort(amounts)
    qs = quick_sort(amounts)

    mean_val  = float(np.mean(ms))
    std_val   = float(np.std(ms))
    threshold = mean_val + 2 * std_val
    outlier_set = set(v for v in ms if v > threshold)

    outlier_rows = df[df["amount"].isin(outlier_set)].head(30)
    outliers = []
    for _, row in outlier_rows.iterrows():
        outliers.append({
            "sender":         str(row.get("sender","N/A")),
            "receiver":       str(row.get("receiver","N/A")),
            "amount":         round(float(row["amount"]),2),
            "payment_method": str(row.get("payment_method","N/A")),
            "fraud_flag":     int(row.get("fraud",0)),
        })

    return {
        "total_processed":      len(amounts),
        "original_sample":      amounts,
        "merge_sorted_sample":  ms,
        "quick_sorted_sample":  qs,
        "outlier_threshold":    round(threshold,2),
        "outlier_count":        len(outlier_set),
        "outlier_transactions": outliers,
    }

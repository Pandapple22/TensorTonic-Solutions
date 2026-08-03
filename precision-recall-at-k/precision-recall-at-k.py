def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    topk = [1 if v in relevant else 0 for v in recommended[:k]]
    r = len(relevant)
    num = sum(topk)
    return [num/k, num/r]
    
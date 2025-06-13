import math
from collections import defaultdict
from itertools import combinations
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize


def get_top_n(predictions, k=10, threshold=3.5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        if est >= threshold:
            top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:k]
    return top_n


def pr_at_k_from_simple(preds, user_id, test_set, idx2movie, k=10, threshold=0.75):
    wrapped = []
    for rank, (idx, est) in enumerate(preds[:k], start=1):
        movie_id = idx2movie[idx]
        true_r = 1 if movie_id in test_set else 0
        wrapped.append((user_id, movie_id, true_r, est, rank))
    return precision_recall_at_k(wrapped, k=k, threshold=threshold)


def precision_recall_at_k(predictions, k=10, threshold=0.75):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = []
    recalls = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        n_rel = sum(1 for _, true_r in user_ratings if true_r >= threshold)
        n_rec_k = sum(1 for est, _ in top_k if est >= threshold)
        n_rel_and_rec_k = sum(1 for est, true_r in top_k if est >= threshold and true_r >= threshold)

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k else 0.0
        recall = n_rel_and_rec_k / n_rel if n_rel else 0.0

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls) if precisions else (0.0, 0.0)


def coverage(predictions, train_df, all_items, threshold=0.75):
    seen_per_user = defaultdict(set)
    for _, row in train_df.iterrows():
        seen_per_user[row["userId"]].add(row["movieId"])

    recs_per_user = defaultdict(set)
    for uid, iid, _, est, _ in predictions:
        if est >= threshold:
            recs_per_user[uid].add(iid)

    coverages = []
    for uid, seen_items in seen_per_user.items():
        unseen = set(all_items) - seen_items
        if not unseen:
            coverages.append(0.0)
            continue
        C_u = recs_per_user.get(uid, set()) & unseen
        cov_u = len(C_u) / len(unseen)
        coverages.append(cov_u)

    return float(np.mean(coverages)) if coverages else 0.0


def serendipity(predictions, user_profiles, embeddings, movieId_to_index,
                threshold=0.75, similarity_threshold=0.75, top_k=10):
    embeddings_norm = normalize(embeddings, axis=1)
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((iid, true_r, est))

    serendipities = []
    for uid, rec_list in user_est_true.items():
        profile = user_profiles.get(uid)
        if profile is None:
            continue
        filtered = [(iid, true_r, est) for iid, true_r, est in rec_list if est >= threshold]
        if not filtered:
            serendipities.append(0.0)
            continue
        filtered.sort(key=lambda x: x[2], reverse=True)
        topk = filtered[:top_k]
        profile_norm = profile / (np.linalg.norm(profile) + 1e-8)
        total_rec = len(topk)
        ser_count = 0
        for iid, true_r, _ in topk:
            if true_r < threshold:
                continue
            idx = movieId_to_index.get(iid)
            if idx is None:
                continue
            item_emb = embeddings_norm[idx]
            sim = np.dot(profile_norm, item_emb)
            if sim < similarity_threshold:
                ser_count += 1
        serendipities.append(ser_count / total_rec)

    return float(np.mean(serendipities)) if serendipities else 0.0


def diversity(recommendations, item_embeddings):
    pairs = list(combinations(recommendations, 2))
    if not pairs:
        return 0.0
    distances = [
        cosine_distances([item_embeddings[i]], [item_embeddings[j]])[0][0]
        for i, j in pairs
    ]
    return float(np.mean(distances))


def novelty(recommendations, item_popularity):
    if not recommendations:
        return 0.0
    total = 0.0
    for iid in recommendations:
        p_i = item_popularity.get(iid, 1e-9)
        if p_i <= 0:
            p_i = 1e-9
        total += -math.log2(p_i)
    return total / len(recommendations)


def compute_ndcg(predictions, k=10, threshold=0.75):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((true_r, est))

    ndcgs = []
    for rec_list in user_est_true.values():
        filtered = [(true_r, est) for true_r, est in rec_list if est >= threshold]
        if not filtered:
            ndcgs.append(0.0)
            continue
        filtered.sort(key=lambda x: x[1], reverse=True)
        topk = filtered[:k]
        n_rel = sum(1 for true_r, _ in filtered if true_r >= threshold)

        dcg = sum(
            ((2 ** (1 if true_r >= threshold else 0)) - 1) / math.log2(i + 2)
            for i, (true_r, _) in enumerate(topk)
        )

        idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel, k)))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def compute_ndcg_fuzzy(recommended, liked_test_ids, embeddings, movieId_to_index, k=10, sim_threshold=0.75):
    matched_test_ids = set()
    liked_test_vectors = {
        mid: embeddings[movieId_to_index[mid]]
        for mid in liked_test_ids if mid in movieId_to_index
    }

    dcg = 0.0
    for i, rec_mid in enumerate(recommended[:k]):
        is_relevant = False
        if rec_mid in liked_test_ids and rec_mid not in matched_test_ids:
            is_relevant = True
            matched_test_ids.add(rec_mid)
        elif rec_mid in movieId_to_index:
            rec_vec = embeddings[movieId_to_index[rec_mid]]
            for test_mid, test_vec in liked_test_vectors.items():
                if test_mid not in matched_test_ids:
                    sim = np.dot(rec_vec, test_vec)
                    if sim >= sim_threshold:
                        is_relevant = True
                        matched_test_ids.add(test_mid)
                        break
        if is_relevant:
            dcg += 1 / math.log2(i + 2)

    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(liked_test_ids), k)))
    return dcg / idcg if idcg else 0.0

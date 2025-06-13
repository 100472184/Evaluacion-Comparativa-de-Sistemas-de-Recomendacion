import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

from metrics import (
    precision_recall_at_k,
    compute_ndcg,
    compute_ndcg_fuzzy,
    serendipity,
    diversity,
    novelty
)

PREDICTIONS_PATH = r"C:\Users\usuario\TFG-Justin\results\openp5\predicciones_final.csv"
RATINGS_PATH = r"C:\Users\usuario\TFG-Justin\datasets\ml25m_filtrado_big.csv"
MOVIES_PATH = r"C:\Users\usuario\TFG-Justin\datasets\movielens-25m\movies_with_overview.csv"
EMBED_PATH = r"C:\Users\usuario\TFG-Justin\datasets\movies_embeddings_content_based.npy"
METRICS_OUTPUT = r"C:\Users\usuario\TFG-Justin\results\openp5\metrics.csv"

SIM_THRESHOLD = 0.75
TOP_K = 10

print("Cargando datos...")
preds_df = pd.read_csv(PREDICTIONS_PATH)
ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)
embeddings = normalize(np.load(EMBED_PATH), axis=1)

movie2idx = {mid: i for i, mid in enumerate(movies.movieId)}
popularity = ratings.movieId.value_counts(normalize=True).to_dict()

print("Construyendo conjunto de test por usuario...")
test_items_per_user = ratings.groupby("userId")["movieId"].apply(set).to_dict()

print("Construyendo perfiles de usuario...")
valid_ratings = ratings[ratings.movieId.isin(movie2idx)]
user_profiles = {}

for uid, grp in valid_ratings.groupby("userId"):
    idxs = [
        movie2idx[m] for m in grp.movieId
        if m in movie2idx and movie2idx[m] < len(embeddings)
    ]
    if idxs:
        user_profiles[uid] = np.mean(embeddings[idxs], axis=0)

print("Perfiles generados.")

print("Evaluando recomendaciones...")
grouped = preds_df.groupby("userId")
user_metrics = []

for user_id, group in tqdm(grouped, desc="Evaluando usuarios"):
    preds = group["predicted"].tolist()
    test_items = test_items_per_user.get(user_id, set())

    if not test_items or user_id not in user_profiles:
        continue

    wrapped = []
    for rank, pred in enumerate(preds[:TOP_K], start=1):
        rel = 1.0 if pred in test_items else 0.0
        wrapped.append((user_id, pred, rel, 1.0, rank))

    pr, rc = precision_recall_at_k(wrapped, k=TOP_K, threshold=SIM_THRESHOLD)
    ndcg = compute_ndcg(wrapped, k=TOP_K, threshold=SIM_THRESHOLD)

    try:
        ndcgf = compute_ndcg_fuzzy(
            preds[:TOP_K],
            test_items,
            embeddings,
            movie2idx,
            k=TOP_K,
            sim_threshold=SIM_THRESHOLD
        )
    except Exception:
        ndcgf = 0.0

    try:
        wrapped_valid = [
            w for w in wrapped
            if w[1] in movie2idx and movie2idx[w[1]] < len(embeddings)
        ]
        ser = serendipity(
            wrapped_valid,
            user_profiles,
            embeddings,
            movie2idx,
            threshold=SIM_THRESHOLD,
            similarity_threshold=SIM_THRESHOLD,
            top_k=TOP_K
        )
    except Exception:
        ser = 0.0

    emb_idxs = [
        movie2idx[i] for i in preds[:TOP_K]
        if i in movie2idx and movie2idx[i] < len(embeddings)
    ]

    div = diversity(emb_idxs, embeddings) if emb_idxs else 0.0
    nov = novelty(preds[:TOP_K], popularity)
    cov = float(any(i in test_items for i in preds[:TOP_K]))

    user_metrics.append({
        f"P@{TOP_K}": pr,
        f"R@{TOP_K}": rc,
        f"nDCG@{TOP_K}": ndcg,
        f"nDCGf@{TOP_K}": ndcgf,
        "Coverage": cov,
        "Serendipity": ser,
        "Diversity": div,
        "Novelty": nov
    })

print("Calculando métricas finales...")
df_um = pd.DataFrame(user_metrics)
df_metrics = pd.DataFrame([df_um.mean().round(4)])
os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)
df_metrics.to_csv(METRICS_OUTPUT, index=False)

print("Métricas guardadas en:", METRICS_OUTPUT)
print(df_metrics.to_string(index=False))

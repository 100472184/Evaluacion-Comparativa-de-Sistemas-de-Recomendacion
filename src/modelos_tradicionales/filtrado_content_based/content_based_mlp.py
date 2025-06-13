import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../utils")))

from metrics import (
    precision_recall_at_k,
    compute_ndcg,
    compute_ndcg_fuzzy,
    serendipity,
    diversity,
    novelty
)

RATINGS_PATH = "datasets/ml25m_filtrado_big.csv"
MOVIES_PATH = "datasets/movielens-25m/movies_with_overview.csv"
EMBED_PATH = "datasets/movies_embeddings_content_based.npy"
OUTPUT_METRICS = "results/content_based/mlp_noleak/metrics.csv"

TOP_K = 10
SIM_THRESHOLD = 0.75
LR = 1e-3
BATCH_SIZE = 512
EPOCHS = 10
NEG_SAMPLES = 2

ratings = pd.read_csv(RATINGS_PATH)
ratings["weight"] = ratings["rating"] / 5.0
movies = pd.read_csv(MOVIES_PATH)

ratings = ratings[ratings.movieId.isin(movies.movieId)]
movies = movies[movies.movieId.isin(ratings.movieId)].reset_index(drop=True)

if os.path.exists(EMBED_PATH):
    embeddings = normalize(np.load(EMBED_PATH), axis=1)
else:
    raise ValueError("Embeddings no encontrados")

movie2idx = {mid: i for i, mid in enumerate(movies.movieId)}
idx2movie = {i: mid for mid, i in movie2idx.items()}
popularity = ratings.movieId.value_counts(normalize=True).to_dict()

train_df, test_df = [], []
for uid, grp in ratings.groupby("userId"):
    if len(grp) < 5:
        continue
    tr, te = train_test_split(grp, test_size=0.2, random_state=42)
    train_df.append(tr)
    test_df.append(te)

train_df = pd.concat(train_df).reset_index(drop=True)
test_df = pd.concat(test_df).reset_index(drop=True)

train_df = train_df[train_df.movieId.isin(movie2idx)]
test_df = test_df[test_df.movieId.isin(movie2idx)]
valid_users = set(train_df.userId)
test_df = test_df[test_df.userId.isin(valid_users)].reset_index(drop=True)


def build_user_profiles(df, movie2idx, embeddings):
    user_profiles = {}
    for uid, grp in df.groupby("userId"):
        idxs = [movie2idx[m] for m in grp.movieId if m in movie2idx]
        if not idxs:
            continue
        vecs = embeddings[idxs]
        ws = grp.rating.values
        user_profiles[uid] = np.average(vecs, axis=0, weights=ws)
    return user_profiles


train_user_profiles = build_user_profiles(train_df, movie2idx, embeddings)
test_user_profiles = build_user_profiles(test_df, movie2idx, embeddings)


class MovieDataset(Dataset):
    def __init__(self, df, user_profiles, movie2idx, neg_samples=1):
        self.data = []
        all_items = set(movie2idx.keys())
        for uid, grp in df.groupby("userId"):
            if uid not in user_profiles:
                continue
            pos = set(grp.movieId)
            for m in pos:
                self.data.append((uid, movie2idx[m], 1))
            negs = list(all_items - pos)
            n_neg = len(pos) * neg_samples
            neg_idxs = np.random.choice(
                [movie2idx[x] for x in negs],
                min(n_neg, len(negs)), replace=False
            )
            for ni in neg_idxs:
                self.data.append((uid, ni, 0))
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uid, iidx, label = self.data[idx]
        return (
            torch.tensor(self.user_profiles[uid], dtype=torch.float32),
            torch.tensor(embeddings[iidx], dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )


class ContentMLP(nn.Module):
    def __init__(self, emb_dim=384, hidden=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, uemb, iemb):
        x = torch.cat([uemb, iemb], dim=1)
        return self.fc(x).squeeze(1)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = ContentMLP(embeddings.shape[1]).to(device)
opt = optim.Adam(model.parameters(), lr=LR)
crit = nn.BCELoss()

train_loader = DataLoader(MovieDataset(train_df, train_user_profiles, movie2idx, NEG_SAMPLES), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(MovieDataset(test_df, train_user_profiles, movie2idx, NEG_SAMPLES), batch_size=BATCH_SIZE, shuffle=False)


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = total_acc = count = 0
    for uemb, iemb, lbl in loader:
        uemb, iemb, lbl = uemb.to(device), iemb.to(device), lbl.to(device)
        with torch.set_grad_enabled(train):
            out = model(uemb, iemb)
            loss = crit(out, lbl)
            pred = (out > 0.5).float()
            acc = (pred == lbl).float().mean()
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        b = uemb.size(0)
        total_loss += loss.item() * b
        total_acc += acc.item() * b
        count += b
    return total_loss / count, total_acc / count


for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, True)
    test_loss, test_acc = run_epoch(test_loader, False)
    print(f"Ep{epoch:2d} tr_l={train_loss:.4f} tr_a={train_acc:.4f} | te_l={test_loss:.4f} te_a={test_acc:.4f}")

model.eval()
user_metrics = []

with torch.no_grad():
    all_iemb = torch.tensor(embeddings, dtype=torch.float32, device=device)
    for user_id, uemb in test_user_profiles.items():
        if user_id not in test_df.userId.values:
            continue
        seen = set(train_df[train_df.userId == user_id].movieId)
        test_items = set(test_df[test_df.userId == user_id].movieId)
        if not test_items:
            continue
        candidates = [i for i, m in idx2movie.items() if m not in seen]
        u = torch.tensor(uemb, dtype=torch.float32, device=device).unsqueeze(0)
        c = all_iemb[candidates]
        out = model(u.expand(c.size(0), -1), c).cpu().numpy()
        preds = [(idx2movie[candidates[i]], out[i]) for i in range(len(out))]
        preds.sort(key=lambda x: x[1], reverse=True)
        topk = [m for m, _ in preds[:TOP_K]]

        wrapped = [
            (user_id, iid, 1.0 if iid in test_items else 0.0, score, rank + 1)
            for rank, (iid, score) in enumerate(preds[:TOP_K])
        ]

        pr, rc = precision_recall_at_k(wrapped, k=TOP_K, threshold=SIM_THRESHOLD)
        ndcg = compute_ndcg(wrapped, k=TOP_K, threshold=SIM_THRESHOLD)
        ndcgf = compute_ndcg_fuzzy(topk, test_items, embeddings, movie2idx, k=TOP_K, sim_threshold=SIM_THRESHOLD)
        ser = serendipity(wrapped, test_user_profiles, embeddings, movie2idx, threshold=SIM_THRESHOLD, similarity_threshold=SIM_THRESHOLD, top_k=TOP_K)
        div = diversity([movie2idx[i] for i in topk if i in movie2idx], embeddings)
        nov = novelty(topk, popularity)
        cov = len(set(topk) & test_items) / len(test_items)

        user_metrics.append({
            f"P@{TOP_K}": round(pr, 4),
            f"R@{TOP_K}": round(rc, 4),
            f"nDCG@{TOP_K}": round(ndcg, 4),
            f"nDCGf@{TOP_K}": round(ndcgf, 4),
            "Coverage": round(cov, 4),
            "Serendipity": round(ser, 4),
            "Diversity": round(div, 4),
            "Novelty": round(nov, 4)
        })

os.makedirs(os.path.dirname(OUTPUT_METRICS), exist_ok=True)
df_metrics = pd.DataFrame([pd.DataFrame(user_metrics).mean().round(4)])
df_metrics.to_csv(OUTPUT_METRICS, index=False)
print("MÉTRICAS SIN LEAKAGE:")
print(df_metrics.to_string(index=False))

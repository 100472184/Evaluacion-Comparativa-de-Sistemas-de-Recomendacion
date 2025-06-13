import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

RATINGS_PATH = "datasets/ml25m_filtrado_big.csv"
MOVIES_PATH = "datasets/movielens-25m/movies_with_overview.csv"
TAGS_PATH = "datasets/movielens-25m/tags.csv"
EMBEDDINGS_PATH = "datasets/movies_embeddings.npy"

MODEL_OUTPUT_PATH = "results/mlp_embeddings/final/model_mlp.pt"
METRICS_OUTPUT_PATH = "results/mlp_embeddings/final/metrics_mlp.csv"
PREDICTIONS_OUTPUT_PATH = "results/mlp_embeddings/final/predictions_mlp.csv"

HIDDEN_DIM = 128
LR = 0.004522085151572059
DROPOUT = 0.2931876371842046
WEIGHT_DECAY = 1.06568992662591e-06

BATCH_SIZE = 1024
TOP_K = 10
THRESHOLD = 3.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)
valid_movie_ids = ratings.movieId.unique()

movies = movies[movies.movieId.isin(valid_movie_ids)].reset_index(drop=True)
ratings = ratings[ratings.movieId.isin(valid_movie_ids)].copy()
ratings["userId_enc"] = ratings.userId.astype("category").cat.codes
code2user = dict(enumerate(ratings.userId.astype("category").cat.categories))

train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df[test_df.userId_enc.isin(set(train_df.userId_enc))].reset_index(drop=True)

if os.path.exists(EMBEDDINGS_PATH):
    embeddings = np.load(EMBEDDINGS_PATH)
else:
    tags = pd.read_csv(TAGS_PATH) if os.path.exists(TAGS_PATH) else pd.DataFrame()
    movies["overview"] = movies.overview.fillna("")
    movies["genres"] = movies.genres.fillna("").str.replace("|", " ")
    movies["title"] = movies.title.fillna("")
    movies["tags"] = ""

    if not tags.empty:
        tags["tag"] = tags["tag"].fillna("").astype(str)
        tag_text = tags.groupby("movieId")["tag"].agg(" ".join)
        movies["tags"] = movies["movieId"].map(tag_text).fillna("")

    movies["text"] = (
        (movies["genres"] + " ") * 3 +
        (movies["tags"] + " ") * 2 +
        (movies["title"] + " ") * 3 +
        (movies["overview"] + " ") * 3
    )

    model_emb = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
    embeddings = model_emb.encode(movies.text.tolist(), show_progress_bar=True, device=DEVICE)
    embeddings = normalize(embeddings, axis=1)
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

all_movies = pd.read_csv(MOVIES_PATH)
movies = all_movies[all_movies.movieId.isin(valid_movie_ids)].reset_index(drop=True)

user_profiles = {}
movie_to_idx = {mid: i for i, mid in enumerate(movies.movieId)}
uids = train_df.userId_enc.unique()
u2i = {u: i for i, u in enumerate(uids)}

for u, grp in train_df.groupby("userId_enc"):
    idxs = grp.movieId.map(movie_to_idx).values
    user_profiles[u2i[u]] = np.average(embeddings[idxs], axis=0, weights=grp.rating.values)


class HybridDataset(Dataset):
    def __init__(self, df, user2idx, movie2idx, user_profiles):
        self.u = df.userId_enc.map(user2idx).values
        self.i = df.movieId.map(movie2idx).values
        self.r = df.rating.values.astype(np.float32)
        self.up = user_profiles

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        u = self.u[idx]
        it = self.i[idx]
        return (
            torch.tensor(self.up[u], dtype=torch.float32),
            torch.tensor(embeddings[it], dtype=torch.float32),
            torch.tensor(u, dtype=torch.long),
            torch.tensor(it, dtype=torch.long),
            torch.tensor(self.r[idx], dtype=torch.float32)
        )


class HybridMLP(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout, n_users):
        super().__init__()
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Sequential(nn.Linear(emb_dim, 1), nn.Tanh())
        self.global_bias = nn.Parameter(torch.tensor([0.0]))
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, up, ie, uidx):
        h = self.mlp(torch.cat([up, ie], dim=1)).squeeze()
        b_u = self.user_bias(uidx).squeeze()
        b_i = self.item_bias(ie).squeeze()
        return h + b_u + b_i + self.global_bias


train_ds = HybridDataset(train_df, u2i, movie_to_idx, user_profiles)
test_ds = HybridDataset(test_df, u2i, movie_to_idx, user_profiles)
train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = HybridMLP(embeddings.shape[1], HIDDEN_DIM, DROPOUT, len(user_profiles)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

start_time = time.time()

for ep in range(EPOCHS):
    model.train()
    total_loss = 0
    for up, ie, uidx, item_dx, r in train_ld:
        up, ie, uidx, r = up.to(DEVICE), ie.to(DEVICE), uidx.to(DEVICE), r.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(up, ie, uidx), r)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {ep}/{EPOCHS} - Loss: {total_loss / len(train_ld):.4f}")

end_time = time.time()
total_time = end_time - start_time

os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_OUTPUT_PATH)

model.eval()
records = []
preds, trues = [], []

with torch.no_grad():
    for up, ie, uidx, item_idx, r in test_ld:
        up, ie, uidx = up.to(DEVICE), ie.to(DEVICE), uidx.to(DEVICE)
        p = model(up, ie, uidx).cpu().numpy()
        for u, i_idx, true, est in zip(uidx.cpu().numpy(), item_idx.cpu().numpy(), r.numpy(), p):
            records.append((int(u), int(i_idx), float(true), float(est)))
        preds.extend(p)
        trues.extend(r.numpy())

preds_arr = np.clip(np.array(preds), 0.5, 5.0)
trues_arr = np.array(trues)

rmse = np.sqrt(np.mean((preds_arr - trues_arr) ** 2))
mae = np.mean(np.abs(preds_arr - trues_arr))
r2 = r2_score(trues_arr, preds_arr)

os.makedirs(os.path.dirname(METRICS_OUTPUT_PATH), exist_ok=True)
pd.DataFrame([{
    "RMSE": round(rmse, 4),
    "MAE": round(mae, 4),
    "R2": round(r2, 4),
    "Training_time": round(total_time, 4)
}]).to_csv(METRICS_OUTPUT_PATH, index=False)

idx2movie = {idx: mid for mid, idx in movie_to_idx.items()}
df_preds = pd.DataFrame(records, columns=["user_enc", "item_idx", "true", "predicted"])
df_preds["userId"] = df_preds["user_enc"].map(code2user)
df_preds["movieId"] = df_preds["item_idx"].map(idx2movie)

os.makedirs(os.path.dirname(PREDICTIONS_OUTPUT_PATH), exist_ok=True)
df_preds[["userId", "movieId", "true", "predicted"]].to_csv(PREDICTIONS_OUTPUT_PATH, index=False)

print("Evaluación completada:")
print(f" RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
print(f" Predicciones guardadas en {PREDICTIONS_OUTPUT_PATH}")
print(f" Métricas guardadas en {METRICS_OUTPUT_PATH}")
print(f" Modelo guardado en {MODEL_OUTPUT_PATH}")
print("Fin del script.")

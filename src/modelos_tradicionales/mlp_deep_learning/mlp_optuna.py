import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, KFold
import optuna

RATINGS_PATH = "datasets/ml25m_filtrado_big.csv"
MOVIES_PATH = "datasets/movielens-25m/movies_with_overview.csv"
TAGS_PATH = "datasets/movielens-25m/tags.csv"
EMBEDDINGS_PATH = "datasets/movies_embeddings_mlp.npy"

OUTPUT_METRICS = "results/mlp_embeddings/optuna/resultados_mlp_embeddings_optuna.csv"
OUTPUT_PRED = "results/mlp_embeddings/optuna/predicciones_mlp_embeddings.csv"

TOP_K = 10
THRESHOLD = 3.5
BATCH_SIZE = 1024
CV_FOLDS = 3
N_TRIALS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Usando dispositivo: {DEVICE}")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

valid_ids = ratings.movieId.unique()
movies = movies[movies.movieId.isin(valid_ids)].reset_index(drop=True)
ratings = ratings[ratings.movieId.isin(valid_ids)].copy()
ratings["userId_enc"] = ratings.userId.astype("category").cat.codes

if os.path.exists(EMBEDDINGS_PATH):
    embeddings = np.load(EMBEDDINGS_PATH)
else:
    tags = pd.read_csv(TAGS_PATH) if os.path.exists(TAGS_PATH) else pd.DataFrame()
    movies["overview"] = movies.overview.fillna("")
    movies["genres"] = movies.genres.fillna("").str.replace("|", " ")
    movies["title"] = movies.title.fillna("")
    movies["tags"] = ""
    if not tags.empty:
        tag_txt = tags.groupby("movieId")["tag"].apply(lambda vs: " ".join(vs))
        movies = movies.merge(tag_txt.rename("tags"), left_on="movieId", right_index=True, how="left").fillna({"tags": ""})
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

movie_to_idx = {mid: i for i, mid in enumerate(movies.movieId)}

train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df[test_df.userId_enc.isin(set(train_df.userId_enc))].reset_index(drop=True)


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


def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=64)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-2, log=True)

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    rmses = []

    for tr_idx, vl_idx in kf.split(train_df):
        tr = train_df.iloc[tr_idx].reset_index(drop=True)
        vl = train_df.iloc[vl_idx].reset_index(drop=True)

        uids = tr.userId_enc.unique()
        mids = tr.movieId.unique()
        u2i = {u: i for i, u in enumerate(uids)}
        m2i = {m: i for i, m in enumerate(mids)}
        profiles = {
            u2i[u]: np.average(embeddings[grp.movieId.map(m2i)], axis=0, weights=grp.rating.values)
            for u, grp in tr.groupby("userId_enc")
        }

        tr_ld = DataLoader(HybridDataset(tr, u2i, m2i, profiles), batch_size=BATCH_SIZE, shuffle=True)
        vl_ld = DataLoader(HybridDataset(vl, u2i, m2i, profiles), batch_size=BATCH_SIZE)

        model = HybridMLP(embeddings.shape[1], hidden_dim, dropout, len(u2i)).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()

        for _ in range(5):
            model.train()
            for up, ie, uidx, r in tr_ld:
                up, ie, uidx, r = up.to(DEVICE), ie.to(DEVICE), uidx.to(DEVICE), r.to(DEVICE)
                opt.zero_grad()
                loss_fn(model(up, ie, uidx), r).backward()
                opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for up, ie, uidx, r in vl_ld:
                up, ie, uidx = up.to(DEVICE), ie.to(DEVICE), uidx.to(DEVICE)
                preds.append(model(up, ie, uidx).cpu().numpy())
                trues.append(r.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        rmse = np.sqrt(((preds - trues) ** 2).mean())
        rmses.append(rmse)

        trial.report(rmse, len(rmses))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(rmses))


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_METRICS), exist_ok=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS)

    best = study.best_trial.params

    mids = train_df.movieId.unique()
    m2i = {m: i for i, m in enumerate(mids)}
    uids = train_df.userId_enc.unique()
    u2i = {u: i for i, u in enumerate(uids)}
    profiles = {
        u2i[u]: np.average(embeddings[grp.movieId.map(m2i)], axis=0, weights=grp.rating.values)
        for u, grp in train_df.groupby("userId_enc")
    }

    tr_ld = DataLoader(HybridDataset(train_df, u2i, m2i, profiles), batch_size=BATCH_SIZE, shuffle=True)
    te_ld = DataLoader(HybridDataset(test_df, u2i, m2i, profiles), batch_size=BATCH_SIZE)

    model = HybridMLP(embeddings.shape[1], best["hidden_dim"], best["dropout"], len(u2i)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=best["lr"], weight_decay=best["wd"])
    loss_fn = nn.MSELoss()

    for _ in range(10):
        model.train()
        for up, ie, uidx, r in tr_ld:
            up, ie, uidx, r = up.to(DEVICE), ie.to(DEVICE), uidx.to(DEVICE), r.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(up, ie, uidx), r).backward()
            opt.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for up, ie, uidx, r in te_ld:
            up, ie, uidx = up.to(DEVICE), ie.to(DEVICE), uidx.to(DEVICE)
            preds.append(model(up, ie, uidx).cpu().numpy())
            trues.append(r.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    rmse = np.sqrt(((preds - trues) ** 2).mean())
    mae = np.mean(np.abs(preds - trues))

    pd.DataFrame([{**best, "RMSE": rmse}]).to_csv(OUTPUT_METRICS, index=False)
    os.makedirs(os.path.dirname(OUTPUT_PRED), exist_ok=True)
    df_out = test_df.copy()
    df_out["pred"] = np.round(preds, 2)
    df_out[["userId_enc", "movieId", "rating", "pred"]].rename(columns={"rating": "true"}).to_csv(OUTPUT_PRED, index=False)

    print("Best params:", best)
    print(f"Final RMSE={rmse:.4f}, MAE={mae:.4f}")

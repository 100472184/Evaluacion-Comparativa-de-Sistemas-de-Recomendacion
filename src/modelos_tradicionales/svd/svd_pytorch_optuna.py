import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
import optuna
from collections import defaultdict

DATASET_PATH = "datasets/ml25m_filtrado_big.csv"
RESULTS_CSV = "results/svd_pytorch/resultados_hiperparametros.csv"
PRED_CSV = "results/svd_pytorch/predicciones.csv"

TOP_K = 10
THRESHOLD = 3.5
BATCH_SIZE = 2048
CV_FOLDS = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(DATASET_PATH)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

valid_users = train_df.userId.unique()
valid_items = train_df.movieId.unique()
test_df = test_df[
    test_df.userId.isin(valid_users) &
    test_df.movieId.isin(valid_items)
].reset_index(drop=True)

user_ids = train_df.userId.unique()
item_ids = train_df.movieId.unique()
user2idx = {uid: i for i, uid in enumerate(user_ids)}
item2idx = {iid: i for i, iid in enumerate(item_ids)}
idx2user = {i: uid for uid, i in user2idx.items()}
idx2item = {i: iid for i, iid in item2idx.items()}

global_mean = train_df.rating.mean()


class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = df.userId.map(user2idx).values
        self.items = df.movieId.map(item2idx).values
        self.ratings = df.rating.values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float)
        )


class SVDModel(nn.Module):
    def __init__(self, n_users, n_items, emb_size, gm):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, emb_size)
        self.item_embed = nn.Embedding(n_items, emb_size)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor([gm], dtype=torch.float32))

    def forward(self, u, i):
        u_vec = self.user_embed(u)
        i_vec = self.item_embed(i)
        dot = (u_vec * i_vec).sum(dim=1)
        b_u = self.user_bias(u).squeeze()
        b_i = self.item_bias(i).squeeze()
        return self.global_bias + b_u + b_i + dot


def objective(trial):
    emb_size = trial.suggest_int("emb_size", 16, 64, step=16)
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    n_epochs = trial.suggest_int("epochs", 10, 30, step=10)
    opt_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    rmses = []

    for tr_idx, val_idx in kf.split(train_df):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        vl_df = train_df.iloc[val_idx].reset_index(drop=True)

        tr_users = set(tr_df.userId)
        tr_items = set(tr_df.movieId)
        vl_df = vl_df[
            vl_df.userId.isin(tr_users) &
            vl_df.movieId.isin(tr_items)
        ].reset_index(drop=True)

        tr_ds = RatingsDataset(tr_df)
        vl_ds = RatingsDataset(vl_df)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        vl_loader = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = SVDModel(len(user2idx), len(item2idx), emb_size, global_mean).to(device)
        optimizer = getattr(torch.optim, opt_name)(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(n_epochs):
            model.train()
            for u, i, r in tr_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                optimizer.zero_grad()
                loss_fn(model(u, i), r).backward()
                optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for u, i, r in vl_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                preds.append(model(u, i).cpu().numpy())
                trues.append(r.cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        rmses.append(np.sqrt(((preds - trues) ** 2).mean()))

    return float(np.mean(rmses))


if __name__ == "__main__":
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best = study.best_trial
    params = best.params

    model = SVDModel(
        len(user2idx), len(item2idx),
        params["emb_size"], global_mean
    ).to(device)
    optimizer = getattr(torch.optim, params["optimizer"])(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()
    train_ds = RatingsDataset(train_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(params["epochs"]):
        model.train()
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            optimizer.zero_grad()
            loss_fn(model(u, i), r).backward()
            optimizer.step()

    model.eval()
    preds_list, actual_list, user_list, item_list = [], [], [], []
    test_ds = RatingsDataset(test_df)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for u, i, r in test_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            preds = model(u, i).cpu().numpy()
            preds_list.extend(preds)
            actual_list.extend(r.cpu().numpy())
            user_list.extend(u.cpu().numpy())
            item_list.extend(i.cpu().numpy())

    preds_arr = np.clip(np.array(preds_list), 0.5, 5.0)
    actual_arr = np.array(actual_list)

    rmse = np.sqrt(((preds_arr - actual_arr) ** 2).mean())
    mae = np.mean(np.abs(preds_arr - actual_arr))

    user_preds = defaultdict(list)
    user_rel = defaultdict(set)
    for u, it, tr, pr in zip(user_list, item_list, actual_arr, preds_arr):
        user_preds[u].append((it, pr))
        if tr >= THRESHOLD:
            user_rel[u].add(it)

    precisions, recalls = [], []
    for u, recs in user_preds.items():
        ranked = sorted(recs, key=lambda x: x[1], reverse=True)[:TOP_K]
        rec_set = set(i for i, _ in ranked)
        rel_set = user_rel[u]
        precisions.append(len(rec_set & rel_set) / len(rec_set) if rec_set else 0)
        recalls.append(len(rec_set & rel_set) / len(rel_set) if rel_set else 0)

    precision_at_k = np.mean(precisions)
    recall_at_k = np.mean(recalls)

    df_res = pd.DataFrame([{
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        f"Precision@{TOP_K}": round(precision_at_k, 4),
        f"Recall@{TOP_K}": round(recall_at_k, 4),
        **params
    }])
    df_res.to_csv(RESULTS_CSV, index=False)

    pd.DataFrame({
        "userId": [user_ids[u] for u in user_list],
        "movieId": [item_ids[i] for i in item_list],
        "true_rating": np.round(actual_arr, 2),
        "predicted_rating": np.round(preds_arr, 2)
    }).to_csv(PRED_CSV, index=False)

    print("Mejor configuración:", params)
    print(f"Test RMSE={rmse:.4f}, MAE={mae:.4f}")
    print(f"P@{TOP_K}={precision_at_k:.4f}, R@{TOP_K}={recall_at_k:.4f}")
    print(f"Resultados guardados en '{RESULTS_CSV}' y '{PRED_CSV}'")

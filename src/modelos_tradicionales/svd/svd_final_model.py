import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "src")))
from utils.metrics import precision_recall_at_k_ratings, coverage

DATASET_PATH = "datasets/ml25m_filtrado_big.csv"
MOVIES_PATH = "datasets/movielens-25m/movies_with_overview.csv"
EMBEDDINGS_PATH = "datasets/movies_embeddings.npy"
MODEL_OUTPUT = "results/svd_pytorch/modelo_final/modelo_final/modelo_entrenado_final.pt"
METRICS_OUTPUT = "results/svd_pytorch/modelo_final/metricas/metricas_final.csv"
PREDICTIONS_OUTPUT = "results/svd_pytorch/modelo_final/predicciones/predicciones_final.csv"

EMBEDDING_SIZE = 16
LR = 0.017048429491956108
EPOCHS = 10
OPTIMIZER = "RMSprop"

BATCH_SIZE = 2048
TOP_K = 10
THRESHOLD = 3.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(DATASET_PATH)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

valid_users = set(train_df.userId)
valid_items = set(train_df.movieId)
test_df = test_df[
    test_df.userId.isin(valid_users) &
    test_df.movieId.isin(valid_items)
].reset_index(drop=True)

movies_df = pd.read_csv(MOVIES_PATH)
movies_df = movies_df[movies_df.movieId.isin(df.movieId)].reset_index(drop=True)
movieId_to_index = {mid: idx for idx, mid in enumerate(movies_df.movieId)}
embeddings = np.load(EMBEDDINGS_PATH)

user_ids = train_df.userId.unique()
item_ids = train_df.movieId.unique()
user2idx = {uid: i for i, uid in enumerate(user_ids)}
item2idx = {iid: i for i, iid in enumerate(item_ids)}
idx2user = {i: uid for uid, i in user2idx.items()}
idx2item = {i: iid for iid, i in item2idx.items()}

encoded_profiles = {}
for uid, grp in train_df.groupby("userId"):
    enc = user2idx[uid]
    idxs = [movieId_to_index[m] for m in grp.movieId if m in movieId_to_index]
    vecs = embeddings[idxs]
    weights = grp.rating.values
    encoded_profiles[enc] = np.average(vecs, axis=0, weights=weights)

global_mean = train_df.rating.mean()


class RatingsDataset(Dataset):
    def __init__(self, df):
        self.u = df.userId.map(user2idx).values
        self.i = df.movieId.map(item2idx).values
        self.r = df.rating.values.astype(np.float32)

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.u[idx], dtype=torch.long),
            torch.tensor(self.i[idx], dtype=torch.long),
            torch.tensor(self.r[idx], dtype=torch.float)
        )


class SVDModel(nn.Module):
    def __init__(self, n_users, n_items, emb_size, gm):
        super().__init__()
        self.uemb = nn.Embedding(n_users, emb_size)
        self.iemb = nn.Embedding(n_items, emb_size)
        self.ubias = nn.Embedding(n_users, 1)
        self.ibias = nn.Embedding(n_items, 1)
        self.gbias = nn.Parameter(torch.tensor([gm], dtype=torch.float32))

    def forward(self, u, i):
        u_v = self.uemb(u)
        i_v = self.iemb(i)
        dot = (u_v * i_v).sum(dim=1)
        b_u = self.ubias(u).squeeze()
        b_i = self.ibias(i).squeeze()
        return self.gbias + b_u + b_i + dot


train_ds = RatingsDataset(train_df)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

model = SVDModel(len(user2idx), len(item2idx), EMBEDDING_SIZE, global_mean).to(device)
optimizer_cls = getattr(torch.optim, OPTIMIZER)
optimizer = optimizer_cls(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

start_time = time.time()

print("Training final SVD...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for u, i, r in train_loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(u, i), r)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS} loss={total_loss / len(train_loader):.4f}")

end_time = time.time()
training_time = end_time - start_time
print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")

os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
torch.save(model.state_dict(), MODEL_OUTPUT)

test_ds = RatingsDataset(test_df)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

preds, trues, users, items = [], [], [], []
model.eval()
with torch.no_grad():
    for u, i, r in test_loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        p = model(u, i).cpu().numpy()
        preds.extend(p)
        trues.extend(r.cpu().numpy())
        users.extend(u.cpu().numpy())
        items.extend(i.cpu().numpy())

preds_arr = np.clip(np.array(preds), 0.5, 5.0)
trues_arr = np.array(trues)

rmse = np.sqrt(np.mean((preds_arr - trues_arr) ** 2))
mae = np.mean(np.abs(preds_arr - trues_arr))
r2 = r2_score(trues_arr, preds_arr)

os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)
pd.DataFrame([{
    "RMSE": round(rmse, 4),
    "MAE": round(mae, 4),
    "R2": round(r2, 4),
    "Training_time": round(training_time, 4),
}]).to_csv(METRICS_OUTPUT, index=False)

os.makedirs(os.path.dirname(PREDICTIONS_OUTPUT), exist_ok=True)
pd.DataFrame({
    "userId": [idx2user[u] for u in users],
    "movieId": [idx2item[i] for i in items],
    "true": np.round(trues_arr, 2),
    "predicted": np.round(preds_arr, 2)
}).to_csv(PREDICTIONS_OUTPUT, index=False)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R2:   {r2:.4f}")

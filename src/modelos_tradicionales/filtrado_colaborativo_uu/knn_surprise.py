import os
import time
import joblib
import numpy as np
import pandas as pd

from surprise import Dataset, Reader, KNNWithMeans, accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DATASET_PATH = "datasets/ml25m_filtrado_big.csv"
MODEL_OUTPUT_PATH = "results/filtrado_uu/final_model/knn_with_means_model.pkl"
PREDICTIONS_OUTPUT_PATH = "results/filtrado_uu/predictions/predicciones.csv"
METRICS_OUTPUT_PATH = "results/filtrado_uu/metrics/metricas.csv"

best_params = {
    "k": 25,
    "min_k": 2,
    "similarity": "pearson",
    "shrinkage": 93
}
RATING_SCALE = (0.5, 5.0)

df = pd.read_csv(DATASET_PATH)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

valid_users = set(train_df.userId)
valid_items = set(train_df.movieId)
test_df = test_df[
    test_df.userId.isin(valid_users) &
    test_df.movieId.isin(valid_items)
].reset_index(drop=True)

reader = Reader(rating_scale=RATING_SCALE)
data_full = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
trainset = data_full.build_full_trainset()

sim_options = {
    "name": best_params["similarity"],
    "user_based": True,
    "shrinkage": best_params["shrinkage"]
}

start_time = time.time()

if os.path.exists(MODEL_OUTPUT_PATH):
    print("Cargando modelo existente...")
    model = joblib.load(MODEL_OUTPUT_PATH)
else:
    print("Entrenando nuevo modelo...")
    model = KNNWithMeans(
        k=best_params["k"],
        min_k=best_params["min_k"],
        sim_options=sim_options
    )
    model.fit(trainset)
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)

training_time = time.time() - start_time
print(f"Modelo listo en {training_time:.2f}s")

test_tuples = list(zip(test_df.userId, test_df.movieId, test_df.rating))
preds = model.test(test_tuples)

rmse = accuracy.rmse(preds, verbose=False)
mae = accuracy.mae(preds, verbose=False)

trues = np.array([true for (_, _, true, _, _) in preds])
ests = np.array([est for (_, _, _, est, _) in preds])
r2 = r2_score(trues, ests)

os.makedirs(os.path.dirname(METRICS_OUTPUT_PATH), exist_ok=True)
pd.DataFrame([{
    "RMSE": round(rmse, 4),
    "MAE": round(mae, 4),
    "R2": round(r2, 4),
    "Training_time_s": round(training_time, 4)
}]).to_csv(METRICS_OUTPUT_PATH, index=False)

os.makedirs(os.path.dirname(PREDICTIONS_OUTPUT_PATH), exist_ok=True)
df_preds = pd.DataFrame([{
    "userId": uid,
    "movieId": iid,
    "true_rating": true,
    "pred_rating": round(est, 2)
} for (uid, iid, true, est, _) in preds])
df_preds.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)

print(f"RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
print(f"Predicciones guardadas en {PREDICTIONS_OUTPUT_PATH}")
print(f"Métricas guardadas en {METRICS_OUTPUT_PATH}")

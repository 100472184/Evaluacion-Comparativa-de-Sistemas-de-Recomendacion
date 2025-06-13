import sys
import os
import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm

from surprise import Dataset, Reader, KNNWithMeans, accuracy
from sklearn.model_selection import train_test_split, KFold
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "src")))
from utils.metrics import get_top_n, precision_recall_at_k_ratings

DATASET_PATH = "datasets/ml25m_filtrado_big.csv"
RESULTS_CSV = "results/filtrado_uu/knn_optuna_resultados.csv"
PRED_CSV = "results/filtrado_uu/knn_predicciones_finales.csv"

TOP_K = 10
THRESHOLD = 3.5
CV_FOLDS = 3
N_TRIALS = 10
RATING_SCALE = (0.5, 5.0)

df = pd.read_csv(DATASET_PATH)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_users = set(train_df.userId)
train_items = set(train_df.movieId)
test_df = test_df[
    test_df.userId.isin(train_users) &
    test_df.movieId.isin(train_items)
].reset_index(drop=True)

reader = Reader(rating_scale=RATING_SCALE)


def objective(trial):
    k = trial.suggest_int("k", 10, 25, step=5)
    min_k = trial.suggest_int("min_k", 1, 20)
    sim_name = trial.suggest_categorical("sim", ["cosine", "pearson"])
    shrink = trial.suggest_int("shrink", 0, 100)

    algo_params = {
        "k": k,
        "min_k": min_k,
        "sim_options": {
            "name": sim_name,
            "user_based": True,
            "shrinkage": shrink
        }
    }

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    rmses = []

    for tr_idx, val_idx in kf.split(train_df):
        df_tr = train_df.iloc[tr_idx]
        df_val = train_df.iloc[val_idx]

        users_tr = set(df_tr.userId)
        items_tr = set(df_tr.movieId)
        df_val = df_val[
            df_val.userId.isin(users_tr) &
            df_val.movieId.isin(items_tr)
        ]

        ds_tr = Dataset.load_from_df(df_tr[['userId', 'movieId', 'rating']], reader)
        tr_set = ds_tr.build_full_trainset()

        algo = KNNWithMeans(**algo_params)
        algo.fit(tr_set)

        val_tuples = list(zip(df_val.userId, df_val.movieId, df_val.rating))
        preds = algo.test(val_tuples)
        rmses.append(accuracy.rmse(preds, verbose=False))

    return float(np.mean(rmses))


if __name__ == "__main__":
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    best = study.best_trial
    print("Mejor params:", best.params)
    print(f"RMSE CV: {best.value:.4f}")

    df_res = pd.DataFrame([{
        "RMSE_CV": round(best.value, 4),
        **best.params
    }])
    df_res.to_csv(RESULTS_CSV, index=False)

    algo_final = KNNWithMeans(
        k=best.params["k"],
        min_k=best.params["min_k"],
        sim_options={
            "name": best.params["sim"],
            "user_based": True,
            "shrink": best.params["shrink"]
        },
        verbose=False
    )

    ds_full = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = ds_full.build_full_trainset()
    algo_final.fit(trainset)

    test_tuples = list(zip(test_df.userId, test_df.movieId, test_df.rating))
    batch_size = 10000
    final_preds = []

    for i in tqdm(range(0, len(test_tuples), batch_size)):
        batch = test_tuples[i:i + batch_size]
        final_preds.extend(algo_final.test(batch))

    rmse = accuracy.rmse(final_preds, verbose=False)
    mae = accuracy.mae(final_preds, verbose=False)
    precision, recall = precision_recall_at_k_ratings(
        final_preds, k=TOP_K, threshold=THRESHOLD
    )

    df_final = pd.DataFrame([{
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        f"Precision@{TOP_K}": round(precision, 4),
        f"Recall@{TOP_K}": round(recall, 4),
        **best.params
    }])
    df_final.to_csv(RESULTS_CSV, index=False)

    df_preds = pd.DataFrame([{
        "userId": uid,
        "movieId": iid,
        "true_rating": tr,
        "pred_rating": pr
    } for uid, iid, tr, pr, _ in final_preds])
    df_preds.to_csv(PRED_CSV, index=False)

    print("Test RMSE:", rmse, "MAE:", mae)
    print(f"P@{TOP_K}={precision:.4f}, R@{TOP_K}={recall:.4f}")
    print("Predicciones guardadas en", PRED_CSV)
    print("Resultados guardados en", RESULTS_CSV)

import os
import optuna
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold
from surprise import Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split as surprise_tt_split

DATASET_PATH = "datasets/ml25m_filtrado_big.csv"
RESULTS_CSV = "results/filtrado_ii/knn_optuna_resultados.csv"
PRED_CSV = "results/filtrado_ii/knn_predicciones_finales.csv"

TOP_K = 10
THRESHOLD = 3.5
CV_FOLDS = 3
N_TRIALS = 20
RATING_SCALE = (0.5, 5.0)


def get_top_n(preds, k=TOP_K, threshold=THRESHOLD):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in preds:
        if est >= threshold:
            top_n[uid].append((iid, est))
    for uid in top_n:
        top_n[uid].sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = top_n[uid][:k]
    return top_n


def precision_recall_at_k(preds, k=TOP_K, threshold=THRESHOLD):
    top_n = get_top_n(preds, k, threshold)
    precisions = []
    recalls = []

    true_by_user = defaultdict(list)
    for uid, iid, true_r, _, _ in preds:
        true_by_user[uid].append((iid, true_r))

    for uid, recs in top_n.items():
        n_rel = sum(1 for _, r in true_by_user[uid] if r >= threshold)
        n_rec_k = len(recs)
        n_rel_and_rec = sum(
            1 for iid, _ in recs
            if any(iid == true_iid and true_r >= threshold for true_iid, true_r in true_by_user[uid])
        )
        precisions.append(n_rel_and_rec / n_rec_k if n_rec_k else 0)
        recalls.append(n_rel_and_rec / n_rel if n_rel else 0)

    return np.mean(precisions), np.mean(recalls)


df = pd.read_csv(DATASET_PATH)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_users = set(train_df.userId)
train_items = set(train_df.movieId)
test_df = test_df[
    test_df.userId.isin(train_users) & test_df.movieId.isin(train_items)
].reset_index(drop=True)

reader = Reader(rating_scale=RATING_SCALE)


def objective(trial):
    k = trial.suggest_int("k", 10, 50, step=5)
    min_k = trial.suggest_int("min_k", 1, 20)
    sim_name = trial.suggest_categorical("sim", ["cosine", "pearson"])
    shrink = trial.suggest_int("shrink", 0, 100)

    algo_params = {
        "k": k,
        "min_k": min_k,
        "sim_options": {
            "name": sim_name,
            "user_based": False,
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
            df_val.userId.isin(users_tr) & df_val.movieId.isin(items_tr)
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
            "user_based": False,
            "shrink": best.params["shrink"]
        },
        verbose=False
    )
    ds_full = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = ds_full.build_full_trainset()
    algo_final.fit(trainset)

    test_tuples = list(zip(test_df.userId, test_df.movieId, test_df.rating))
    final_preds = algo_final.test(test_tuples)

    rmse = accuracy.rmse(final_preds, verbose=False)
    mae = accuracy.mae(final_preds, verbose=False)
    precision, recall = precision_recall_at_k(final_preds, k=TOP_K, threshold=THRESHOLD)

    df_final = pd.DataFrame([{
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        f"Precision@{TOP_K}": round(precision, 4),
        f"Recall@{TOP_K}": round(recall, 4),
        **best.params
    }])
    df_final.to_csv(RESULTS_CSV, index=False)

    pd.DataFrame([{
        "userId": uid,
        "movieId": iid,
        "true_rating": tr,
        "pred_rating": pr
    } for uid, iid, tr, pr, _ in final_preds]).to_csv(PRED_CSV, index=False)

    print("Test RMSE:", rmse, "MAE:", mae)
    print(f"P@{TOP_K}={precision:.4f}, R@{TOP_K}={recall:.4f}")
    print("Predicciones guardadas en", PRED_CSV)
    print("Resultados guardados en", RESULTS_CSV)

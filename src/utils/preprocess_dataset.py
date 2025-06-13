import os
import pandas as pd

DATASET_DIR = "datasets/movielens-25m"
OUTPUT_FILE = "datasets/ml25m_filtrado.csv"

MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 500


def load_data():
    ratings_path = os.path.join(DATASET_DIR, "ratings.csv")
    movies_path = os.path.join(DATASET_DIR, "movies.csv")

    ratings = pd.read_csv(ratings_path, nrows=500_000)
    movies = pd.read_csv(movies_path)
    return ratings, movies


def filter_data(ratings):
    user_counts = ratings["userId"].value_counts()
    users_to_keep = user_counts[user_counts >= MIN_USER_RATINGS].index
    ratings = ratings[ratings["userId"].isin(users_to_keep)]

    movie_counts = ratings["movieId"].value_counts()
    movies_to_keep = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index
    ratings = ratings[ratings["movieId"].isin(movies_to_keep)]

    return ratings


def preprocess():
    ratings, movies = load_data()
    ratings = filter_data(ratings)
    merged = ratings.merge(movies, on="movieId")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    preprocess()

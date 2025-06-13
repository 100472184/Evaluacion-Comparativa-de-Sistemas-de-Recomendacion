import os
import requests
import pandas as pd
from tqdm import tqdm

LINKS_PATH = "datasets/movielens-25m/links.csv"
MOVIES_PATH = "datasets/movielens-25m/movies.csv"
OUTPUT_PATH = "datasets/movielens-25m/movies_with_overview.csv"

TMDB_API_KEY = "897dc17df0393c9d755f74d07471f846"


def get_tmdb_overview(imdb_id):
    url = f"https://api.themoviedb.org/3/find/{imdb_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "external_source": "imdb_id"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("movie_results"):
            return data["movie_results"][0].get("overview", "")
    except Exception:
        return ""
    return ""


links = pd.read_csv(LINKS_PATH)
movies = pd.read_csv(MOVIES_PATH)

movies = movies.merge(links[["movieId", "imdbId"]], on="movieId", how="left")
movies["imdbId"] = movies["imdbId"].apply(
    lambda x: f"tt{int(x):07d}" if pd.notna(x) else None
)

tqdm.pandas()
movies["overview"] = movies["imdbId"].progress_apply(get_tmdb_overview)

num_with_overview = (movies["overview"].str.strip() != "").sum()
total_movies = len(movies)
print(f"Sinopsis añadidas: {num_with_overview}/{total_movies} "
      f"({(num_with_overview / total_movies) * 100:.2f}%)")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
movies.to_csv(OUTPUT_PATH, index=False)

print(f"Archivo guardado en: {OUTPUT_PATH}")

import os
import pandas as pd

BASE_DIR = "datasets"
INPUT_FILE = os.path.join(BASE_DIR, "ml25m_filtrado_big.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "user_sequence.txt")

MIN_SEQUENCE_LENGTH = 3

def main():
    ratings = pd.read_csv(INPUT_FILE)
    ratings = ratings.sort_values(["userId", "timestamp"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for uid, group in ratings.groupby("userId"):
            items = group["movieId"].tolist()
            if len(items) >= MIN_SEQUENCE_LENGTH:
                f.write(f"{uid} {' '.join(map(str, items))}\n")


if __name__ == "__main__":
    main()

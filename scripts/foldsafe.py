import zipfile
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------------------
# Load MovieLens data
# -------------------------------------------------------------------
zip_path = "/home/bpipher/Projects/GraphExplore/data/MovieLens/ml-32m.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    with z.open("ml-32m/ratings.csv") as f:
        ratings = pd.read_csv(f)
    with z.open("ml-32m/movies.csv") as f:
        movies = pd.read_csv(f)

print("ratings shape:", ratings.shape)
print("movies shape:", movies.shape)

# -------------------------------------------------------------------
# Build sparse user-movie ratings matrix
# -------------------------------------------------------------------
user_codes = ratings["userId"].astype("category")
movie_codes = ratings["movieId"].astype("category")

X_ratings = csr_matrix(
    (
        ratings["rating"].to_numpy(),
        (
            user_codes.cat.codes.to_numpy(),
            movie_codes.cat.codes.to_numpy(),
        ),
    )
)

movie_id_order = pd.Index(movie_codes.cat.categories, name="movieId")
print("Sparse ratings matrix shape:", X_ratings.shape)

# -------------------------------------------------------------------
# Comedy labels for movies in the ratings matrix
# -------------------------------------------------------------------
movies = movies.copy()
movies["is_comedy"] = (
    movies["genres"]
    .fillna("")
    .str.contains(r"(?:^|\|)Comedy(?:\||$)", regex=True)
    .astype(int)
)

movie_labels = (
    pd.DataFrame({"movieId": movie_id_order})
    .merge(movies[["movieId", "is_comedy"]], on="movieId", how="left")
)

valid_movie_mask = movie_labels["is_comedy"].notna().to_numpy()
y_all = movie_labels.loc[valid_movie_mask, "is_comedy"].to_numpy()
valid_movie_positions = np.where(valid_movie_mask)[0]

print("Movies in ratings matrix:", len(movie_id_order))
print("Movies with labels:", len(y_all))
print("Comedy prevalence:", y_all.mean())

# -------------------------------------------------------------------
# Fold-safe SVD evaluation
# -------------------------------------------------------------------
def evaluate_fold_safe_svd(
    X_ratings,
    y,
    valid_movie_positions,
    n_components,
    n_splits=5,
    random_state=42,
    C=1.0,
    class_weight="balanced",
):
    """
    Fold-safe evaluation:
      - split movies into train/test folds
      - fit SVD using only training-movie columns
      - represent held-out test movies using the training user latent factors
      - fit classifier on training movie embeddings only
      - evaluate on held-out movie embeddings
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        train_movie_cols = valid_movie_positions[train_idx]
        test_movie_cols = valid_movie_positions[test_idx]

        # Training matrix uses only training-movie columns
        X_train_ratings = X_ratings[:, train_movie_cols]

        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        svd.fit(X_train_ratings)

        # Training movie embeddings: one row per training movie
        X_train_movies = svd.components_.T

        # User factors from training matrix
        user_factors = svd.transform(X_train_ratings)   # shape: n_users x k
        singular_vals = svd.singular_values_

        # Build test movie embeddings from held-out movie columns
        # If X_train_ratings ≈ U S V^T, then for held-out movie column x_j,
        # movie coordinates in the training latent space are:
        #   v_j ≈ x_j^T U S^{-1}
        #
        # user_factors = U S, so:
        #   v_j ≈ x_j^T (U S) S^{-2}
        #       = x_j^T user_factors / S^2
        X_test_cols = X_ratings[:, test_movie_cols].T   # shape: n_test_movies x n_users
        X_test_movies = (X_test_cols @ user_factors) / (singular_vals ** 2)

        X_test_movies = np.asarray(X_test_movies)

        y_train = y[train_idx]
        y_test = y[test_idx]

        clf = LogisticRegression(
            max_iter=3000,
            C=C,
            class_weight=class_weight,
            random_state=random_state,
        )
        clf.fit(X_train_movies, y_train)

        test_proba = clf.predict_proba(X_test_movies)[:, 1]
        fold_auc = roc_auc_score(y_test, test_proba)
        auc_scores.append(fold_auc)

        print(
            f"n_components={n_components:>3} | "
            f"fold={fold} | "
            f"train_movies={len(train_idx):>6} | "
            f"test_movies={len(test_idx):>6} | "
            f"auc={fold_auc:.6f}"
        )

    return np.array(auc_scores)

# -------------------------------------------------------------------
# Sweep dimensions
# -------------------------------------------------------------------
component_grid = [8, 16, 32, 64, 128, 256]
results = []

for k in component_grid:
    auc_scores = evaluate_fold_safe_svd(
        X_ratings=X_ratings,
        y=y_all,
        valid_movie_positions=valid_movie_positions,
        n_components=k,
        n_splits=5,
        random_state=42,
        C=1.0,
        class_weight="balanced",
    )

    results.append(
        {
            "n_components": k,
            "mean_auc": auc_scores.mean(),
            "std_auc": auc_scores.std(),
        }
    )

results_df = pd.DataFrame(results).sort_values("mean_auc", ascending=False)
print("\nFold-safe results:")
print(results_df.to_string(index=False))
import zipfile
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

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
# Build movie labels
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
valid_movie_positions = np.where(valid_movie_mask)[0]
y_all = movie_labels.loc[valid_movie_mask, "is_comedy"].to_numpy()

print("Movies in ratings matrix:", len(movie_id_order))
print("Movies with labels:", len(y_all))
print("Comedy prevalence:", y_all.mean())

# -------------------------------------------------------------------
# Fold-safe evaluation function
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
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    auprc_scores = []
    auc_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        train_movie_cols = valid_movie_positions[train_idx]
        test_movie_cols = valid_movie_positions[test_idx]

        # Fit SVD on training-movie columns only
        X_train_ratings = X_ratings[:, train_movie_cols]

        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        svd.fit(X_train_ratings)

        # Training movie embeddings
        X_train_movies = svd.components_.T

        # User latent factors from the training matrix
        user_factors = svd.transform(X_train_ratings)   # shape: n_users x k
        singular_vals = svd.singular_values_

        # Project held-out movies into the training latent space
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

        fold_auprc = average_precision_score(y_test, test_proba)
        fold_auc = roc_auc_score(y_test, test_proba)

        auprc_scores.append(fold_auprc)
        auc_scores.append(fold_auc)

        print(
            f"n_components={n_components:>3} | "
            f"fold={fold} | "
            f"train_movies={len(train_idx):>6} | "
            f"test_movies={len(test_idx):>6} | "
            f"auprc={fold_auprc:.6f} | "
            f"auc={fold_auc:.6f}"
        )

    return {
        "mean_auprc": np.mean(auprc_scores),
        "std_auprc": np.std(auprc_scores),
        "mean_auc": np.mean(auc_scores),
        "std_auc": np.std(auc_scores),
    }

# -------------------------------------------------------------------
# Sweep dimensions
# -------------------------------------------------------------------
component_grid = [8, 16, 32, 64, 128, 256]
results = []

for k in component_grid:
    metrics = evaluate_fold_safe_svd(
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
            "mean_auprc": metrics["mean_auprc"],
            "std_auprc": metrics["std_auprc"],
            "mean_auc": metrics["mean_auc"],
            "std_auc": metrics["std_auc"],
        }
    )

results_df = pd.DataFrame(results).sort_values(
    ["mean_auprc", "mean_auc"],
    ascending=[False, False],
)

print("\nFold-safe results sorted by mean_auprc:")
print(results_df.to_string(index=False))
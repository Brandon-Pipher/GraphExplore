import zipfile
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

# -------------------------------------------------------------------
# Load MovieLens data from zip
# -------------------------------------------------------------------
zip_path = "/home/bpipher/Projects/GraphExplore/data/MovieLens/ml-32m.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    print("Files in zip:")
    print(z.namelist())

with zipfile.ZipFile(zip_path, "r") as z:
    with z.open("ml-32m/ratings.csv") as f:
        ratings = pd.read_csv(f)

    with z.open("ml-32m/movies.csv") as f:
        movies = pd.read_csv(f)

    with z.open("ml-32m/tags.csv") as f:
        tags = pd.read_csv(f)

    with z.open("ml-32m/links.csv") as f:
        links = pd.read_csv(f)

print("\nratings shape:", ratings.shape)
print("movies shape:", movies.shape)
print("tags shape:", tags.shape)
print("links shape:", links.shape)

# -------------------------------------------------------------------
# Build sparse user-movie rating matrix
# -------------------------------------------------------------------
# Keep category objects so we preserve the exact mapping from movieId
# to column position in the sparse matrix / SVD components.
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

print("\nSparse ratings matrix shape:", X_ratings.shape)

# -------------------------------------------------------------------
# Fit SVD on user-movie matrix
# -------------------------------------------------------------------
n_components = 16
svd = TruncatedSVD(n_components=n_components, random_state=42)
svd.fit(X_ratings)

# User embeddings would be:
# user_embeddings = svd.transform(X_ratings)

# Movie embeddings are the right singular vectors
movie_embeddings = svd.components_.T
movie_id_order = movie_codes.cat.categories

movie_embedding_df = pd.DataFrame(
    movie_embeddings,
    index=movie_id_order,
    columns=[f"svd_{i}" for i in range(n_components)],
).reset_index()

movie_embedding_df = movie_embedding_df.rename(columns={"index": "movieId"})

print("Movie embedding frame shape:", movie_embedding_df.shape)

# -------------------------------------------------------------------
# Create movie-level comedy label
# -------------------------------------------------------------------
movies = movies.copy()
movies['is_comedy'] = movies['genres'].str.contains('Comedy', na=False).astype(int)

print("\nComedy prevalence in full movies table:", movies["is_comedy"].mean())

# -------------------------------------------------------------------
# Join embeddings to movie metadata
# -------------------------------------------------------------------
movie_features = movies.merge(movie_embedding_df, on="movieId", how="inner")

feature_cols = [c for c in movie_features.columns if c.startswith("svd_")]
X_movies = movie_features[feature_cols]
y_movies = movie_features["is_comedy"]

print("Movies with learned embeddings:", len(movie_features))
print("Comedy prevalence in modeling set:", y_movies.mean())

# -------------------------------------------------------------------
# Train / evaluate logistic regression on SVD movie embedding
# -------------------------------------------------------------------
clf = LogisticRegression(max_iter=2000, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = cross_val_score(
    clf,
    X_movies,
    y_movies,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
)

acc_scores = cross_val_score(
    clf,
    X_movies,
    y_movies,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)

print("\nCross-validated ROC-AUC scores:", auc_scores)
print("Mean ROC-AUC:", auc_scores.mean())

print("\nCross-validated accuracy scores:", acc_scores)
print("Mean accuracy:", acc_scores.mean())

# -------------------------------------------------------------------
# Fit on all movies for interpretation
# -------------------------------------------------------------------
clf.fit(X_movies, y_movies)

pred_proba = clf.predict_proba(X_movies)[:, 1]
pred_label = clf.predict(X_movies)

print("\nIn-sample ROC-AUC:", roc_auc_score(y_movies, pred_proba))
print("\nClassification report:")
print(classification_report(y_movies, pred_label, digits=3))

# -------------------------------------------------------------------
# Inspect which latent dimensions matter most
# -------------------------------------------------------------------
coef_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "coef": clf.coef_[0],
        "abs_coef": np.abs(clf.coef_[0]),
    }
).sort_values("abs_coef", ascending=False)

print("\nTop SVD dimensions by absolute coefficient:")
print(coef_df.head(10))

# -------------------------------------------------------------------
# Inspect movies most / least predicted as comedy
# -------------------------------------------------------------------
movie_features = movie_features.copy()
movie_features["pred_comedy_proba"] = pred_proba

inspection_cols = ["movieId", "title", "genres", "is_comedy", "pred_comedy_proba"]

print("\nMost comedy-like movies by embedding:")
print(
    movie_features[inspection_cols]
    .sort_values("pred_comedy_proba", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\nLeast comedy-like movies by embedding:")
print(
    movie_features[inspection_cols]
    .sort_values("pred_comedy_proba", ascending=True)
    .head(15)
    .to_string(index=False)
)



import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

component_grid = [8, 16, 32, 64, 128, 256]
results = []

# Rebuild this once if not already in memory
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

movie_id_order = movie_codes.cat.categories

movies = movies.copy()
movies["is_comedy"] = (
    movies["genres"]
    .fillna("")
    .str.contains(r"(?:^|\|)Comedy(?:\||$)", regex=True)
    .astype(int)
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in component_grid:
    svd = TruncatedSVD(n_components=k, random_state=42)
    svd.fit(X_ratings)

    movie_embeddings = svd.components_.T
    movie_embedding_df = pd.DataFrame(
        movie_embeddings,
        index=movie_id_order,
        columns=[f"svd_{i}" for i in range(k)],
    ).reset_index().rename(columns={"index": "movieId"})

    movie_features = movies.merge(movie_embedding_df, on="movieId", how="inner")
    feature_cols = [c for c in movie_features.columns if c.startswith("svd_")]

    X_movies = movie_features[feature_cols]
    y_movies = movie_features["is_comedy"]

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )

    auc_scores = cross_val_score(
        clf,
        X_movies,
        y_movies,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    results.append({
        "n_components": k,
        "mean_auc": auc_scores.mean(),
        "std_auc": auc_scores.std(),
    })

results_df = pd.DataFrame(results).sort_values("mean_auc", ascending=False)
print(results_df)
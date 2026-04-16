import zipfile
import pandas as pd

zip_path = "/home/bpipher/Projects/GraphExplore/data/MovieLens/ml-32m.zip"

# List the files in the zip to verify the structure
with zipfile.ZipFile(zip_path, 'r') as z:
    print(z.namelist())

# Create dataframes from the CSV files inside the zip
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open('ml-32m/ratings.csv') as f:
        ratings = pd.read_csv(f)

    with z.open('ml-32m/movies.csv') as f:
        movies = pd.read_csv(f)

    with z.open('ml-32m/tags.csv') as f:
        tags = pd.read_csv(f)

    with z.open('ml-32m/links.csv') as f:
        links = pd.read_csv(f)


from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

svd = TruncatedSVD(n_components=16, random_state=42)
X = csr_matrix(
    (
        ratings["rating"].to_numpy(),
        (
            ratings["userId"].astype("category").cat.codes.to_numpy(),
            ratings["movieId"].astype("category").cat.codes.to_numpy(),
        ),
    )
)
svd.fit(X)
user_embeddings = svd.transform(X)
movie_embeddings = svd.components_.T
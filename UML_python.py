import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

np.random.seed(42)
base_dir = Path(__file__).resolve().parent


def resolve_input_file(explicit_path, default_name, patterns):
    search_dirs = [
        Path.cwd(),
        base_dir,
        base_dir / "data",
        Path.home() / "Desktop",
        Path.home() / "Downloads",
    ]

    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Provided file not found: {candidate}")

    for directory in search_dirs:
        candidate = directory / default_name
        if candidate.exists() and candidate.is_file():
            return candidate

    for directory in search_dirs:
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0].resolve()

    return None


parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--cooccurrence", type=str, default=None)
parser.add_argument("--products", type=str, default=None)
args, _ = parser.parse_known_args()

cooccurrence_path = resolve_input_file(
    explicit_path=args.cooccurrence,
    default_name="product_cooccurrence.csv",
    patterns=["*cooccurrence*.csv", "*co_occur*.csv"],
)
products_path = resolve_input_file(
    explicit_path=args.products,
    default_name="products.csv",
    patterns=["products*.csv", "*product*.csv"],
)

if cooccurrence_path is None or products_path is None:
    missing_files = []
    if cooccurrence_path is None:
        missing_files.append("product_cooccurrence.csv")
    if products_path is None:
        missing_files.append("products.csv")
    raise FileNotFoundError(
        "Missing required input file(s):\n- "
        + "\n- ".join(missing_files)
        + "\n\nUse arguments to pass full paths:\n"
        + "python UML_python.py --cooccurrence <path_to_cooccurrence_csv> --products <path_to_products_csv>"
    )

print(f"Using cooccurrence file: {cooccurrence_path}")
print(f"Using products file: {products_path}")

df_cooccurrence = pd.read_csv(cooccurrence_path, index_col=0)
df_products = pd.read_csv(products_path)
product_names = df_cooccurrence.index.tolist()
n_products = len(product_names)

print(f"Loaded {n_products} products")

print("\n" + "=" * 80)
print("preprocessing")
print("=" * 80)

x_norm =normalize(df_cooccurrence.values, norm="l2")

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_norm)

if HAS_UMAP:
    reducer = umap.UMAP(
        n_neighbors = 15,
        min_dist = 0.05,
        metric = "cosine",
        random_state = 42
    )
    x_reduced = np.asarray(reducer.fit_transform(x_scaled))
    print("Using UMAP for dimensionality reduction")
else:
    reducer = PCA(n_components=2, random_state=42)
    x_reduced = np.asarray(reducer.fit_transform(x_scaled))
    print("UMAP not available; using PCA fallback")

results = []

for k in range(2, 10):
    kmeans = KMeans(
        n_clusters = k,
        random_state = 42,
        n_init = 50,
        max_iter = 500,
        tol = 1e-4
    )
    labels = kmeans.fit_predict(x_reduced)

    s_values = np.asarray(silhouette_samples(x_reduced, labels))
    sil_min = np.min(s_values)
    sil_g25 = np.percentile(s_values, 25)
    db = davies_bouldin_score(x_reduced, labels)
    sil = silhouette_score(x_reduced, labels)

    unique, counts = np.unique(labels, return_counts=True)
    single_clusters = sum(counts == 1)

    results.append({
        "k": k,
        "db": db,
        "sill": sil,
        "sil_min": sil_min,
        "sil_g25": sil_g25,
        "single_clusters": single_clusters,
        "min_size": counts.min(),   
        "max_size": counts.max()
    })
    print(f"k={k}: DB={db:.3f}, sil={sil:.3f}, single_clusters={single_clusters}, min_size={counts.min()}, max_size={counts.max()}, size range={counts.min()}-{counts.max()}")
df_results = pd.DataFrame(results)

def minmax(series):
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


df_results["sil_norm"] = minmax(df_results["sill"])
df_results["db_inv_norm"] = minmax(df_results["db"].max() - df_results["db"])
df_results["sil_min_norm"] = minmax(df_results["sil_min"])
df_results["sil_g25_norm"] = minmax(df_results["sil_g25"])
df_results["size_balance"] = df_results["min_size"] / df_results["max_size"]
df_results["size_balance_norm"] = minmax(df_results["size_balance"])
df_results["single_penalty"] = 1 - minmax(df_results["single_clusters"])

df_results["score"] = (
    (0.40 * df_results["sil_norm"])
    + (0.30 * df_results["db_inv_norm"])
    + (0.10 * df_results["sil_min_norm"])
    + (0.10 * df_results["sil_g25_norm"])
    + (0.05 * df_results["size_balance_norm"])
    + (0.05 * df_results["single_penalty"])
)

top_k = (
    df_results.sort_values("score", ascending=False)
    .head(3)
    [["k", "score", "db", "sill", "single_clusters", "min_size", "max_size"]]
    .copy()
)

print("\nTop 3 k by score:")
print(top_k.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best_config = df_results.loc[df_results["score"].idxmax()]
best_k = int(best_config["k"].item())

print(f"\n{'=' * 80}\nBest k: {best_k}\n{'=' * 80}\n")
print(
    f"SELECTED: k={best_k}, DB={best_config['db'].item():.3f}, "
    f"sil={best_config['sill'].item():.3f}, "
    f"single_clusters={int(best_config['single_clusters'].item())}, "
    f"min_size={int(best_config['min_size'].item())}, "
    f"max_size={int(best_config['max_size'].item())}"
)
print("=" * 80)

kmeans = KMeans(n_clusters = best_k, random_state = 42, n_init = 50, max_iter = 500, tol = 1e-4)
labels = kmeans.fit_predict(x_reduced)

sil = silhouette_score(x_reduced, labels, metric="euclidean")
db = davies_bouldin_score(x_reduced, labels)
print(f"Final evaluation for k={best_k}: DB={db:.3f}, sil={sil:.3f}")
df_products["cluster"] = labels
cluster_counts = df_products["cluster"].value_counts().sort_index()

print("\nCluster distribution:")
print(cluster_counts)

plt.bar(cluster_counts.index, np.asarray(cluster_counts.values, dtype=np.int64))
plt.xlabel("Cluster ID")
plt.ylabel("Number of Products")
plt.title("Product Distribution Across Clusters")
plt.show()


name_col_candidates = ["product_name", "name", "product", "title"]
name_col = next((col for col in name_col_candidates if col in df_products.columns), None)

print("\nCluster details:")
for cluster_id, count in cluster_counts.items():
    print(f"\nCluster {cluster_id} ({count} products)")
    cluster_products = df_products[df_products["cluster"] == cluster_id]

    if name_col:
        for p_name in cluster_products[name_col].head(10).astype(str).tolist():
            print(f"- {p_name}")
    else:
        for row_idx in cluster_products.index[:10]:
            print(f"- Row {row_idx}")

output_path = base_dir / "products_with_clusters.csv"
df_products.to_csv(output_path, index=False)
print(f"\nSaved clustered products to: {output_path}")



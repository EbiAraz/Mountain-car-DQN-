import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
import umap
from pathlib import Path

np.random.seed(42)
script_dir = Path(__file__).resolve().parent
cooccurrence_path = script_dir / "product_cooccurrence.csv"
products_path = script_dir / "products.csv"

if cooccurrence_path.exists() and products_path.exists():
    df_cooccurrence = pd.read_csv(cooccurrence_path, index_col=0)
    df_products = pd.read_csv(products_path)
    print("Loaded data from CSV files.")
else:
    print("CSV files not found. Using synthetic demo data.")
    n_products_demo = 300
    latent_groups = np.random.choice([0, 1, 2, 3], size=n_products_demo, p=[0.25, 0.25, 0.25, 0.25])

    base_vectors = np.array(
        [
            [2.0, 0.3, 0.2, 0.1],
            [0.2, 2.0, 0.3, 0.1],
            [0.3, 0.2, 2.0, 0.2],
            [0.1, 0.3, 0.2, 2.0],
        ]
    )
    noise = np.random.normal(0, 0.25, size=(n_products_demo, 4))
    latent_features = base_vectors[latent_groups] + noise
    latent_features = np.clip(latent_features, 0, None)

    cooccurrence = latent_features @ latent_features.T
    cooccurrence += np.random.normal(0, 0.05, size=cooccurrence.shape)
    cooccurrence = np.clip(cooccurrence, 0, None)

    names = [f"product_{i:03d}" for i in range(n_products_demo)]
    df_cooccurrence = pd.DataFrame(cooccurrence, index=names, columns=names)
    df_products = pd.DataFrame({"product_name": names})

product_names = df_cooccurrence.index.tolist()
n_products = len(product_names)


print(f"loaded {n_products} products")

print("\n" + "=" * 80)
print("preprocessing")
print("=" * 80)

x_norm = normalize(df_cooccurrence.values, norm="l2")

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_norm)

reducer = umap.UMAP(
    n_neighbors = 15,
    min_dist = 0.05,
    metric = "cosine",
    random_state = 42
)
x_reduced = np.asarray(reducer.fit_transform(x_scaled))

result = []

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
    sil_mean = np.mean(s_values)
    sil_g25 = np.percentile(s_values, 25)
    db = davies_bouldin_score(x_reduced, labels)
    sil = silhouette_score(x_reduced, labels)

    unique, counts = np.unique(labels, return_counts=True)
    single_clusters = sum(counts == 1)

    result.append({
        "k": k,
        "db": db,
        "sil": sil,
        "sil_mean": sil_mean,
        "sil_q25": sil_g25,
        "single_clusters": single_clusters,
        "min_size": counts.min(),
        "max_size": counts.max()
    })
    print(
        f"k={k}: DB={db:.3f}, sil={sil:.3f}, sil_q25={sil_g25:.3f}, "
        f"single_clusters={single_clusters}, min_size={counts.min()}, max_size={counts.max()}"
    )

df_results = pd.DataFrame(result)
df_results["score"] = (
    (-df_results["db"] * 2.0)
    + (df_results["sil"] * 2.0)
    + (df_results["sil_q25"] * 1.0)
    - (df_results["single_clusters"] * 0.5)
    + (df_results["min_size"] * 0.1)
    - (df_results["max_size"] * 0.02)
)

best_config = df_results.loc[df_results["score"].idxmax()]
best_k = int(best_config["k"])

print(f"\n{'=' * 80}\nbest k: {best_k}\n{'=' * 80}\n")
print(
    f"SELECTED: k={best_k}, DB={best_config['db']:.3f}, sil={best_config['sil']:.3f}, "
    f"single_clusters={int(best_config['single_clusters'])}, "
    f"min_size={int(best_config['min_size'])}, max_size={int(best_config['max_size'])}"
)
print("=" * 80)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=50, max_iter=500, tol=1e-4)
labels = kmeans.fit_predict(x_reduced)

sil = silhouette_score(x_reduced, labels, metric="euclidean")
db = davies_bouldin_score(x_reduced, labels)
print(f"Final evaluation for k={best_k}: DB={db:.3f}, sil={sil:.3f}")
df_products["cluster"] = labels
cluster_counts = df_products["cluster"].value_counts().sort_index()

print("\nCluster distribution:")

df_products.to_csv("products_with_clusters.csv", index=False)
df_results.to_csv("k_selection_metrics.csv", index=False)

plt.figure(figsize=(8, 4.5))
plt.bar(cluster_counts.index, np.asarray(cluster_counts.values, dtype=np.int64))
plt.xlabel("cluster ID")
plt.ylabel("number of products")
plt.title("Cluster distribution")
plt.tight_layout()
plt.savefig("umap_cluster_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id}: {count} products")

print("\nSaved: products_with_clusters.csv, k_selection_metrics.csv, umap_cluster_distribution.png")


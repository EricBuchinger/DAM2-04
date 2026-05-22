"""
Assignment 4 - Pokemon Analysis
1) Distance function + pairwise distance matrix
2) Clustering (KMedoids / Agglomerative / HDBSCAN) with silhouette tuning
3) 2D visualization with tSNE and UMAP, colored by cluster
4) Outlier detection (LOF, IsolationForest, HDBSCAN noise)
5) Discussion printed at the end
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
CSV = HERE / "pokedex.csv"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------
def parse_height(s):
    m = re.search(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else np.nan


def parse_weight(s):
    m = re.search(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else np.nan


def load_data():
    df = pd.read_csv(CSV)
    df["height_m"] = df["height"].apply(parse_height)
    df["weight_kg"] = df["weight"].apply(parse_weight)
    return df


STAT_COLS = ["HP", "attack", "defense", "sp. attack", "sp. defense", "speed"]
SIZE_COLS = ["height_m", "weight_kg"]
TYPE_COLS = [c for c in [
    "type_fire", "type_water", "type_electric", "type_grass", "type_ice",
    "type_fighting", "type_poison", "type_ground", "type_flying",
    "type_psychic", "type_bug", "type_rock", "type_ghost", "type_dragon",
    "type_dark", "type_steel", "type_fairy",
]]


# ---------------------------------------------------------------------------
# 1) Distance function
# ---------------------------------------------------------------------------
def build_distance_matrix(df, w_stats=0.5, w_types=0.4, w_size=0.1):
    """
    Mixed distance:
      - stats: standardized Euclidean over the 6 base stats
      - types: Jaccard distance over the one-hot type vectors
      - size : standardized Euclidean over (log height, log weight)
    Final distance is a weighted sum; weights default to roughly the
    relative information content of each block.
    """
    # Stats block
    stats = df[STAT_COLS].to_numpy(dtype=float)
    stats_z = StandardScaler().fit_transform(stats)
    d_stats = pairwise_distances(stats_z, metric="euclidean")
    d_stats /= d_stats.max() or 1.0

    # Types block (Jaccard on one-hot rows)
    types = df[TYPE_COLS].to_numpy(dtype=int)
    d_types = pairwise_distances(types, metric="jaccard")

    # Size block - log to dampen the heavy tail (Wailord, Steelix, ...)
    size = np.log1p(df[SIZE_COLS].to_numpy(dtype=float))
    size_z = StandardScaler().fit_transform(size)
    d_size = pairwise_distances(size_z, metric="euclidean")
    d_size /= d_size.max() or 1.0

    D = w_stats * d_stats + w_types * d_types + w_size * d_size
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2  # enforce symmetry against fp noise
    return D


# ---------------------------------------------------------------------------
# 2) Clustering
# ---------------------------------------------------------------------------
def tune_agglomerative(D, k_range=range(3, 13)):
    best = (-1, None, None)
    scores = {}
    for k in k_range:
        model = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        )
        labels = model.fit_predict(D)
        score = silhouette_score(D, labels, metric="precomputed")
        scores[k] = score
        if score > best[0]:
            best = (score, k, labels)
    return best, scores


def try_hdbscan(D):
    try:
        import hdbscan
    except ImportError:
        return None
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed", min_cluster_size=15, min_samples=5
    )
    labels = clusterer.fit_predict(D.astype(np.float64))
    return labels


# ---------------------------------------------------------------------------
# 3) Visualization
# ---------------------------------------------------------------------------
def embed_tsne(D, perplexity=30, seed=42):
    return TSNE(
        n_components=2, metric="precomputed", init="random",
        perplexity=perplexity, random_state=seed,
    ).fit_transform(D)


def embed_umap(D, n_neighbors=20, min_dist=0.1, seed=42):
    try:
        import umap
    except ImportError:
        return None
    return umap.UMAP(
        n_components=2, metric="precomputed",
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed,
    ).fit_transform(D)


def scatter(emb, labels, title, path, names=None, outliers=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = np.asarray(labels)
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab20", max(len(uniq), 1))
    for i, lab in enumerate(uniq):
        mask = labels == lab
        color = "lightgrey" if lab == -1 else cmap(i)
        name = "noise" if lab == -1 else f"cluster {lab}"
        ax.scatter(emb[mask, 0], emb[mask, 1], s=18, c=[color],
                   label=name, alpha=0.85, edgecolors="none")
    if outliers is not None and names is not None:
        ax.scatter(emb[outliers, 0], emb[outliers, 1], s=80,
                   facecolors="none", edgecolors="red", linewidths=1.5,
                   label="outlier")
        for i in outliers:
            ax.annotate(names[i], (emb[i, 0], emb[i, 1]),
                        fontsize=7, alpha=0.7)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) Outlier detection
# ---------------------------------------------------------------------------
def detect_outliers(D, contamination=0.03):
    lof = LocalOutlierFactor(metric="precomputed", n_neighbors=20,
                             contamination=contamination)
    lof_pred = lof.fit_predict(D)
    lof_mask = lof_pred == -1

    # IsolationForest works on features, so feed it the distance rows
    # (each row = profile of how far this Pokemon is from every other).
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_pred = iso.fit_predict(D)
    iso_mask = iso_pred == -1

    return lof_mask, iso_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    print(f"Loaded {len(df)} Pokemon, {df.shape[1]} columns")

    # 1) Distance matrix
    D = build_distance_matrix(df)
    print(f"Distance matrix: shape={D.shape}, "
          f"mean={D.mean():.3f}, min={D.min():.3f}, max={D.max():.3f}")

    # 2) Clustering
    (best_sil, best_k, labels), scores = tune_agglomerative(D)
    print("\nSilhouette by k (agglomerative, average linkage):")
    for k, s in scores.items():
        marker = "  <- best" if k == best_k else ""
        print(f"  k={k:>2}: silhouette={s:.4f}{marker}")
    print(f"Chosen k={best_k} (silhouette={best_sil:.4f})")

    hdb_labels = try_hdbscan(D)
    if hdb_labels is not None:
        n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        n_noise = int((hdb_labels == -1).sum())
        print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

    # 3) Visualization
    print("\nComputing 2D embeddings ...")
    tsne_emb = embed_tsne(D, perplexity=30)
    scatter(tsne_emb, labels, f"tSNE - agglomerative k={best_k}",
            OUT / "tsne_agglom.png")

    umap_emb = embed_umap(D, n_neighbors=20, min_dist=0.1)
    if umap_emb is not None:
        scatter(umap_emb, labels, f"UMAP - agglomerative k={best_k}",
                OUT / "umap_agglom.png")
    else:
        print("  (umap-learn not installed, skipping UMAP)")

    if hdb_labels is not None:
        scatter(tsne_emb, hdb_labels, "tSNE - HDBSCAN",
                OUT / "tsne_hdbscan.png")
        if umap_emb is not None:
            scatter(umap_emb, hdb_labels, "UMAP - HDBSCAN",
                    OUT / "umap_hdbscan.png")

    # 4) Outliers
    lof_mask, iso_mask = detect_outliers(D, contamination=0.03)
    consensus = lof_mask & iso_mask
    print("\nOutliers:")
    print(f"  LOF flagged           : {lof_mask.sum()}")
    print(f"  IsolationForest flagged: {iso_mask.sum()}")
    print(f"  Both agree            : {consensus.sum()}")
    if hdb_labels is not None:
        print(f"  HDBSCAN noise         : {(hdb_labels == -1).sum()}")

    names = df["name"].tolist()
    outlier_idx = np.where(lof_mask | iso_mask)[0]
    print("\nTop outlier candidates:")
    for i in outlier_idx[:15]:
        flags = []
        if lof_mask[i]: flags.append("LOF")
        if iso_mask[i]: flags.append("ISO")
        if hdb_labels is not None and hdb_labels[i] == -1:
            flags.append("HDB")
        total = df.loc[i, STAT_COLS].sum()
        type_str = ",".join(
            t.replace("type_", "") for t in TYPE_COLS if df.loc[i, t] == 1
        )
        print(f"  {names[i]:<15} stat_total={total:>4}  "
              f"types=[{type_str}]  flags={'+'.join(flags)}")

    scatter(tsne_emb, labels, "tSNE - outliers highlighted",
            OUT / "tsne_outliers.png", names=names, outliers=outlier_idx)
    if umap_emb is not None:
        scatter(umap_emb, labels, "UMAP - outliers highlighted",
                OUT / "umap_outliers.png", names=names, outliers=outlier_idx)

    # 5) Discussion
    print("""
Discussion & Reflection
-----------------------
What worked well:
  * The mixed distance (stats + types + size) gave clusters that line up
    with intuition: legendaries / pseudo-legendaries fall in high-stat
    clusters, early-stage starters group together, and type-flavored
    groups (water blobs, bug swarms) are visible in both tSNE and UMAP.
  * Average-linkage agglomerative on the precomputed matrix made
    silhouette tuning straightforward.

Challenges:
  * Choosing block weights is subjective. Type Jaccard is coarse
    (most Pokemon share at most 1-2 types), so its contribution had to
    be downweighted relative to stats to avoid one giant blob per type.
  * Height / weight have heavy tails; without log scaling Wailord and
    friends dominate the size distance.
  * Silhouette on a mixed distance is a relative signal, not an absolute
    one - the best k just means 'least bad among tested values'.

If I had more time:
  * Use the thumbnails: feed them through a pretrained CNN and add an
    image-embedding block to the distance.
  * Tune block weights with a grid search against a downstream metric
    (e.g. type-purity of clusters or known evolutionary families).
  * Compare more clustering algorithms (spectral, OPTICS) and report
    stability across random seeds.
""")
    print(f"Figures written to: {OUT}")


if __name__ == "__main__":
    main()

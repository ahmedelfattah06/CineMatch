"""
K-Means Clustering Algorithm
Clusters movies based on score, weighted_score, votes (log-scaled), and runtime.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

_cached_clusters = None


def _smart_label(avg_score, avg_votes):
    if avg_score >= 8 and avg_votes > 200000:
        return "Elite Movies"
    elif avg_score >= 7 and avg_votes > 100000:
        return "Popular & Loved"
    elif avg_score >= 7:
        return "Hidden Gems"
    elif avg_score < 6:
        return "Mixed Reception"
    else:
        return "Average Movies"


def run_kmeans(df: pd.DataFrame, k: int = 5) -> list:
    global _cached_clusters
    if _cached_clusters is not None:
        return _cached_clusters
    _cached_clusters = _compute_kmeans(df, k)
    return _cached_clusters


def run_kmeans_custom(df: pd.DataFrame, k: int = 5) -> list:
    return _compute_kmeans(df, k)


def _compute_kmeans(df: pd.DataFrame, k: int) -> list:
    features_df = df[['movie_id', 'name', 'genre', 'score', 'weighted_score',
                       'votes', 'runtime', 'classification']].copy()
    features_df = features_df.dropna(subset=['score', 'weighted_score', 'votes', 'runtime'])
    features_df['votes_log'] = np.log1p(features_df['votes'])

    X = features_df[['score', 'weighted_score', 'votes_log', 'runtime']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    features_df = features_df.copy()
    features_df['cluster'] = kmeans.fit_predict(X_scaled)

    colors = ['#B8752A', '#245A40', '#1C2B3A', '#7A3020', '#5C4B7A',
              '#3A5A6B', '#6B5C1A', '#6B3060', '#1A4A6B', '#5C1A1A']

    clusters = []
    for cluster_id in range(k):
        cluster_movies = features_df[features_df['cluster'] == cluster_id]
        if cluster_movies.empty:
            continue

        dominant_genre = cluster_movies['genre'].mode()[0]
        avg_score = round(float(cluster_movies['score'].mean()), 2)
        avg_votes = float(cluster_movies['votes'].mean())
        avg_runtime = round(float(cluster_movies['runtime'].mean()), 0)
        top_genres = cluster_movies['genre'].value_counts().head(3).index.tolist()

        top_movies = cluster_movies.nlargest(3, 'score')[
            ['movie_id', 'name', 'genre', 'score', 'classification']
        ].to_dict('records')
        for m in top_movies:
            m['movie_id'] = int(m['movie_id'])
            m['score'] = round(float(m['score']), 1)

        sample = cluster_movies.sample(min(50, len(cluster_movies)), random_state=42)
        scatter_data = [
            {'x': round(float(r['score']), 1), 'y': int(r['runtime']), 'name': r['name']}
            for _, r in sample.iterrows()
        ]

        # ── all_movie_ids: the key fix ──────────────────────────────
        all_ids = [int(x) for x in cluster_movies['movie_id'].tolist()]

        clusters.append({
            'id':           cluster_id,
            'label':        f"{dominant_genre} Cluster",
            'smart_label':  _smart_label(avg_score, avg_votes),
            'dominant_genre': dominant_genre,
            'top_genres':   top_genres,
            'count':        int(len(cluster_movies)),
            'avg_score':    avg_score,
            'avg_runtime':  int(avg_runtime),
            'top_movies':   top_movies,
            'scatter_data': scatter_data,
            'all_movie_ids': all_ids,
            'color':        colors[cluster_id % len(colors)],
            'color_index':  cluster_id,
        })

    return clusters
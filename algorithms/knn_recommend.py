"""
KNN Collaborative-Style Recommender
Uses NearestNeighbors on movie features (genre encoded, score, runtime, year).
Given a list of liked movie_ids, finds nearest neighbors and returns recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder


class KNNRecommender:
    def __init__(self, df: pd.DataFrame):
        """
        Fit the KNN model at startup on genre-encoded movie features.
        """
        self.df = df.reset_index(drop=True)
        self.id_to_idx = {row['movie_id']: i for i, row in self.df.iterrows()}

        # Select and encode features
        features_df = self.df[['movie_id', 'genre', 'score', 'runtime', 'year']].copy()
        features_df = features_df.fillna({'score': 0.0, 'runtime': 90, 'year': 2000, 'genre': 'Drama'})

        # Encode genre as numeric
        le = LabelEncoder()
        features_df['genre_encoded'] = le.fit_transform(features_df['genre'])

        X = features_df[['genre_encoded', 'score', 'runtime', 'year']].values
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)

        # Fit KNN model
        self.knn = NearestNeighbors(n_neighbors=15, metric='euclidean', algorithm='ball_tree')
        self.knn.fit(self.X_scaled)

    def get_recommendations(self, movie_ids: list, top_n: int = 10) -> list:
        """
        Given a list of liked movie_ids, find KNN neighbors for each liked movie,
        aggregate results, and return top_n recommendations excluding liked movies.
        """
        liked_indices = set()
        neighbor_counts = {}

        for mid in movie_ids:
            if mid not in self.id_to_idx:
                continue
            idx = self.id_to_idx[mid]
            liked_indices.add(idx)
            # Get neighbors for this liked movie
            distances, indices = self.knn.kneighbors([self.X_scaled[idx]])
            for dist, neighbor_idx in zip(distances[0], indices[0]):
                if neighbor_idx not in liked_indices:
                    if neighbor_idx not in neighbor_counts:
                        neighbor_counts[neighbor_idx] = 0
                    neighbor_counts[neighbor_idx] += 1.0 / (1 + dist)

        # Remove liked movies from candidates
        for idx in liked_indices:
            neighbor_counts.pop(idx, None)

        # Sort by score descending
        sorted_candidates = sorted(neighbor_counts.items(), key=lambda x: x[1], reverse=True)[:top_n * 2]

        results = []
        for idx, score in sorted_candidates:
            row = self.df.iloc[idx]
            results.append({
                'movie_id': int(row['movie_id']),
                'name': row['name'],
                'genre': row['genre'],
                'score': float(row['score']) if pd.notna(row['score']) else 0.0,
                'year': int(row['year']) if pd.notna(row['year']) else 0,
                'star': row['star'] if pd.notna(row['star']) else '',
                'director': row['director'] if pd.notna(row['director']) else '',
                'classification': row['classification'],
                'reason': f"Similar genre: {row['genre']}, Score: {row['score']}"
            })
            if len(results) >= top_n:
                break

        return results

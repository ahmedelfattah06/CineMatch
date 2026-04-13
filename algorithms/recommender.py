"""
TF-IDF + Cosine Similarity Recommender
Builds a content string per movie and computes pairwise similarity.
The similarity matrix is cached at startup so requests are fast.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


class ContentRecommender:
    def __init__(self, df: pd.DataFrame):
        """
        Build and cache the TF-IDF similarity matrix at startup.
        Content string combines genre, director, star, writer, country, classification.
        """
        self.df = df.reset_index(drop=True)
        # Create a mapping from movie_id to dataframe index
        self.id_to_idx = {row['movie_id']: i for i, row in self.df.iterrows()}
        # Build content string for each movie
        self.df['content'] = (
            self.df['genre'].fillna('') + ' ' +
            self.df['director'].fillna('') + ' ' +
            self.df['star'].fillna('') + ' ' +
            self.df['writer'].fillna('') + ' ' +
            self.df['country'].fillna('') + ' ' +
            self.df['classification'].fillna('')
        )
        # Fit TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df['content'])
        # Compute cosine similarity matrix
        self.sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def get_similar(self, movie_id: int, top_n: int = 6) -> list:
        """
        Given a movie_id, return the top_n most similar movies as a list of dicts.
        """
        if movie_id not in self.id_to_idx:
            return []
        idx = self.id_to_idx[movie_id]
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        # Sort by similarity descending, skip the movie itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
        results = []
        for i, score in sim_scores:
            row = self.df.iloc[i]
            results.append({
                'movie_id': int(row['movie_id']),
                'name': row['name'],
                'genre': row['genre'],
                'score': float(row['score']) if pd.notna(row['score']) else 0.0,
                'year': int(row['year']) if pd.notna(row['year']) else 0,
                'star': row['star'] if pd.notna(row['star']) else '',
                'classification': row['classification'],
                'similarity': round(float(score), 3)
            })
        return results

    def get_recommendations_for_multiple(self, movie_ids: list, top_n: int = 10) -> list:
        """
        Given a list of liked movie_ids, aggregate similarity scores and return top_n recommendations.
        Excludes the liked movies from results.
        """
        liked_indices = set()
        agg_scores = np.zeros(len(self.df))
        for mid in movie_ids:
            if mid in self.id_to_idx:
                idx = self.id_to_idx[mid]
                liked_indices.add(idx)
                agg_scores += self.sim_matrix[idx]
        # Zero out liked movies
        for idx in liked_indices:
            agg_scores[idx] = 0
        top_indices = np.argsort(agg_scores)[::-1][:top_n * 2]
        results = []
        for idx in top_indices:
            if idx in liked_indices:
                continue
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

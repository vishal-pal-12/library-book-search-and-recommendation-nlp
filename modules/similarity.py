"""
Experiment 9: Text Similarity & Book Recommendation System
Converts book catalog metadata (Title, Genre, Description) and user search queries
into TF-IDF vector space, computes Cosine Similarity, and returns ranked book recommendations.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modules.preprocessing import preprocess_text
from modules.morphology import expand_query_morphologically

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "books.csv")

class BookRecommender:
    def __init__(self, data_path=None):
        self.data_path = data_path or DATA_PATH
        self.df_books = None
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=2500
        )
        self.book_tfidf_matrix = None
        self.load_and_index_books()

    def load_and_index_books(self):
        """Loads books.csv and builds TF-IDF representation."""
        if os.path.exists(self.data_path):
            self.df_books = pd.read_csv(self.data_path)
        else:
            self.df_books = pd.DataFrame(columns=["book_id", "title", "author", "genre", "published_year", "rating", "description"])

        # Create rich composite representation for each book
        self.df_books["composite_content"] = (
            self.df_books["title"].fillna("") + " " +
            self.df_books["author"].fillna("") + " " +
            self.df_books["genre"].fillna("") + " " +
            self.df_books["description"].fillna("")
        )

        # Preprocess each book text
        processed_docs = []
        for text in self.df_books["composite_content"]:
            prep = preprocess_text(str(text))
            lemmas = " ".join(prep["lemmatized_tokens"])
            processed_docs.append(lemmas)

        self.df_books["processed_content"] = processed_docs
        self.book_tfidf_matrix = self.vectorizer.fit_transform(processed_docs)

    def recommend_books(self, query, top_k=5, apply_morphological_expansion=True):
        """
        Ranks library books based on cosine similarity with the user query.
        Optionally uses morphological expansion from Exp 4 to maximize recall.
        """
        if self.df_books is None or self.df_books.empty:
            self.load_and_index_books()

        # Preprocess user query
        prep = preprocess_text(query)
        query_lemmas = prep["lemmatized_tokens"]

        if apply_morphological_expansion and query_lemmas:
            expansion = expand_query_morphologically(query_lemmas)
            enhanced_query = " ".join(expansion["expanded_tokens"])
        else:
            enhanced_query = " ".join(query_lemmas) if query_lemmas else query

        # Transform query into TF-IDF vector
        query_vec = self.vectorizer.transform([enhanced_query])

        # Compute cosine similarity
        cosine_scores = cosine_similarity(query_vec, self.book_tfidf_matrix).flatten()

        # Rank indices in descending order
        ranked_indices = np.argsort(cosine_scores)[::-1]

        recommendations = []
        for idx in ranked_indices[:top_k]:
            score = float(cosine_scores[idx])
            book = self.df_books.iloc[idx]
            
            recommendations.append({
                "book_id": int(book["book_id"]),
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "published_year": int(book["published_year"]),
                "rating": float(book["rating"]),
                "description": book["description"],
                "similarity_score": round(score, 4),
                "match_percentage": f"{round(score * 100, 1)}%"
            })

        return {
            "query": query,
            "processed_query": enhanced_query,
            "recommendations": recommendations,
            "top_match": recommendations[0] if recommendations else None
        }

    def find_similar_to_book(self, title_or_id, top_k=5):
        """Recommends books similar to a given target book (Content-Based Filtering)."""
        if isinstance(title_or_id, int) or str(title_or_id).isdigit():
            match = self.df_books[self.df_books["book_id"] == int(title_or_id)]
        else:
            match = self.df_books[self.df_books["title"].str.lower().str.contains(str(title_or_id).lower())]

        if match.empty:
            return {"error": f"Book '{title_or_id}' not found in library catalog."}

        target_idx = match.index[0]
        target_vec = self.book_tfidf_matrix[target_idx]

        sim_scores = cosine_similarity(target_vec, self.book_tfidf_matrix).flatten()
        ranked_indices = np.argsort(sim_scores)[::-1]

        recommendations = []
        for idx in ranked_indices:
            if idx == target_idx:
                continue  # Skip self
            score = float(sim_scores[idx])
            book = self.df_books.iloc[idx]
            recommendations.append({
                "book_id": int(book["book_id"]),
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "published_year": int(book["published_year"]),
                "rating": float(book["rating"]),
                "description": book["description"],
                "similarity_score": round(score, 4),
                "match_percentage": f"{round(score * 100, 1)}%"
            })
            if len(recommendations) >= top_k:
                break

        return {
            "target_book": self.df_books.iloc[target_idx]["title"],
            "target_author": self.df_books.iloc[target_idx]["author"],
            "recommendations": recommendations
        }

# Singleton instance
book_recommender = BookRecommender()

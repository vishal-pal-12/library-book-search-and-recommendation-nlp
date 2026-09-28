"""
Experiment 5: N-Gram Language Model
Implements Unigram, Bigram, and Trigram language models with <s> and </s> sentence boundary markers,
frequency counting, probability estimations, and library search query suggestion/autocomplete.
"""

from collections import Counter, defaultdict
import os
import pandas as pd
from nltk.tokenize import word_tokenize

class NGramLanguageModel:
    def __init__(self):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.bigram_context = defaultdict(Counter)
        self.trigram_context = defaultdict(Counter)
        self.total_unigrams = 0
        self.is_fitted = False

    def build_ngrams(self, tokens, n, add_boundaries=True):
        """
        Builds n-grams from a list of tokens.
        If add_boundaries is True, pads sequence with <s> and </s>.
        """
        if add_boundaries:
            if n > 1:
                padding_start = ["<s>"] * (n - 1)
                padded_tokens = padding_start + tokens + ["</s>"]
            else:
                padded_tokens = ["<s>"] + tokens + ["</s>"]
        else:
            padded_tokens = tokens

        ngrams = []
        for i in range(len(padded_tokens) - n + 1):
            ngram = tuple(padded_tokens[i:i + n])
            ngrams.append(ngram)
        return ngrams

    def fit_from_library_corpus(self, corpus=None):
        """
        Fits n-gram models on the combined library titles, genres, descriptions, and reviews.
        """
        if corpus is None:
            # Load default library books and reviews
            base_dir = os.path.dirname(os.path.dirname(__file__))
            books_path = os.path.join(base_dir, "data", "books.csv")
            reviews_path = os.path.join(base_dir, "data", "library_reviews.csv")

            sentences = []
            if os.path.exists(books_path):
                df_books = pd.read_csv(books_path)
                sentences.extend(df_books["title"].dropna().tolist())
                sentences.extend(df_books["genre"].dropna().tolist())
                sentences.extend(df_books["description"].dropna().tolist())
            
            if os.path.exists(reviews_path):
                df_rev = pd.read_csv(reviews_path)
                sentences.extend(df_rev["review"].dropna().tolist())
            
            # Common library search query patterns
            seed_queries = [
                "find science fiction books for beginners",
                "suggest fantasy adventure books",
                "show books written by J.K. Rowling",
                "books about artificial intelligence and machine learning",
                "historical books about modern Indian democracy",
                "programming guides for python beginners",
                "detective novels and murder mysteries in english",
                "biography of famous scientists and inventors",
                "best books on software engineering and clean code",
                "borrow books from library front desk",
                "search library catalogue for classic literature"
            ]
            sentences.extend(seed_queries)
            corpus = sentences

        self.unigrams.clear()
        self.bigrams.clear()
        self.trigrams.clear()
        self.bigram_context.clear()
        self.trigram_context.clear()

        for sent in corpus:
            tokens = [t.lower() for t in word_tokenize(str(sent)) if t.isalnum()]
            if not tokens:
                continue

            # Unigrams
            u_list = self.build_ngrams(tokens, 1, add_boundaries=True)
            self.unigrams.update(u_list)

            # Bigrams
            b_list = self.build_ngrams(tokens, 2, add_boundaries=True)
            self.bigrams.update(b_list)
            for w1, w2 in b_list:
                self.bigram_context[w1][w2] += 1

            # Trigrams
            t_list = self.build_ngrams(tokens, 3, add_boundaries=True)
            self.trigrams.update(t_list)
            for w1, w2, w3 in t_list:
                self.trigram_context[(w1, w2)][w3] += 1

        self.total_unigrams = sum(self.unigrams.values())
        self.is_fitted = True

    def analyze_query(self, query):
        """
        Analyzes a single user search query into its unigrams, bigrams, and trigrams.
        """
        tokens = [t.lower() for t in word_tokenize(query) if t.isalnum()]
        
        # Build ngrams with boundaries
        unigrams_bounded = self.build_ngrams(tokens, 1, add_boundaries=True)
        bigrams_bounded = self.build_ngrams(tokens, 2, add_boundaries=True)
        trigrams_bounded = self.build_ngrams(tokens, 3, add_boundaries=True)

        # Build raw word sequences without boundary markers for display
        unigrams_raw = self.build_ngrams(tokens, 1, add_boundaries=False)
        bigrams_raw = self.build_ngrams(tokens, 2, add_boundaries=False)
        trigrams_raw = self.build_ngrams(tokens, 3, add_boundaries=False)

        if not self.is_fitted:
            self.fit_from_library_corpus()

        # Get frequencies from corpus model
        unigram_counts = [{"ngram": " ".join(ug), "count": self.unigrams[ug]} for ug in unigrams_bounded]
        bigram_counts = [{"ngram": " ".join(bg), "count": self.bigrams[bg]} for bg in bigrams_bounded]
        trigram_counts = [{"ngram": " ".join(tg), "count": self.trigrams[tg]} for tg in trigrams_bounded]

        return {
            "query": query,
            "tokens": tokens,
            "unigrams_raw": [" ".join(u) for u in unigrams_raw],
            "bigrams_raw": [" ".join(b) for b in bigrams_raw],
            "trigrams_raw": [" ".join(t) for t in trigrams_raw],
            "unigram_counts": unigram_counts,
            "bigram_counts": bigram_counts,
            "trigram_counts": trigram_counts
        }

    def suggest_next_word(self, context_tokens, top_k=5):
        """
        Predicts next words given previous context tokens using bigram & trigram statistics.
        """
        if not self.is_fitted:
            self.fit_from_library_corpus()

        clean_context = [t.lower() for t in context_tokens if t.isalnum()]
        suggestions = []

        # Try trigram match first if at least 2 tokens
        if len(clean_context) >= 2:
            key = (clean_context[-2], clean_context[-1])
            if key in self.trigram_context:
                top_tri = self.trigram_context[key].most_common(top_k)
                for word, cnt in top_tri:
                    if word not in ["</s>", "<s>"]:
                        suggestions.append({"word": word, "count": cnt, "type": "Trigram Model"})

        # Backoff to bigram match if needed
        if len(suggestions) < top_k and len(clean_context) >= 1:
            key = clean_context[-1]
            if key in self.bigram_context:
                top_bi = self.bigram_context[key].most_common(top_k)
                for word, cnt in top_bi:
                    if word not in ["</s>", "<s>"] and not any(s["word"] == word for s in suggestions):
                        suggestions.append({"word": word, "count": cnt, "type": "Bigram Model"})

        return suggestions[:top_k]

# Singleton instance
ngram_model = NGramLanguageModel()

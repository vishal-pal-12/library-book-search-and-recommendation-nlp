"""
Interactive Book Predictor for VS Code
Library Book Search & Recommendation System Using NLP
College Mini Project — Integrates Experiments 1 through 10
"""

import os
import sys

# Ensure UTF-8 output encoding in Windows Terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.similarity import book_recommender
from modules.preprocessing import preprocess_text
from modules.morphology import analyze_morphology
from modules.pos_tagging import tag_pos
from modules.chunking import extract_chunks
from modules.ner import extract_named_entities


def predict_by_book_name(book_title: str, top_k: int = 5):
    """
    Finds a target book in the catalog, runs NLP linguistic analysis on its synopsis
    across experiments (Preprocessing, POS Tagging, Chunking, NER), and predicts
    the top similar books using TF-IDF Cosine Similarity (Exp 9).
    """
    clean_query = book_title.strip()
    if not clean_query:
        print("Please enter a valid book title.")
        return None

    # Search in dataset (books.csv)
    df_books = book_recommender.df_books
    match = df_books[df_books["title"].str.lower().str.contains(clean_query.lower())]

    if match.empty:
        print("\n" + "=" * 78)
        print(f"X Book '{clean_query}' not found in library catalog.")
        print("=" * 78)
        print("Hint: Try searching with any keyword or title from dataset, for example:")
        print("   - Harry Potter        - The Hobbit          - Dune")
        print("   - Clean Code          - Python Crash Course - 1984")
        print("   - Sapiens             - The Da Vinci Code   - Cosmos")
        return None

    target_row = match.iloc[0]
    target_title = target_row["title"]
    target_author = target_row["author"]
    target_genre = target_row["genre"]
    target_year = target_row["published_year"]
    target_rating = target_row["rating"]
    target_desc = target_row["description"]

    # Compute recommendations via Exp 9 (similarity module)
    res = book_recommender.find_similar_to_book(target_title, top_k=top_k)

    print("\n" + "=" * 78)
    print("📚 TARGET BOOK IDENTIFIED IN LIBRARY CATALOG (data/books.csv)")
    print("=" * 78)
    print(f"Title:       {target_title}")
    print(f"Author:      {target_author}")
    print(f"Genre:       {target_genre}")
    print(f"Published:   {target_year} | Rating: ⭐ {target_rating}/5")
    print(f"Synopsis:    {target_desc}")

    # =========================================================================
    # NLP PIPELINE EXECUTION FOR THIS SELECTED BOOK (EXPERIMENTS 2, 3, 6, 7, 8)
    # =========================================================================
    print("\n" + "-" * 78)
    print("🧠 CONNECTED NLP PIPELINE STAGES EXECUTED ON THIS BOOK")
    print("-" * 78)

    # 1. Exp 2 & 3: Preprocessing, Stop Words & Lemmatization
    prep_res = preprocess_text(target_desc)
    key_lemmas = prep_res["lemmatized_tokens"][:8]
    print(f"• [Exp 2 & 3 - Preprocessing]: Lemmatized Tokens:")
    print(f"   -> {', '.join(key_lemmas)} ...")

    # 2. Exp 6: POS Tagging
    pos_res = tag_pos(target_desc)
    sample_tags = [f"{w}/{t}" for w, t in pos_res["tagged_tuples"][:6]]
    print(f"• [Exp 6 - POS Tagging]: Grammatical Part-of-Speech Tags:")
    print(f"   -> {' '.join(sample_tags)} ...")

    # 3. Exp 7: Chunking (Noun Phrase Extraction)
    chunk_res = extract_chunks(pos_res["tagged_tuples"])
    if chunk_res.get("extracted_phrases"):
        phrases = [
            p["phrase"] if isinstance(p, dict) else str(p)
            for p in chunk_res["extracted_phrases"][:4]
        ]
        print(f"• [Exp 7 - Phrase Chunking]: Extracted Key Noun Phrases:")
        print(f"   -> {', '.join(phrases)}")

    # 4. Exp 8: Named Entity Recognition (NER)
    ner_res = extract_named_entities(target_desc)
    if ner_res.get("entities"):
        ents = [
            f"{e['entity']} ({e['label']})"
            for e in ner_res["entities"][:5]
        ]
        print(f"• [Exp 8 - Named Entity Recognition]: Catalog Entities Identified:")
        print(f"   -> {', '.join(ents)}")

    # =========================================================================
    # PREDICTION RESULTS VIA EXPERIMENT 9 (TF-IDF COSINE SIMILARITY)
    # =========================================================================
    recs = res.get("recommendations", [])
    print("\n" + "=" * 78)
    print(f"🎯 [Exp 9 - Text Similarity]: TOP {len(recs)} PREDICTED SIMILAR BOOKS")
    print("=" * 78)

    header = f"{'#':<3} | {'Book Title':<32} | {'Author':<20} | {'Genre':<14} | {'Match %':<8}"
    print(header)
    print("-" * len(header))

    for idx, b in enumerate(recs, 1):
        title = (b["title"][:29] + "...") if len(b["title"]) > 32 else b["title"]
        author = (b["author"][:17] + "...") if len(b["author"]) > 20 else b["author"]
        print(f"{idx:<3} | {title:<32} | {author:<20} | {b['genre']:<14} | {b['match_percentage']:<8}")

    print("=" * 78)
    print("\nDetailed Descriptions & Similarity Breakdown:")
    for idx, b in enumerate(recs, 1):
        print(f"\n{idx}. {b['title']} (Cosine Match: {b['match_percentage']})")
        print(f"   Author: {b['author']} | Genre: {b['genre']} | Rating: ⭐ {b['rating']}/5")
        print(f"   Synopsis: {b['description']}")

    print("\n" + "=" * 78)
    return res


SAMPLE_BOOKS = [
    "Harry Potter and the Sorcerer's Stone",
    "The Hobbit",
    "Dune",
    "Foundation",
    "Clean Code: A Handbook of Agile Software Craftsmanship",
    "Python Crash Course",
    "1984",
    "Sapiens: A Brief History of Humankind",
    "The Da Vinci Code",
    "Cosmos"
]


def interactive_cli():
    """Interactive loop for entering multiple book queries inside VS Code."""
    print("=" * 78)
    print("📚 LIBRARY BOOK SEARCH & RECOMMENDATION SYSTEM — PREDICTOR (VS CODE)")
    print("=" * 78)
    print("Dataset contains 40 curated books across 10 diverse genres.")
    print("\nSample Book Titles you can test from catalog:")
    for i, t in enumerate(SAMPLE_BOOKS, 1):
        print(f"  [{i:2d}] {t}")

    print("-" * 78)
    print("Type a book name (e.g. 'Harry Potter', 'Dune', '1984'), a number [1-10],")
    print("or type 'exit' to quit.")
    print("-" * 78)

    while True:
        try:
            user_choice = input("\nEnter Book Name (or 1-10, Enter for Harry Potter): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Thank you!")
            break

        if not user_choice:
            user_choice = "Harry Potter"
            print(f"No title typed. Defaulting to: '{user_choice}'")

        if user_choice.lower() in ["exit", "quit", "q"]:
            print("Exiting predictor. Goodbye!")
            break

        if user_choice.isdigit() and 1 <= int(user_choice) <= len(SAMPLE_BOOKS):
            user_choice = SAMPLE_BOOKS[int(user_choice) - 1]

        predict_by_book_name(user_choice)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_title = " ".join(sys.argv[1:])
        predict_by_book_name(query_title)
    else:
        interactive_cli()

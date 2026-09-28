"""
Experiment 8: Named Entity Recognition (NER)
Extracts library entities: BOOK, AUTHOR, PUBLISHER, LOCATION, ORGANIZATION, DATE
using a hybrid approach: NLTK ne_chunk + Domain Library Gazetteer & Rule-based Matcher.
"""

import re
import os
import pandas as pd
import nltk
from nltk import ne_chunk, pos_tag
from nltk.tokenize import word_tokenize
from nltk.tree import Tree

# Ensure NLTK chunker models are present
for r in ["maxent_ne_chunker", "maxent_ne_chunker_tab", "words"]:
    try:
        nltk.data.find(f"chunkers/{r}" if "chunker" in r else f"corpora/{r}")
    except LookupError:
        nltk.download(r, quiet=True)

# Domain Knowledge Base / Library Gazetteer
KNOWN_PUBLISHERS = [
    "penguin", "o'reilly", "harpercollins", "bloomsbury", "oxford university press",
    "cambridge university press", "mit press", "routledge", "pearson", "mcgraw hill",
    "simon & schuster", "vintage", "scholastic", "packt", "wiley"
]

KNOWN_LOCATIONS = [
    "pune", "mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata",
    "london", "boston", "new york", "oxford", "cambridge", "paris", "tokyo", "san francisco"
]

def load_catalog_gazetteer():
    """Loads known authors and book titles from books.csv dataset."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    books_path = os.path.join(base_dir, "data", "books.csv")
    
    known_books = {}
    known_authors = {}
    
    if os.path.exists(books_path):
        df = pd.read_csv(books_path)
        for _, row in df.iterrows():
            title = str(row["title"]).strip()
            author = str(row["author"]).strip()
            known_books[title.lower()] = title
            known_authors[author.lower()] = author

    # Common variations and famous names
    known_authors["j.k. rowling"] = "J.K. Rowling"
    known_authors["jk rowling"] = "J.K. Rowling"
    known_authors["tolkien"] = "J.R.R. Tolkien"
    known_authors["j.r.r. tolkien"] = "J.R.R. Tolkien"
    known_authors["asimov"] = "Isaac Asimov"
    known_authors["abdul kalam"] = "A.P.J. Abdul Kalam"
    known_authors["dr. kalam"] = "A.P.J. Abdul Kalam"
    known_authors["george orwell"] = "George Orwell"
    known_authors["agatha christie"] = "Agatha Christie"

    known_books["harry potter"] = "Harry Potter"
    known_books["the hobbit"] = "The Hobbit"
    known_books["dune"] = "Dune"
    known_books["clean code"] = "Clean Code"
    known_books["sapiens"] = "Sapiens"
    known_books["wings of fire"] = "Wings of Fire"
    known_books["1984"] = "1984"

    return known_books, known_authors

KNOWN_BOOKS, KNOWN_AUTHORS = load_catalog_gazetteer()

def extract_named_entities(text):
    """
    Extracts entities using both library domain rules and NLTK ne_chunk.
    """
    entities = []
    text_lower = text.lower()
    matched_spans = []

    # 1. Match Known Book Titles
    for key, formal_title in sorted(KNOWN_BOOKS.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r"\b" + re.escape(key) + r"\b"
        for m in re.finditer(pattern, text_lower):
            start, end = m.span()
            if not any(s <= start < e or s < end <= e for s, e in matched_spans):
                matched_spans.append((start, end))
                entities.append({
                    "entity": text[start:end],
                    "canonical_name": formal_title,
                    "label": "BOOK",
                    "source": "Library Catalog Matcher",
                    "confidence": 0.95
                })

    # 2. Match Known Authors
    for key, formal_author in sorted(KNOWN_AUTHORS.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r"\b" + re.escape(key) + r"\b"
        for m in re.finditer(pattern, text_lower):
            start, end = m.span()
            if not any(s <= start < e or s < end <= e for s, e in matched_spans):
                matched_spans.append((start, end))
                entities.append({
                    "entity": text[start:end],
                    "canonical_name": formal_author,
                    "label": "AUTHOR",
                    "source": "Author Gazetteer",
                    "confidence": 0.95
                })

    # 3. Match Publishers
    for pub in KNOWN_PUBLISHERS:
        pattern = r"\b" + re.escape(pub) + r"\b"
        for m in re.finditer(pattern, text_lower):
            start, end = m.span()
            if not any(s <= start < e or s < end <= e for s, e in matched_spans):
                matched_spans.append((start, end))
                entities.append({
                    "entity": text[start:end],
                    "canonical_name": pub.title(),
                    "label": "PUBLISHER",
                    "source": "Publisher Gazetteer",
                    "confidence": 0.90
                })

    # 4. Match Locations
    for loc in KNOWN_LOCATIONS:
        pattern = r"\b" + re.escape(loc) + r"\b"
        for m in re.finditer(pattern, text_lower):
            start, end = m.span()
            if not any(s <= start < e or s < end <= e for s, e in matched_spans):
                matched_spans.append((start, end))
                entities.append({
                    "entity": text[start:end],
                    "canonical_name": loc.title(),
                    "label": "LOCATION",
                    "source": "City/Region Gazetteer",
                    "confidence": 0.92
                })

    # 5. Match Dates / Years (e.g. 1997, 2024, 19th century)
    date_patterns = [
        r"\b(18\d{2}|19\d{2}|20\d{2})\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:century|decade)\b"
    ]
    for pat in date_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            start, end = m.span()
            if not any(s <= start < e or s < end <= e for s, e in matched_spans):
                matched_spans.append((start, end))
                entities.append({
                    "entity": text[start:end],
                    "canonical_name": text[start:end],
                    "label": "DATE",
                    "source": "Temporal Regex Matcher",
                    "confidence": 0.88
                })

    # 6. Fallback / Augment with NLTK ne_chunk
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    chunked = ne_chunk(tagged)

    for subtree in chunked:
        if isinstance(subtree, Tree):
            ent_text = " ".join([leaf[0] for leaf in subtree.leaves()])
            ent_type = subtree.label()
            
            # Map NLTK labels to our library schema
            if ent_type in ["GPE", "GEO"]:
                mapped_label = "LOCATION"
            elif ent_type == "PERSON":
                mapped_label = "AUTHOR"
            elif ent_type == "ORGANIZATION":
                mapped_label = "ORGANIZATION"
            else:
                mapped_label = ent_type

            # Check if already captured by gazetteer
            if not any(ent_text.lower() in e["entity"].lower() for e in entities):
                entities.append({
                    "entity": ent_text,
                    "canonical_name": ent_text,
                    "label": mapped_label,
                    "source": f"NLTK ne_chunk ({ent_type})",
                    "confidence": 0.80
                })

    return {
        "text": text,
        "entities": entities,
        "entity_count": len(entities)
    }

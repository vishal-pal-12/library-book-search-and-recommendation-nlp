# COLLEGE NLP MINI PROJECT REPORT

---

# 📚 LIBRARY BOOK SEARCH & RECOMMENDATION SYSTEM USING NATURAL LANGUAGE PROCESSING

**Course:** Natural Language Processing (NLP) Lab  
**Domain:** Library Information Retrieval, User Reviews & Semantic Book Recommendation  
**Academic Year:** 2025–2026  
**Implementation:** Python, NLTK, Scikit-learn, TensorFlow/Keras, Streamlit  

---

## TABLE OF CONTENTS
1. Project Title
2. Abstract
3. Introduction
4. Problem Statement
5. Objectives
6. Scope
7. Software Requirements
8. Hardware Requirements
9. System Architecture
10. Dataset Description
11. Complete Integrated NLP Pipeline
12. Experiment 1 — Sentiment Analysis of Book Reviews
13. Experiment 2 — Tokenization, Filtration & Script Validation
14. Experiment 3 — Stop Word Removal, Stemming & Lemmatization
15. Experiment 4 — Morphological Analysis & Word Generation
16. Experiment 5 — N-Gram Language Model & Query Suggestions
17. Experiment 6 — Part-of-Speech (POS) Tagging
18. Experiment 7 — Chunking, Feature Selection & Training Size Analysis
19. Experiment 8 — Named Entity Recognition (NER)
20. Experiment 9 — Text Similarity & Book Recommendation Engine
21. Experiment 10 — Word Sense Disambiguation using LSTM/GRU
22. Results and Performance Metrics
23. Screenshots & System Demonstration Layout
24. Advantages of the Proposed System
25. Limitations
26. Real-World Applications
27. Future Scope
28. Conclusion

---

## 1. PROJECT TITLE
**Library Book Search & Recommendation System Using Natural Language Processing**

---

## 2. ABSTRACT
Traditional library cataloging and search systems rely strictly on exact keyword matching, database lookup, and standardized classification codes (e.g., Dewey Decimal, ISBN). When library patrons submit conversational, natural language search queries such as *"Find interesting science fiction books for beginners"* or *"Suggest fantasy books similar to Harry Potter"*, keyword-based engines often return zero matches or irrelevant results. Furthermore, evaluating patron book reviews manually is time-consuming and subjective.

This project presents a comprehensive, production-ready **Library Book Search & Recommendation System** that unifies all **ten core Natural Language Processing (NLP) curriculum practical experiments** into a single cohesive, domain-specific architecture. The application processes raw queries and reader reviews through sentiment analysis, tokenization, script validation, stopword removal, stemming, lemmatization, morphological analysis, N-gram modeling, part-of-speech (POS) tagging, noun phrase chunking, named entity recognition (NER), TF-IDF cosine similarity recommendation, and deep learning-based Word Sense Disambiguation (WSD) using Bidirectional LSTM/GRU neural networks. Delivered as an interactive Streamlit application with a dual mode (Integrated Recommendation Engine and Individual Experiment Viva Labs), this system serves as an exemplary academic mini-project and real-world software prototype.

---

## 3. INTRODUCTION
Natural Language Processing is a critical subfield of Artificial Intelligence bridging computational linguistics, machine learning, and human communication. Libraries are knowledge hubs housing thousands of titles, authors, genres, and reader community reviews. As digital libraries expand, the need for intelligent information retrieval that understands syntax, semantics, and context becomes paramount.

Rather than treating academic NLP experiments as disjointed, abstract programming tasks, this project embeds every experiment into the lifecycle of an actual library application. A search query or book review traverses an organized linguistic pipeline, where low-level lexical tokens are transformed into high-level syntactic and semantic representations, enabling intelligent book recommendation and contextual disambiguation.

---

## 4. PROBLEM STATEMENT
Modern library patrons frequently formulate search queries using conversational, expressive phrasing:
- *"Find interesting science fiction books for beginners."*
- *"Show books written by J.K. Rowling."*
- *"Suggest books similar to Harry Potter."*
- *"I found this book very interesting and informative."*

Standard catalog search interfaces face critical challenges:
1. **Keyword Rigidity:** Fails on synonyms, morphological variations (e.g., *"reading"* vs *"read"*), and grammatical inflections.
2. **Review Processing Bottleneck:** Unable to automatically quantify reader sentiment and detect reader satisfaction.
3. **Ambiguity and Polysemy:** Words such as *"novel"* possess multiple meanings (e.g., a literary book vs an original/innovative idea), which keyword search engines cannot distinguish.
4. **Lack of Semantic Matching:** Queries describing themes or plots cannot find matching book descriptions without semantic similarity measures.

---

## 5. OBJECTIVES
The primary objectives of this project are:
1. To design and implement a single, unified Python-based NLP application integrating all 10 prescribed curriculum experiments.
2. To classify patron book reviews into Positive, Negative, and Neutral sentiments with quantitative evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
3. To build a robust preprocessing pipeline including tokenization, non-English script validation, stop word removal, Porter stemming, and WordNet lemmatization.
4. To implement morphological decomposition, inflectional word generation, and search query expansion to improve retrieval recall.
5. To develop Unigram, Bigram, and Trigram language models with sentence boundary markers (`<s>`, `</s>`) and next-word autocomplete suggestions.
6. To perform POS tagging and noun phrase chunking for semantic phrase extraction.
7. To empirically analyze the effect of feature selection and training dataset size (60%, 70%, 80%) on chunking performance.
8. To extract domain entities (Books, Authors, Publishers, Locations, Dates) using a hybrid gazetteer and NLTK NER pipeline.
9. To rank catalog books using TF-IDF vector space modeling and Cosine Similarity.
10. To disambiguate ambiguous words (*"novel"*) using a Bidirectional LSTM/GRU neural network trained on contextual sequences.
11. To deliver an intuitive, interactive Streamlit web interface with a dedicated Viva Demo mode.

---

## 6. SCOPE
- **Domain:** Library catalogs, book descriptions across 10 genres (Fantasy, Sci-Fi, History, Tech, Biography, Mystery, Romance, Programming, Self-Help, Literature), reader reviews, and search queries.
- **Academic Scope:** Direct coverage of all 10 NLP practical syllabus experiments.
- **Operational Scope:** Runs locally on any standard computer without external paid cloud APIs, utilizing local CSV datasets and saved model checkpoints.

---

## 7. SOFTWARE REQUIREMENTS
- **Operating System:** Windows 10/11, Linux, or macOS
- **Programming Language:** Python 3.11.x
- **Key Python Libraries:**
  - `nltk` (v3.8.1+) — Tokenization, Corpora, Stemming, Lemmatization, POS Tagging, NER, Chunking
  - `scikit-learn` (v1.3.0+) — TF-IDF Vectorization, Logistic Regression, Naive Bayes, Cosine Similarity, Metrics
  - `tensorflow` / `keras` (v2.15.0+) — Deep Learning LSTM/GRU Sequence Modeling
  - `pandas` (v2.0.0+) & `numpy` (v1.24.0+) — Structured Data Processing and Vector Operations
  - `streamlit` (v1.30.0+) — Interactive Web User Interface
  - `matplotlib` (v3.7.0+) & `seaborn` (v0.12.0+) — Confusion Matrices & Evaluation Visualizations
  - `wordcloud` (v1.9.0+) — Visual Text Analytics
  - `joblib` (v1.3.0+) — Model Persistence and Serialization

---

## 8. HARDWARE REQUIREMENTS
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher (Intel Core i5 recommended)
- **RAM:** Minimum 4 GB (8 GB recommended for Deep Learning training)
- **Storage:** 1 GB free disk space (datasets, NLTK corpora, virtual environment, model checkpoints)
- **Display Resolution:** 1280 x 720 or higher

---

## 9. SYSTEM ARCHITECTURE
The system operates as a layered modular architecture:

```
[ User Query / Book Review ]
             |
             v
+-----------------------------------------------------------+
|                   STREAMLIT WEB UI                        |
|  - Integrated Pipeline Mode  |  - Viva Lab Experiments    |
+-----------------------------------------------------------+
             |
             v
+-----------------------------------------------------------+
|                   NLP PROCESSING ENGINE                   |
| 1. Sentiment Analyzer (TF-IDF + Logistic Regression)      |
| 2. Preprocessing & Script Validator (Latin/ASCII check)   |
| 3. Stopwords, Stemmer (Porter), Lemmatizer (WordNet)      |
| 4. Morphology Engine & Query Expander (Roots & Affixes)   |
| 5. N-Gram Language Model (Unigrams, Bigrams, Trigrams)    |
| 6. POS Tagger (Perceptron with Grammatical Mappings)       |
| 7. NP Chunker & Training Size Evaluator (60%/70%/80%)     |
| 8. Hybrid NER Engine (Catalogue Gazetteer + ne_chunk)     |
| 9. Recommendation Engine (TF-IDF + Cosine Similarity)     |
| 10. Neural WSD Module (Bidirectional LSTM/GRU)            |
+-----------------------------------------------------------+
             |
             v
+-----------------------------------------------------------+
|                     DATA & MODELS LAYER                   |
| - books.csv (40 titles)   - library_reviews.csv (60 revs) |
| - wsd_dataset.csv (60)    - wsd_lstm_model.keras, PKL     |
+-----------------------------------------------------------+
```

---

## 10. DATASET DESCRIPTION
The system is powered by three curated, domain-specific datasets located in `data/`:
1. **`books.csv` (Book Catalog):** Contains 40 books across 10 genres: Fantasy, Science Fiction, History, Technology, Biography, Mystery, Romance, Programming, Self-Help, and Literature. Fields include `book_id`, `title`, `author`, `genre`, `published_year`, `rating`, and `description`.
2. **`library_reviews.csv` (Reader Reviews):** Contains 60 labeled reviews reflecting genuine student and reader feedback categorized into `Positive`, `Negative`, and `Neutral` sentiments.
3. **`wsd_dataset.csv` (Word Sense Disambiguation):** Contains 60 carefully curated sentences featuring the ambiguous target word *"novel"*, balanced between sense `BOOK` (narrative prose) and sense `NEW` (innovative, original).

---

## 11. COMPLETE INTEGRATED NLP PIPELINE
When a user submits a query (e.g., *"Find interesting science fiction books for beginners"*), the system executes a continuous 10-step sequence:
1. **Sentiment Detection:** Determines whether the input is an informative search query or an evaluative reader review.
2. **Tokenization & Script Validation:** Segments into sentences and words, cleans noise, and validates English script integrity.
3. **Stopword Filtering, Stemming & Lemmatization:** Normalizes tokens into dictionary base forms.
4. **Morphological Expansion:** Resolves roots and generates inflectional variants to expand retrieval keywords.
5. **N-Gram Modeling:** Computes Unigram, Bigram, and Trigram frequencies with boundary tokens `<s>` and `</s>`.
6. **POS Tagging:** Identifies grammatical syntactic categories.
7. **Phrase Chunking:** Groups adjectives and nouns into compound noun phrases (e.g., *"interesting science fiction books"*).
8. **Named Entity Recognition:** Identifies explicit books, authors, publishers, or cities mentioned in the query.
9. **Cosine Similarity Recommendation:** Vectorizes the enhanced query against the book catalog and ranks top matches.
10. **Word Sense Disambiguation:** Checks for polysemous target words and applies recurrent neural sequence inference.

---

## 12. EXPERIMENT 1 — SENTIMENT ANALYSIS OF BOOK REVIEWS
- **Purpose:** Automatically classify library reviews to identify high-quality books and reader satisfaction.
- **Algorithm:** Term Frequency-Inverse Document Frequency (TF-IDF) feature extraction coupled with Logistic Regression / Multinomial Naive Bayes.
- **Evaluation Metrics:**
  - Accuracy: Evaluates overall correctness.
  - Precision & Recall: Measures true positive retrieval vs false alarms.
  - F1-Score: Harmonic mean of precision and recall.
  - Confusion Matrix: Visualized via Seaborn heatmap.
- **Interactive UI:** Allows entering any custom review with instant sentiment label and probability distribution.

---

## 13. EXPERIMENT 2 — TOKENIZATION, FILTRATION & SCRIPT VALIDATION
- **Purpose:** Structure raw query strings into discrete lexical units and filter invalid input.
- **Techniques:**
  - Sentence Tokenization: NLTK `sent_tokenize`.
  - Word Tokenization: NLTK `word_tokenize`.
  - Filtration: Punctuation removal, non-alphanumeric filtering, and lowercasing.
  - Script Validation: Evaluates character Unicode categories to confirm standard English (Latin/ASCII) script, flagging unsupported scripts (Devanagari, Cyrillic, Arabic, CJK) or corrupted characters.

---

## 14. EXPERIMENT 3 — STOP WORD REMOVAL, STEMMING & LEMMATIZATION
- **Purpose:** Reduce dimensionality while retaining meaningful semantic content.
- **Techniques:**
  - Stopword Removal: Eliminates common grammatical function words using NLTK English stopwords corpus.
  - Porter Stemming: Heuristic suffix stripping algorithm (`PorterStemmer`).
  - WordNet Lemmatization: Morphological vocabulary lookup with POS context mapping (`WordNetLemmatizer`).
- **Output:** Comparative table contrasting raw tokens, stemmed forms, and lemmatized dictionary entries.

---

## 15. EXPERIMENT 4 — MORPHOLOGICAL ANALYSIS & WORD GENERATION
- **Purpose:** Analyze internal word structure and generate inflectional/derivational variants for search expansion.
- **Components:**
  - Root Word Extraction: Identifies base morphemes (e.g., `reading` -> `read`).
  - Affix Detection: Identifies productive English prefixes (`un-`, `re-`, `pre-`, `mis-`) and suffixes (`-ing`, `-ed`, `-er`, `-tion`, `-able`).
  - Word Generation: Rules generating plural, past tense, gerund, and agent noun forms.
  - Query Expansion: Enhances user queries by appending root forms, preventing zero-match results in catalog search.

---

## 16. EXPERIMENT 5 — N-GRAM LANGUAGE MODEL & QUERY SUGGESTIONS
- **Purpose:** Model word co-occurrence sequences and generate search autocomplete suggestions.
- **Techniques:**
  - Boundary Padding: Adds sentence start `<s>` and end `</s>` tokens.
  - Unigram, Bigram, and Trigram extraction using sliding windows.
  - Frequency Counting: `collections.Counter` tracking N-gram counts across catalog titles, reviews, and search logs.
  - Autocomplete / Next-Word Suggestion: Predicts the most likely subsequent word given prior Bigram/Trigram context.

---

## 17. EXPERIMENT 6 — PART-OF-SPEECH (POS) TAGGING
- **Purpose:** Assign grammatical categories to every token in the query.
- **Implementation:** NLTK Perceptron Tagger utilizing the Penn Treebank tagset.
- **Human-Readable Mapping:** Translates tags (`NN`, `VBD`, `JJ`, `RB`, etc.) into intuitive descriptions (Singular Noun, Past Tense Verb, Adjective, Adverb) essential for college viva presentation.

---

## 18. EXPERIMENT 7 — CHUNKING, FEATURE SELECTION & TRAINING SIZE ANALYSIS
- **Purpose:** Extract meaningful noun phrases and investigate empirical machine learning principles.
- **Chunking Implementation:** Regular Expression Parser using the grammar `<DT>?<JJ.*|VBG>*<NN.*>+`.
- **Syllabus Investigation (Feature Selection & Training Size):**
  - Features evaluated: Word token, POS tag, word length, capitalization, prefixes (`prefix_2`), suffixes (`suffix_3`), and context window (`prev_pos`, `next_pos`).
  - Training Splits: Evaluated across 60%, 70%, and 80% splits.
  - Empirical Finding: Rich contextual features and larger training sizes (80%) yield superior boundary classification accuracy compared to simple isolated word features.

---

## 19. EXPERIMENT 8 — NAMED ENTITY RECOGNITION (NER)
- **Purpose:** Identify specific library entities in patron queries.
- **Target Entities:** `BOOK`, `AUTHOR`, `PUBLISHER`, `LOCATION`, `ORGANIZATION`, `DATE`.
- **Hybrid Architecture:**
  - Domain Gazetteer Matcher: Cross-references titles, author names, publishers, and cities from the library database.
  - Temporal Regular Expressions: Extracts publication years and century references.
  - Statistical NER: NLTK `ne_chunk` fallback for general person, location, and organizational entities.

---

## 20. EXPERIMENT 9 — TEXT SIMILARITY & BOOK RECOMMENDATION ENGINE
- **Purpose:** Rank catalog books according to semantic relevance with patron search queries.
- **Mathematical Formulation:**
  - Composite text formed by concatenating title, author, genre, and description.
  - TF-IDF Vectorization:
    $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right)$$
  - Cosine Similarity:
    $$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$
- **Output:** Ranked book cards showing match percentage, genre tags, author metadata, and synopsis.

---

## 21. EXPERIMENT 10 — WORD SENSE DISAMBIGUATION USING LSTM/GRU
- **Purpose:** Resolve semantic polysemy in library texts using Deep Learning recurrent neural networks.
- **Target Polysemous Word:** *"novel"*
  - **Sense 1 (BOOK):** A fictional literary work in prose.
  - **Sense 2 (NEW):** An original, innovative, or unprecedented method/idea.
- **Deep Learning Architecture:**
  1. Input Sentence -> Keras Text Tokenizer (`max_vocab=500`, `max_len=25`)
  2. Sequence Padding (`pad_sequences`)
  3. Embedding Layer (`embedding_dim=32`)
  4. Spatial Dropout (0.2)
  5. Bidirectional LSTM / GRU Layer (32 units)
  6. Dense Layer (16 units, ReLU activation)
  7. Output Softmax Layer (Class probabilities for `BOOK` vs `NEW`)
- **Optimization:** Adam optimizer, sparse categorical cross-entropy loss, trained over 25 epochs.

---

## 22. RESULTS AND PERFORMANCE METRICS

### Sentiment Analysis (Exp 1)
- **Accuracy:** 73.3% – 86.7% (depending on split)
- **Weighted F1-Score:** 0.81+
- **Confusion Matrix:** High true positive rate across Positive and Negative classes.

### Feature Selection & Training Size on Chunking (Exp 7)
| Feature Set | Training Size | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Basic (Word + POS) | 60% | 0.9487 | 0.9490 | 0.9487 | 0.9471 |
| Basic (Word + POS) | 70% | 0.9655 | 0.9670 | 0.9655 | 0.9647 |
| Basic (Word + POS) | 80% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Full Contextual Features | 60% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Full Contextual Features | 70% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Full Contextual Features | 80% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### WSD LSTM Neural Network (Exp 10)
- **Test Accuracy:** 75.0% – 83.3% on unseen test sentences.
- **Prediction Confidence:** 98%+ on clear contextual sentences (e.g., *"I borrowed a novel from the library"* -> `BOOK` at 1.00 confidence).

---

## 23. SCREENSHOTS & SYSTEM DEMONSTRATION LAYOUT
The Streamlit application provides a clean, responsive layout:
1. **Header Section:** Modern blue header with course, title, and quick-example query buttons.
2. **Tabbed Pipeline View:** 10 sequential tabs displaying real-time metrics, tables, bar charts, and parse trees.
3. **Recommendation Cards:** Highlighting top books with similarity percentage badges, metadata, and synopses.
4. **Viva Demo Laboratory:** Dedicated sidebar mode with interactive sliders for training size, model selectors, confusion matrices, and live WSD testing.

---

## 24. ADVANTAGES OF THE PROPOSED SYSTEM
1. **Holistic Integration:** Connects 10 discrete academic practicals into one working real-world application.
2. **Context-Aware Recommendations:** Combines morphological query expansion with TF-IDF cosine similarity.
3. **Polysemy Resolution:** Leverages recurrent deep learning (LSTM/GRU) to eliminate semantic ambiguity.
4. **Script Validation:** Prevents encoding corruption and flags non-English scripts.
5. **Zero External API Cost:** Runs entirely offline on local CPU without reliance on expensive third-party APIs.
6. **Viva-Ready Design:** Features dedicated demonstration controls and evaluation metric visualizations for oral exams.

---

## 25. LIMITATIONS
1. **Dataset Scale:** The local catalog contains 40 books and 60 reviews, which is ideal for classroom demonstration but would require distributed indexing (e.g., Elasticsearch) for millions of library volumes.
2. **Polysemy Scope:** The deep learning WSD module currently focuses on core library polysemous terms (*"novel"*), though its architecture easily scales to additional words.
3. **Language Scope:** Currently optimized for the English language.

---

## 26. REAL-WORLD APPLICATIONS
- **University and College Libraries:** Upgrading legacy OPAC (Online Public Access Catalog) systems into semantic discovery portals.
- **Public & Community Libraries:** Automated sentiment analysis of reader book feedback.
- **Digital Archives & Bookstores:** Semantic search engines and book recommendation systems for digital reading platforms.
- **Academic Research & NLP Education:** Comprehensive educational reference implementation for computer science students.

---

## 27. FUTURE SCOPE
1. **Transformer Integration:** Incorporating pre-trained BERT/RoBERTa embeddings for sentence-level similarity.
2. **Multilingual Support:** Extending script validation and lemmatization to Hindi, Marathi, and other regional languages.
3. **Hybrid Collaborative Filtering:** Blending user borrowing history with content-based TF-IDF similarity.
4. **Speech-to-Text Querying:** Enabling voice search for visually impaired library visitors.

---

## 28. CONCLUSION
The **Library Book Search & Recommendation System Using Natural Language Processing** successfully demonstrates how classical computational linguistics, statistical machine learning, and deep recurrent neural networks can be harmonized into an end-to-end software solution. By completing all ten curriculum experiments within a unified library domain, the project fulfills all academic mini-project criteria and provides students with an impressive, demonstrable application for college practical examinations and viva-voce.

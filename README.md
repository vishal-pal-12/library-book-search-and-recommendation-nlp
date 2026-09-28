# 📚 Library Book Search & Recommendation System Using NLP

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-2E7D32?style=for-the-badge)](https://www.nltk.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow / Keras](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

> **College NLP Mini Project**  
> **Domain:** Library Information Retrieval, Reader Sentiment Analytics & Semantic Recommendations  
> **Architecture:** 10 Fully Connected NLP Experiments functioning as one unified system  
> **Author:** Vishal Pal ([@vishal-pal-12](https://github.com/vishal-pal-12))

---

## 📌 Table of Contents
- [1. Problem Statement & Motivation](#-1-problem-statement--motivation)
- [2. Complete System Architecture](#-2-complete-system-architecture)
- [3. The 10 Connected NLP Experiments](#-3-the-10-connected-nlp-experiments)
- [4. Project Directory Structure](#-4-project-directory-structure)
- [5. Installation & Setup](#-5-installation--setup)
- [6. Three Ways to Run & Demonstrate](#-6-three-ways-to-run--demonstrate)
  - [Way 1: Interactive CLI Predictor (`book_predictor.py`)](#way-1-interactive-cli-predictor-book_predictorpy)
  - [Way 2: Interactive Jupyter Notebook (`.ipynb`)](#way-2-interactive-jupyter-notebook-ipynb)
  - [Way 3: Full Streamlit Web Portal (`app.py`)](#way-3-full-streamlit-web-portal-apppy)
- [7. Evaluation Metrics & Experimental Results](#-7-evaluation-metrics--experimental-results)
- [8. College Viva Defense & Common Q&A](#-8-college-viva-defense--common-qa)
- [9. Datasets](#-9-datasets)
- [10. License](#-10-license)

---

## 🎯 1. Problem Statement & Motivation

Modern university and public libraries house extensive catalogs across diverse genres. Readers and patrons frequently interact with library systems using natural language queries such as:
- *"Find interesting science fiction books for beginners."*
- *"Books written by J.K. Rowling."*
- *"Suggest books on machine learning and artificial intelligence."*
- *"I found this book very informative and easy to understand."* (Reader Review)
- *"The researcher proposed a novel approach to solve the problem."* (Polysemous search query)

Standard keyword search mechanisms often suffer from low recall, fail to recognize synonyms and inflectional word variants, cannot parse complex grammatical phrases, and cannot determine reader review sentiment.

This project integrates **10 core natural language processing techniques** into one unified, production-ready system to evaluate reader reviews, extract syntactic and semantic phrases, resolve word ambiguity, and recommend relevant books with mathematical vector precision.

---

## 🏛️ 2. Complete System Architecture

```
                       LIBRARY SEARCH QUERY / READER REVIEW
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   STAGE 1: SENTIMENT ANALYSIS (Exp 1)     │
                  │   TF-IDF + Classifier (Pos / Neg / Neu)   │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   STAGE 2: TOKENIZATION & SCRIPT CHECK    │
                  │   Sentence / Word Tokens + Non-ASCII Flag │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   STAGE 3: STOPWORDS, STEM & LEMMA        │
                  │   Stopwords Removal + Porter + WordNet    │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   STAGE 4: MORPHOLOGICAL WORD GENERATION  │
                  │   Root & Affixes + Query Expansion Form   │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   STAGE 5: N-GRAM LANGUAGE MODELING       │
                  │   Unigram / Bigram / Trigram Predictions  │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   STAGE 6: PART-OF-SPEECH (POS) TAGGING   │
                  │   Penn Treebank Syntactic Roles           │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   STAGE 7: NOUN PHRASE CHUNKING           │
                  │   Regexp Chunk Parser + 60/70/80% Study   │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   STAGE 8: NAMED ENTITY RECOGNITION (NER) │
                  │   Books, Authors, Publishers, Locations   │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   STAGE 9: TEXT SIMILARITY ENGINE         │
                  │   TF-IDF Vector Space + Cosine Similarity │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   STAGE 10: WORD SENSE DISAMBIGUATION     │
                  │   Bidirectional LSTM Neural Network (WSD) │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                    FINAL RECOMMENDED BOOKS & LINGUISTIC REPORT
```

---

## 🧪 3. The 10 Connected NLP Experiments

| Exp # | Experiment Title | Module | Syllabus Role & Real-World Application |
|---|---|---|---|
| **Exp 1** | **Sentiment Analysis** | [`modules/sentiment.py`](modules/sentiment.py) | Evaluates reader book reviews using TF-IDF and Logistic Regression / Naive Bayes with Confusion Matrix and Precision/Recall/F1 metrics. |
| **Exp 2** | **Tokenization, Filtration & Script Validation** | [`modules/preprocessing.py`](modules/preprocessing.py) | Tokenizes queries into sentences and words, cleans punctuation, and validates English Latin scripts against unsupported character sets (Devanagari, Cyrillic, etc.). |
| **Exp 3** | **Stop Word Removal, Stemming & Lemmatization** | [`modules/preprocessing.py`](modules/preprocessing.py) | Removes grammatical stopwords, applies Porter Stemming, and applies POS-aware WordNet Lemmatization. |
| **Exp 4** | **Morphological Analysis & Word Generation** | [`modules/morphology.py`](modules/morphology.py) | Extracts base morphemes, prefixes, and suffixes; automatically generates word forms (e.g., `read` -> `reads, reading, reader`) to expand catalog queries. |
| **Exp 5** | **N-Gram Language Model** | [`modules/ngram.py`](modules/ngram.py) | Models Unigrams, Bigrams, and Trigrams with boundary tokens (`<s>`, `</s>`) for next-word suggestions and search autocomplete. |
| **Exp 6** | **Part-of-Speech (POS) Tagging** | [`modules/pos_tagging.py`](modules/pos_tagging.py) | Tags words with Penn Treebank POS markers and maps them to human-readable grammatical roles (Noun, Verb, Adjective, Modifier). |
| **Exp 7** | **Chunking + Feature Selection + Training Size Study** | [`modules/chunking.py`](modules/chunking.py) | Regexp Noun Phrase chunker (`<DT>?<JJ.*>*<NN.*>+`) for semantic phrase extraction, accompanied by an empirical model study comparing 60%, 70%, and 80% training splits. |
| **Exp 8** | **Named Entity Recognition (NER)** | [`modules/ner.py`](modules/ner.py) | Hybrid catalog gazetteer + NLTK NER identifying `BOOK`, `AUTHOR`, `PUBLISHER`, `LOCATION`, `ORGANIZATION`, and `DATE`. |
| **Exp 9** | **Text Similarity & Book Recommendation** | [`modules/similarity.py`](modules/similarity.py) | Content-based recommendation calculating Cosine Similarity over TF-IDF vectors of 40 catalog books across 10 genres. |
| **Exp 10** | **Word Sense Disambiguation (Bi-LSTM / GRU)** | [`modules/wsd.py`](modules/wsd.py) | Deep learning Bidirectional LSTM sequence model classifying polysemous words (e.g., resolving whether *"novel"* refers to a `BOOK` or `NEW`). |

---

## 📂 4. Project Directory Structure

```
library_nlp_project/
├── app.py                      # Academic Streamlit Web Portal (Presentation & Viva Demo)
├── book_predictor.py           # Interactive CLI Book Predictor for VS Code / Terminal
├── predict_book.py             # CLI Prediction Engine
├── test_all_modules.py         # 100% Automated Unit Test Suite verifying all 10 modules
├── requirements.txt            # Python environment dependencies
├── README.md                   # Professional project documentation
├── PROJECT_REPORT.md           # Complete 28-section academic university project report
├── .gitignore                  # Git ignore rules for Python, models, caches
│
├── data/                       # Domain Datasets
│   ├── books.csv               # 40 curated books across 10 academic genres
│   ├── library_reviews.csv     # 60 labeled positive/negative/neutral reader reviews
│   └── wsd_dataset.csv         # 60 polysemous context sentences for 'novel'
│
├── modules/                    # The 10 NLP Experiment Modules
│   ├── __init__.py
│   ├── sentiment.py            # Exp 1: Review Sentiment Classifier
│   ├── preprocessing.py        # Exp 2 & 3: Preprocessing, Script Validation & Lemmatization
│   ├── morphology.py           # Exp 4: Morpheme Decomposition & Query Expansion
│   ├── ngram.py                # Exp 5: N-Gram Autocomplete & Language Modeling
│   ├── pos_tagging.py          # Exp 6: POS Tagging & Grammatical Role Mapping
│   ├── chunking.py             # Exp 7: Regexp NP Chunking & 60/70/80% Training Study
│   ├── ner.py                  # Exp 8: Hybrid Gazetteer & NER Engine
│   ├── similarity.py           # Exp 9: TF-IDF & Cosine Similarity Recommender
│   └── wsd.py                  # Exp 10: Deep Learning Bi-LSTM Polysemy Disambiguator
│
├── models/                     # Serialized Pre-Trained Models
│   ├── sentiment_vectorizer.pkl
│   ├── sentiment_logistic_regression.pkl
│   ├── wsd_tokenizer.pkl
│   └── wsd_lstm_model.keras    # Pre-trained Keras Bi-LSTM Model
│
└── Library_Book_Search_and_Recommendation_NLP_Project.ipynb  # Unified Jupyter Notebook
```

---

## ⚙️ 5. Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/vishal-pal-12/library-book-search-and-recommendation-nlp.git
cd library-book-search-and-recommendation-nlp
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Test Verification
Ensure all 10 experiments are passing:
```bash
python test_all_modules.py
```
*(All 10 tests will execute and confirm `[PASSED]` with zero errors).*

---

## 💻 6. Three Ways to Run & Demonstrate

### Way 1: Interactive CLI Predictor (`book_predictor.py`)

Run directly inside VS Code or command prompt:

```bash
python book_predictor.py
```

The system presents an interactive menu of sample catalog books:
```text
Sample Book Titles you can test from catalog:
  [ 1] Harry Potter and the Sorcerer's Stone
  [ 2] The Hobbit
  [ 3] Dune
  [ 4] Foundation
  [ 5] Clean Code: A Handbook of Agile Software Craftsmanship
  [ 6] Python Crash Course
  [ 7] 1984
  [ 8] Sapiens: A Brief History of Humankind
  [ 9] The Da Vinci Code
  [10] Cosmos
------------------------------------------------------------------------------
Enter Book Name (or 1-10, Enter for Harry Potter): 1
```

Or pass any book title as a direct CLI argument:
```bash
python book_predictor.py "Dune"
python book_predictor.py "Clean Code"
python book_predictor.py "1984"
```

**Output:**
- Fetches target book details (Author, Genre, Rating, Synopsis)
- `[Exp 2 & 3] Preprocessing`: Lemmatized and stopword-filtered tokens
- `[Exp 6] POS Tagging`: Grammatical tags (`A/DT`, `young/JJ`, `wizard/JJ`, etc.)
- `[Exp 7] Chunking`: Key Noun Phrases (`"wizard discovers"`, `"magical heritage"`)
- `[Exp 8] NER`: Entities identified (`Hogwarts School`, `Witchcraft`, `Wizardry`)
- `[Exp 9] Similarity Engine`: Top 5 predicted similar books ranked by **TF-IDF Cosine Similarity** with match percentages and synopses!

---

### Way 2: Interactive Jupyter Notebook (`.ipynb`)

Open [`Library_Book_Search_and_Recommendation_NLP_Project.ipynb`](Library_Book_Search_and_Recommendation_NLP_Project.ipynb) inside VS Code or JupyterLab:

1. Run all cells sequentially (Shift + Enter).
2. Use **Cell 29** to type custom queries/reviews on the unified 10-stage pipeline.
3. Use **Cell 31** to enter any book title from the dataset to trigger interactive content-based predictions with pandas tables and visualizations.

---

### Way 3: Full Streamlit Web Portal (`app.py`)

Launch the modern academic presentation portal:

```bash
streamlit run app.py
```
Then navigate to `http://localhost:8501`.

**Features in Portal:**
- **🚀 Integrated Search & Recommendation:**
  - Free-text natural language queries with full 10-stage breakdown
  - Catalog book selector for instant visual book cards and match ratings
- **🧪 NLP Experiments Laboratory (Viva Demo Mode):**
  - Dedicated tabs for Experiments 1 through 10 with interactive sliders, model retrainers, parse tree visualizers, and confusion matrices
- **📖 Library Catalog & Datasets:**
  - Searchable catalog tables, filter by genre, and review datasets

---

## 📊 7. Evaluation Metrics & Experimental Results

### Experiment 1: Sentiment Analysis
- **Model:** TF-IDF Vectorizer (1,2-grams) + Logistic Regression / Naive Bayes
- **Dataset:** 60 library reviews (Balanced: 30 Positive, 30 Negative)
- **Accuracy:** `~80.0%` | **Macro F1-Score:** `~0.80`

### Experiment 7: Chunking Feature Selection & Training Size Study
Empirical evaluation comparing feature sets across 60%, 70%, and 80% splits:

| Training Split | Basic Features (Word + POS) Accuracy | Full Features (+ Context + Affixes) Accuracy |
|---|---|---|
| **60%** | `94.8%` | `100.0%` |
| **70%** | `96.5%` | `100.0%` |
| **80%** | `100.0%` | `100.0%` |

*Insight:* Incorporating contextual neighboring words ($w_{i-1}, w_{i+1}$) and morphological prefixes/suffixes reduces reliance on large training corpora, achieving optimal accuracy at earlier thresholds.

### Experiment 10: Deep Learning Word Sense Disambiguation
- **Architecture:** Embedding (64-dim) $\rightarrow$ Bidirectional LSTM (32 units) $\rightarrow$ Dropout (0.3) $\rightarrow$ Dense (16, ReLU) $\rightarrow$ Softmax (2 classes: `BOOK` vs `NEW`)
- **Dataset:** 60 polysemous sentences featuring the ambiguous word *"novel"*
- **Sample Predictions:**
  - *"I borrowed a novel from the library."* $\rightarrow$ **BOOK** (Confidence: `99.8%`)
  - *"The researcher proposed a novel method."* $\rightarrow$ **NEW** (Confidence: `71.4%`)

---

## 🎓 8. College Viva Defense & Common Q&A

**Q1: Why is POS Tagging essential before Chunking and Lemmatization?**  
> *Answer:* POS tagging resolves grammatical ambiguity (e.g., whether "read" is a verb or a noun). WordNet Lemmatizer requires the POS tag to correctly convert "running" to "run" (verb) rather than keeping it unchanged. Regexp chunkers rely on POS tag patterns (`<DT>?<JJ>*<NN>+`) to detect phrase boundaries.

**Q2: What is the mathematical basis of Experiment 9 (Text Similarity)?**  
> *Answer:* Text descriptions are converted into TF-IDF vectors in high-dimensional vector space. The similarity between book $A$ and book $B$ is measured by the cosine of the angle between their vectors:
> $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$
> A score of 1.0 indicates identical vocabulary distributions, while 0.0 indicates orthogonal/unrelated topics.

**Q3: Why use an LSTM for Word Sense Disambiguation rather than dictionary lookup?**  
> *Answer:* Polysemous words like *"novel"* depend heavily on sequential context. In *"I borrowed a novel"*, *"borrowed"* indicates a physical book. In *"A novel algorithm"*, *"algorithm"* implies novelty. Bi-directional LSTMs capture long-range forward and backward syntactic dependencies that simple dictionary lookups miss.

---

## 📚 9. Datasets

All datasets are included in the repository under [`data/`](data/):
1. **`books.csv`**: 40 curated books across 10 genres (*Fantasy, Science Fiction, History, Biography, Mystery, Romance, Programming, Technology, Self Help, Literature*) with ISBN, Title, Author, Genre, Publication Year, Rating, and Detailed Descriptions.
2. **`library_reviews.csv`**: 60 reader satisfaction reviews labeled as Positive or Negative.
3. **`wsd_dataset.csv`**: 60 sentences containing *"novel"* in both literature and innovation contexts.

---

## 📄 10. License

This project is licensed under the **MIT License** — feel free to use it for academic coursework, college viva demonstrations, and educational reference.

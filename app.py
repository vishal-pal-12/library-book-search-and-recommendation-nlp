"""
Library Book Search & Recommendation System Using Natural Language Processing
College NLP Mini Project integrating Experiments 1 through 10.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 and project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.sentiment import sentiment_analyzer
from modules.preprocessing import tokenize_and_filter, preprocess_text, validate_script
from modules.morphology import analyze_morphology, expand_query_morphologically, generate_word_forms
from modules.ngram import ngram_model
from modules.pos_tagging import tag_pos
from modules.chunking import extract_chunks, analyze_training_size_and_features
from modules.ner import extract_named_entities
from modules.similarity import book_recommender
from modules.wsd import wsd_model

# Streamlit Page Config
st.set_page_config(
    page_title="Library NLP Search & Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .exp-badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        text-align: center;
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .book-card {
        border: 1px solid #E2E8F0;
        border-left: 5px solid #3B82F6;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/library.png", width=80)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose System Mode:",
    [
        "🚀 Integrated Search & Recommendation",
        "🧪 NLP Experiments Lab (Viva Demo)",
        "📖 Library Catalog & Datasets"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**College NLP Mini Project**
- **Domain:** Library Books & Reviews
- **Architecture:** 10 Unified Experiments
- **Deep Learning:** Bi-LSTM / GRU for WSD
""")

# ==============================================================================
# MODE 1: INTEGRATED SEARCH & RECOMMENDATION SYSTEM
# ==============================================================================
if app_mode == "🚀 Integrated Search & Recommendation":
    st.markdown('<div class="main-title">📚 Library Book Search & Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A Unified Natural Language Processing Pipeline for Query Processing, Review Analysis, and Semantic Book Recommendation</div>', unsafe_allow_html=True)

    mode_selection = st.radio(
        "Select Operation Mode:",
        [
            "🔍 Process Search Query / Review (10-Stage Pipeline)",
            "🎯 Predict Similar Books by Book Title (Catalog Similarity)"
        ],
        horizontal=True
    )

    if mode_selection == "🔍 Process Search Query / Review (10-Stage Pipeline)":
        # Example query quick select
        st.markdown("**Sample Natural Language Queries / Reviews:**")
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        query_text = ""
        if col_q1.button("🔍 Science Fiction for Beginners"):
            st.session_state["query_input"] = "Find interesting science fiction books for beginners."
        if col_q2.button("🧙 Harry Potter by J.K. Rowling"):
            st.session_state["query_input"] = "Find Harry Potter books written by J.K. Rowling."
        if col_q3.button("⭐ Book Review Analysis"):
            st.session_state["query_input"] = "This book is excellent and very informative."
        if col_q4.button("🧠 Context Sense (WSD)"):
            st.session_state["query_input"] = "The researcher proposed a novel method to solve the equation."

        default_val = st.session_state.get("query_input", "Find interesting science fiction books for beginners.")
        user_query = st.text_input("Enter your book search query, reader review, or question:", value=default_val, key="main_search_input")

        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([2, 2, 2, 4])
        run_all = col_btn1.button("⚡ Run Complete NLP Pipeline", type="primary")
        run_search = col_btn2.button("🔍 Search & Recommend")
        run_sentiment = col_btn3.button("❤️ Analyze Sentiment")

        if run_all or run_search or run_sentiment:
            if not user_query.strip():
                st.warning("Please enter a query or review first!")
            else:
                st.markdown("---")
                st.subheader("NLP Pipeline Execution Results")

                # Execute pipeline modules
                with st.spinner("Processing text through all 10 NLP stages..."):
                    # 1. Sentiment Analysis
                    sent_res = sentiment_analyzer.predict(user_query)

                    # 2. Tokenization, Filtration & Script Validation
                    exp2_res = tokenize_and_filter(user_query)

                    # 3. Stop Word Removal, Stemming & Lemmatization
                    prep_res = preprocess_text(user_query)

                    # 4. Morphological Analysis & Query Expansion
                    morph_res = expand_query_morphologically(prep_res["lemmatized_tokens"])

                    # 5. N-Gram Model
                    ngram_res = ngram_model.analyze_query(user_query)
                    suggestions = ngram_model.suggest_next_word(prep_res["lemmatized_tokens"][-2:] if len(prep_res["lemmatized_tokens"]) >= 2 else prep_res["lemmatized_tokens"])

                    # 6. POS Tagging
                    pos_res = tag_pos(user_query)

                    # 7. Chunking
                    chunk_res = extract_chunks(pos_res["tagged_tuples"])

                    # 8. Named Entity Recognition
                    ner_res = extract_named_entities(user_query)

                    # 9. Book Recommendation
                    rec_res = book_recommender.recommend_books(user_query, top_k=5)

                    # 10. Word Sense Disambiguation
                    has_novel = "novel" in user_query.lower()
                    wsd_res = wsd_model.disambiguate(user_query) if has_novel else None

                # Render Pipeline Stages
                tabs = st.tabs([
                    "1. Sentiment",
                    "2. Tokenization & Script",
                    "3. Stem & Lemma",
                    "4. Morphology",
                    "5. N-Grams",
                    "6. POS Tags",
                    "7. Chunking",
                    "8. Entities (NER)",
                    "9. Recommendations",
                    "10. WSD (LSTM)"
                ])

                # Tab 1: Sentiment
                with tabs[0]:
                    st.markdown('<span class="exp-badge">Experiment 1</span> **Sentiment Analysis**', unsafe_allow_html=True)
                    s_col1, s_col2, s_col3 = st.columns(3)
                    sentiment_label = sent_res["sentiment"]
                    color = "green" if sentiment_label == "Positive" else "red" if sentiment_label == "Negative" else "gray"
                    s_col1.metric("Predicted Sentiment", sentiment_label)
                    s_col2.metric("Confidence Score", f"{sent_res['confidence']*100:.1f}%")
                    s_col3.metric("Review Type", "Evaluated Reader Review" if "book" in user_query.lower() else "Search Query")
                
                    if sent_res.get("probabilities"):
                        st.write("**Class Probability Distribution:**")
                        prob_df = pd.DataFrame(list(sent_res["probabilities"].items()), columns=["Sentiment", "Probability"])
                        st.bar_chart(prob_df.set_index("Sentiment"))

                # Tab 2: Tokenization & Script Validation
                with tabs[1]:
                    st.markdown('<span class="exp-badge">Experiment 2</span> **Tokenization, Filtration & Script Validation**', unsafe_allow_html=True)
                    script_info = exp2_res["script_validation"]
                    if script_info["is_valid"]:
                        st.success(f"✓ Script Validation: {script_info['script']} (Accepted)")
                    else:
                        st.error(f"⚠ Script Validation: {script_info['warning_message']}")

                    st.write("**Sentence Tokens:**", exp2_res["sentence_tokens"])
                    st.write("**Raw Word Tokens:**", exp2_res["word_tokens"])
                    st.write("**Filtered Tokens (Lowercased & Cleaned):**", exp2_res["filtered_tokens"])

                # Tab 3: Stopwords, Stemming & Lemmatization
                with tabs[2]:
                    st.markdown('<span class="exp-badge">Experiment 3</span> **Stop Word Removal, Porter Stemming & WordNet Lemmatization**', unsafe_allow_html=True)
                    st.write("**Stop Word Removed Tokens:**", prep_res["stopword_removed_tokens"])
                
                    st.markdown("**Comparative Linguistic Transformation:**")
                    df_compare = pd.DataFrame(prep_res["comparison_table"])
                    st.dataframe(df_compare, use_container_width=True)

                # Tab 4: Morphology & Word Generation
                with tabs[3]:
                    st.markdown('<span class="exp-badge">Experiment 4</span> **Morphological Analysis & Query Expansion**', unsafe_allow_html=True)
                    st.write("**Morphologically Expanded Tokens for Search:**", morph_res["expanded_tokens"])
                    st.markdown("**Word Root & Affix Decomposition:**")
                    morph_records = []
                    for tok, details in morph_res["expansion_details"].items():
                        morph_records.append({
                            "Original Token": tok,
                            "Base / Root": details["root"],
                            "Generated Word Forms": ", ".join(details["expanded_to"])
                        })
                    st.dataframe(pd.DataFrame(morph_records), use_container_width=True)

                # Tab 5: N-Grams
                with tabs[4]:
                    st.markdown('<span class="exp-badge">Experiment 5</span> **N-Gram Model & Next-Word Suggestions**', unsafe_allow_html=True)
                    c_u, c_b, c_t = st.columns(3)
                    with c_u:
                        st.write("**Unigrams:**")
                        st.write(ngram_res["unigrams_raw"])
                    with c_b:
                        st.write("**Bigrams:**")
                        st.write(ngram_res["bigrams_raw"])
                    with c_t:
                        st.write("**Trigrams:**")
                        st.write(ngram_res["trigrams_raw"])

                    st.markdown("**Library Search Autocomplete / Suggestions:**")
                    if suggestions:
                        st.success(f"Suggested next library words: **{', '.join([s['word'] for s in suggestions])}**")
                    else:
                        st.info("No specific autocomplete suggestions for this tail token.")

                # Tab 6: POS Tagging
                with tabs[5]:
                    st.markdown('<span class="exp-badge">Experiment 6</span> **Part-of-Speech Tagging**', unsafe_allow_html=True)
                    df_pos = pd.DataFrame(pos_res["structured_results"])
                    st.dataframe(df_pos[["word", "tag", "category", "description"]], use_container_width=True)

                # Tab 7: Chunking
                with tabs[6]:
                    st.markdown('<span class="exp-badge">Experiment 7</span> **Phrase Chunking (Noun Phrases)**', unsafe_allow_html=True)
                    if chunk_res["extracted_phrases"]:
                        for p in chunk_res["extracted_phrases"]:
                            st.info(f"🏷️ Extracted Phrase: **{p['phrase']}** ({p['label']})")
                    else:
                        st.info("No compound noun phrases detected in this query.")
                    st.text("Parse Tree Structure:")
                    st.code(chunk_res["parse_tree"])

                # Tab 8: Named Entity Recognition
                with tabs[7]:
                    st.markdown('<span class="exp-badge">Experiment 8</span> **Named Entity Recognition (NER)**', unsafe_allow_html=True)
                    if ner_res["entities"]:
                        df_ner = pd.DataFrame(ner_res["entities"])
                        st.dataframe(df_ner[["entity", "label", "canonical_name", "source", "confidence"]], use_container_width=True)
                    else:
                        st.info("No named entities (Books, Authors, Publishers, Locations) found in this query.")

                # Tab 9: Recommendations
                with tabs[8]:
                    st.markdown('<span class="exp-badge">Experiment 9</span> **Text Similarity & Book Recommendations**', unsafe_allow_html=True)
                    st.markdown(f"**Processed Semantic Query:** `{rec_res['processed_query']}`")
                
                    recs = rec_res["recommendations"]
                    if recs:
                        for i, book in enumerate(recs, 1):
                            st.markdown(f"""
                            <div class="book-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <h4 style="margin:0; color:#1E3A8A;">#{i} {book['title']}</h4>
                                    <span style="background-color:#EBF5FF; color:#1D4ED8; font-weight:700; padding:2px 10px; border-radius:12px;">
                                        Match: {book['match_percentage']}
                                    </span>
                                </div>
                                <p style="margin:4px 0; color:#475569; font-size:0.95rem;">
                                    <strong>Author:</strong> {book['author']} | <strong>Genre:</strong> {book['genre']} | <strong>Year:</strong> {book['published_year']} | ⭐ <strong>Rating:</strong> {book['rating']}/5
                                </p>
                                <p style="margin:6px 0 0 0; color:#334155; font-size:0.9rem;">
                                    <em>"{book['description']}"</em>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No matching books found.")

                # Tab 10: WSD
                with tabs[9]:
                    st.markdown('<span class="exp-badge">Experiment 10</span> **Word Sense Disambiguation (LSTM / GRU)**', unsafe_allow_html=True)
                    if wsd_res:
                        st.success(f"Disambiguating word: **'{wsd_res['target_word']}'**")
                        w_col1, w_col2 = st.columns(2)
                        w_col1.metric("Predicted Sense", wsd_res["predicted_sense"])
                        w_col2.metric("Neural Confidence", f"{wsd_res['confidence']*100:.1f}%")
                        st.markdown(f"**Sense Meaning:** {wsd_res['meaning_explanation']}")
                        st.write("**Neural Output Probabilities:**", wsd_res["probabilities"])
                    else:
                        st.info("The polysemous target word **'novel'** was not detected in this input sentence.")
                        st.caption("Try entering a sentence containing 'novel' (e.g., 'I borrowed a novel from the library' or 'The researcher proposed a novel approach') to trigger Experiment 10.")


    else:
        st.markdown("### 📚 Find Similar Books by Catalog Title")
        st.markdown("Select any book or type its title from our library catalog to predict similar recommendations using TF-IDF Cosine Similarity and extract key linguistic features.")

        df_b = book_recommender.df_books
        all_titles = df_b["title"].tolist()
        
        col_sel, col_custom = st.columns([1, 1])
        with col_sel:
            selected_dropdown = st.selectbox("Choose a Book from Catalog:", all_titles, index=0)
        with col_custom:
            custom_title = st.text_input("Or enter title / keyword (optional):", placeholder="e.g. Dune, Harry Potter, 1984")
        
        active_title = custom_title.strip() if custom_title.strip() else selected_dropdown
        
        if st.button("🎯 Predict Similar Books", type="primary"):
            sim_res = book_recommender.find_similar_to_book(active_title, top_k=5)
            if "error" in sim_res:
                st.error(sim_res["error"])
            else:
                st.markdown("---")
                st.markdown(f"#### 📖 Target Book: **{sim_res['target_book']}**")
                col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                col_t1.metric("Author", sim_res["target_author"])
                col_t2.metric("Genre", sim_res["target_genre"])
                col_t3.metric("Year", str(sim_res["target_year"]))
                col_t4.metric("Rating", f"⭐ {sim_res['target_rating']}/5")
                
                st.info(f"**Synopsis:** {sim_res['target_description']}")
                
                pos_info = tag_pos(sim_res["target_description"])
                chunk_info = extract_chunks(pos_info["tagged_tuples"])
                ner_info = extract_named_entities(sim_res["target_description"])
                
                with st.expander("🧠 NLP Linguistic Analysis (Key Phrases & Entities)", expanded=True):
                    col_nlp1, col_nlp2 = st.columns(2)
                    with col_nlp1:
                        st.markdown("**Key Noun Phrases (Chunking):**")
                        if chunk_info["extracted_phrases"]:
                            phrases = [p["phrase"] if isinstance(p, dict) else str(p) for p in chunk_info["extracted_phrases"][:5]]
                            st.write(", ".join([f"`{p}`" for p in phrases]))
                        else:
                            st.write("None detected")
                    with col_nlp2:
                        st.markdown("**Named Entities (NER):**")
                        if ner_info["entities"]:
                            ents = [f"`{e['entity']}` ({e['label']})" for e in ner_info["entities"][:5]]
                            st.write(", ".join(ents))
                        else:
                            st.write("None detected")

                st.markdown("---")
                st.markdown(f"#### 🎯 Top {len(sim_res['recommendations'])} Predicted Similar Books (TF-IDF Cosine Similarity)")
                for b in sim_res["recommendations"]:
                    st.markdown(f"""
                    <div class="book-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4 style="margin:0; color:#1E3A8A;">{b['title']}</h4>
                            <span style="background-color:#E0F2FE; color:#0369A1; font-weight:700; padding:3px 10px; border-radius:12px;">Match: {b['match_percentage']}</span>
                        </div>
                        <p style="margin:4px 0; color:#4B5563; font-size:0.9rem;">
                            <strong>Author:</strong> {b['author']} | <strong>Genre:</strong> {b['genre']} | <strong>Rating:</strong> ⭐ {b['rating']}/5 | <strong>Year:</strong> {b['published_year']}
                        </p>
                        <p style="margin:6px 0 0 0; color:#1F2937; font-size:0.95rem;">{b['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ==============================================================================
# MODE 2: NLP EXPERIMENTS LAB (VIVA DEMO MODE)
# ==============================================================================
elif app_mode == "🧪 NLP Experiments Lab (Viva Demo)":
    st.markdown('<div class="main-title">🧪 NLP Experiments Laboratory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive Laboratory Environment for Demonstrating Experiments 1 Through 10 Individually During Viva & Practical Examinations</div>', unsafe_allow_html=True)

    exp_choice = st.selectbox(
        "Select Experiment to Demonstrate:",
        [
            "Experiment 1: Sentiment Analysis (TF-IDF + ML)",
            "Experiment 2: Tokenization, Filtration & Script Validation",
            "Experiment 3: Stop Word Removal, Stemming & Lemmatization",
            "Experiment 4: Morphological Analysis & Word Generation",
            "Experiment 5: N-Gram Language Model & Query Suggestions",
            "Experiment 6: Part-of-Speech (POS) Tagging",
            "Experiment 7: Chunking + Feature Selection + Training Size (60%/70%/80%)",
            "Experiment 8: Named Entity Recognition (NER)",
            "Experiment 9: Text Similarity & Book Recommendation Engine",
            "Experiment 10: Word Sense Disambiguation (TensorFlow LSTM/GRU)"
        ]
    )

    # -------------------------------------------------------------
    # EXP 1 LAB
    # -------------------------------------------------------------
    if "Experiment 1" in exp_choice:
        st.subheader("Experiment 1: Sentiment Analysis of Library Reviews")
        st.markdown("**Objective:** Train TF-IDF + Classifier to evaluate reader reviews as Positive, Negative, or Neutral.")
        
        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            model_type = st.radio("Select Model Classifier:", ["logistic_regression", "naive_bayes"], format_func=lambda x: "Logistic Regression" if x=="logistic_regression" else "Multinomial Naive Bayes")
            test_size = st.slider("Test Split Ratio:", 0.1, 0.4, 0.25, step=0.05)
            if st.button("Re-train & Evaluate Model"):
                sentiment_analyzer.model_type = model_type
                sentiment_analyzer.train_and_evaluate(test_size=test_size)
                st.success("Model trained successfully!")

        with m_col2:
            metrics = sentiment_analyzer.evaluation_metrics or sentiment_analyzer.train_and_evaluate()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            c2.metric("Precision", f"{metrics['precision']*100:.1f}%")
            c3.metric("Recall", f"{metrics['recall']*100:.1f}%")
            c4.metric("F1 Score", f"{metrics['f1_score']*100:.1f}%")

            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=metrics["labels"], yticklabels=metrics["labels"], ax=ax)
            ax.set_ylabel("Actual Label")
            ax.set_xlabel("Predicted Label")
            st.pyplot(fig)

        st.markdown("---")
        st.write("### Test Custom Review")
        rev_input = st.text_input("Enter a book review to test:", value="This book is excellent and very informative.")
        if st.button("Classify Review"):
            pred = sentiment_analyzer.predict(rev_input)
            st.success(f"Sentiment: **{pred['sentiment']}** (Confidence: {pred['confidence']*100:.1f}%)")

    # -------------------------------------------------------------
    # EXP 2 LAB
    # -------------------------------------------------------------
    elif "Experiment 2" in exp_choice:
        st.subheader("Experiment 2: Tokenization, Filtration & Script Validation")
        st.markdown("**Objective:** Perform sentence tokenization, word tokenization, punctuation filtration, and English script validation.")
        
        test_text = st.text_area("Input Library Query:", value="Find science fiction books for beginners! Can you also find Harry Potter?")
        script_test = st.selectbox("Or choose a test query with unsupported scripts:", [
            "Default English Query",
            "Hindi (Devanagari): किताब खोजें",
            "Russian (Cyrillic): Найти книги",
            "Corrupted Unicode: Book \u2603 \u2602 test"
        ])
        if script_test != "Default English Query":
            test_text = script_test.split(": ")[-1]

        if st.button("Run Tokenization & Validation"):
            res = tokenize_and_filter(test_text)
            
            c_val, c_tok = st.columns([1, 2])
            with c_val:
                st.write("#### Script Validation Result")
                v = res["script_validation"]
                if v["is_valid"]:
                    st.success(f"✅ {v['warning_message']}")
                else:
                    st.error(f"❌ {v['warning_message']}")
                    st.json(v["unsupported_characters"])

            with c_tok:
                st.write("#### Tokenization Breakdown")
                st.write("**Sentence Tokens:**", res["sentence_tokens"])
                st.write("**Word Tokens:**", res["word_tokens"])
                st.write("**Filtered Clean Tokens:**", res["filtered_tokens"])

    # -------------------------------------------------------------
    # EXP 3 LAB
    # -------------------------------------------------------------
    elif "Experiment 3" in exp_choice:
        st.subheader("Experiment 3: Stop Word Removal, Stemming & Lemmatization")
        st.markdown("**Objective:** Remove high-frequency grammatical stopwords, apply Porter Stemming, and compute WordNet Lemmatization with POS context.")
        
        query_e3 = st.text_input("Enter Library Query:", value="Find the books about science and technology.")
        if st.button("Preprocess Text"):
            prep = preprocess_text(query_e3)
            st.write("**Stop Word Removed Tokens:**", prep["stopword_removed_tokens"])
            
            st.write("### Comparative Table (Original vs Stemmed vs Lemmatized)")
            df_c = pd.DataFrame(prep["comparison_table"])
            st.dataframe(df_c, use_container_width=True)

    # -------------------------------------------------------------
    # EXP 4 LAB
    # -------------------------------------------------------------
    elif "Experiment 4" in exp_choice:
        st.subheader("Experiment 4: Morphological Analysis & Word Generation")
        st.markdown("**Objective:** Decompose library words into root, prefix, and suffix, and generate inflectional/derivational forms for search expansion.")
        
        sample_word = st.selectbox("Select or enter a library word:", ["reading", "searches", "recommendation", "unreadable", "publisher", "educated", "borrowing"])
        custom_word = st.text_input("Or enter custom word:", value="")
        target = custom_word.strip() if custom_word.strip() else sample_word

        if st.button("Analyze Morphology"):
            m_info = analyze_morphology(target)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Base / Root Word", m_info["root"])
            c2.metric("Detected Prefix", m_info["prefix"])
            c3.metric("Detected Suffix", m_info["suffix"])

            st.write(f"### Generated Word Forms for '{m_info['root']}':")
            st.write(m_info["generated_forms"])

            st.markdown("---")
            st.write("### Search Query Expansion Demonstration:")
            demo_q = [target, "books"]
            expansion = expand_query_morphologically(demo_q)
            st.write(f"Original Query Tokens: `{demo_q}`")
            st.success(f"Morphologically Expanded Query: `{expansion['expanded_tokens']}`")

    # -------------------------------------------------------------
    # EXP 5 LAB
    # -------------------------------------------------------------
    elif "Experiment 5" in exp_choice:
        st.subheader("Experiment 5: N-Gram Language Model")
        st.markdown("Implement Unigram, Bigram, and Trigram language models with sentence boundary markers `<s>` and `</s>`, frequency counting, and next-word suggestions.")
        
        ngram_q = st.text_input("Enter Library Phrase:", value="find science fiction books")
        if st.button("Compute N-Grams"):
            res = ngram_model.analyze_query(ngram_q)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("### Unigrams (with count)")
                st.dataframe(pd.DataFrame(res["unigram_counts"]), use_container_width=True)
            with col2:
                st.write("### Bigrams (with count)")
                st.dataframe(pd.DataFrame(res["bigram_counts"]), use_container_width=True)
            with col3:
                st.write("### Trigrams (with count)")
                st.dataframe(pd.DataFrame(res["trigram_counts"]), use_container_width=True)

        st.markdown("---")
        st.write("### Interactive Autocomplete / Next-Word Prediction:")
        ctx_word = st.text_input("Enter context word(s) (e.g. 'science fiction' or 'harry'):", value="science fiction")
        if st.button("Predict Next Word"):
            words = ctx_word.strip().split()
            preds = ngram_model.suggest_next_word(words)
            if preds:
                for p in preds:
                    st.info(f"Predicted next word: **{p['word']}** (Frequency: {p['count']}, via {p['type']})")
            else:
                st.warning("No predictions found for this prefix.")

    # -------------------------------------------------------------
    # EXP 6 LAB
    # -------------------------------------------------------------
    elif "Experiment 6" in exp_choice:
        st.subheader("Experiment 6: Part-of-Speech (POS) Tagging")
        st.markdown("**Objective:** Tag each word in library queries with Penn Treebank POS and explanatory grammatical categories.")
        
        pos_q = st.text_input("Enter query for POS tagging:", value="Find interesting science books written by famous authors.")
        if st.button("Generate POS Tags"):
            tagged = tag_pos(pos_q)
            df_p = pd.DataFrame(tagged["structured_results"])
            st.dataframe(df_p, use_container_width=True)

    # -------------------------------------------------------------
    # EXP 7 LAB
    # -------------------------------------------------------------
    elif "Experiment 7" in exp_choice:
        st.subheader("Experiment 7: Chunking + Feature Selection + Training Size")
        st.markdown("**Objective:** Extract noun phrase semantic chunks and demonstrate empirical study on Feature Selection and Training Size (60%, 70%, 80%).")
        
        st.write("### 1. Phrase Chunking")
        chunk_q = st.text_input("Enter query for chunking:", value="Find interesting science fiction books for college beginners.")
        if st.button("Extract Noun Phrases"):
            c_res = extract_chunks(chunk_q)
            if c_res["extracted_phrases"]:
                for p in c_res["extracted_phrases"]:
                    st.success(f"Phrase: **{p['phrase']}** | Type: `{p['label']}`")
            st.text("Parse Tree:")
            st.code(c_res["parse_tree"])

        st.markdown("---")
        st.write("### 2. Empirical Analysis: Training Size & Feature Selection")
        if st.button("Run Training Size Study (60% vs 70% vs 80%)"):
            with st.spinner("Evaluating models across training sizes..."):
                study = analyze_training_size_and_features()
                df_study = pd.DataFrame(study["comparison_table"])
                st.dataframe(df_study, use_container_width=True)

                st.write("#### Performance Comparison Chart:")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=df_study, x="Training Size (%)", y="Accuracy", hue="Feature Set", ax=ax, palette="Set2")
                ax.set_ylim(0.85, 1.05)
                ax.set_title("Impact of Training Size & Feature Richness on Chunking Accuracy")
                st.pyplot(fig)

                st.info(study["explanation"])

    # -------------------------------------------------------------
    # EXP 8 LAB
    # -------------------------------------------------------------
    elif "Experiment 8" in exp_choice:
        st.subheader("Experiment 8: Named Entity Recognition (NER)")
        st.markdown("**Objective:** Extract library entities: `BOOK`, `AUTHOR`, `PUBLISHER`, `LOCATION`, `ORGANIZATION`, `DATE` using hybrid catalogue + NLTK NER.")
        
        ner_input = st.text_input("Enter sentence for NER extraction:", value="Find Harry Potter books written by J.K. Rowling published by Bloomsbury in London.")
        if st.button("Extract Entities"):
            res_ner = extract_named_entities(ner_input)
            if res_ner["entities"]:
                st.dataframe(pd.DataFrame(res_ner["entities"]), use_container_width=True)
            else:
                st.warning("No named entities detected.")

    # -------------------------------------------------------------
    # EXP 9 LAB
    # -------------------------------------------------------------
    elif "Experiment 9" in exp_choice:
        st.subheader("Experiment 9: Text Similarity & Book Recommendation Engine")
        st.markdown("**Objective:** Convert book descriptions and user queries into TF-IDF vectors and calculate Cosine Similarity for ranking.")
        
        sim_q = st.text_input("Enter recommendation query:", value="Suggest fantasy adventure books with wizards and dragons.")
        top_k = st.slider("Number of recommendations:", 1, 10, 5)
        if st.button("Compute Similarities & Recommend"):
            res = book_recommender.recommend_books(sim_q, top_k=top_k)
            st.write(f"**Indexed Query:** `{res['processed_query']}`")
            
            recs_df = pd.DataFrame(res["recommendations"])
            st.dataframe(recs_df[["book_id", "title", "author", "genre", "similarity_score", "match_percentage"]], use_container_width=True)

    # -------------------------------------------------------------
    # EXP 10 LAB
    # -------------------------------------------------------------
    elif "Experiment 10" in exp_choice:
        st.subheader("Experiment 10: Word Sense Disambiguation using LSTM / GRU")
        st.markdown("**Objective:** Disambiguate polysemous word `'novel'` (Meaning 1: `BOOK` vs Meaning 2: `NEW/ORIGINAL`) using deep learning LSTM/GRU.")
        
        wsd_sent = st.text_input("Enter sentence containing 'novel':", value="I borrowed an interesting novel from the college library.")
        col_w1, col_w2 = st.columns([1, 2])
        with col_w1:
            st.write("#### Quick Test Examples:")
            if st.button("📚 Novel = BOOK example"):
                wsd_sent = "The novel has an interesting story and memorable characters."
            if st.button("💡 Novel = NEW/ORIGINAL example"):
                wsd_sent = "The researcher proposed a novel approach to solve the problem."

        if st.button("Disambiguate Word Sense"):
            with st.spinner("Running Bidirectional LSTM neural inference..."):
                res = wsd_model.disambiguate(wsd_sent)
                st.success(f"Target Word: **{res['target_word']}** | Predicted Sense: **{res['predicted_sense']}**")
                st.metric("Neural Confidence", f"{res['confidence']*100:.1f}%")
                st.info(f"**Meaning:** {res['meaning_explanation']}")
                
                st.write("**Prediction Probabilities:**")
                st.bar_chart(pd.DataFrame(list(res["probabilities"].items()), columns=["Sense", "Probability"]).set_index("Sense"))

# ==============================================================================
# MODE 3: LIBRARY CATALOG & DATASETS EXPLORER
# ==============================================================================
elif app_mode == "📖 Library Catalog & Datasets":
    st.markdown('<div class="main-title">📖 Library Catalog & Datasets Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect Local Curated Datasets Supporting Experiments 1 Through 10</div>', unsafe_allow_html=True)

    tab_books, tab_reviews, tab_wsd = st.tabs(["📚 Book Catalog (`books.csv`)", "⭐ Reader Reviews (`library_reviews.csv`)", "🧠 WSD Sentences (`wsd_dataset.csv`)"])

    with tab_books:
        st.subheader("Library Book Catalog")
        df_b = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "books.csv"))
        st.write(f"Total Books: **{len(df_b)}** across 10 diverse genres.")
        
        genre_filter = st.multiselect("Filter by Genre:", sorted(df_b["genre"].unique()), default=[])
        if genre_filter:
            df_filtered = df_b[df_b["genre"].isin(genre_filter)]
        else:
            df_filtered = df_b
        st.dataframe(df_filtered, use_container_width=True)

    with tab_reviews:
        st.subheader("Reader Reviews & Sentiment Labels")
        df_r = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "library_reviews.csv"))
        st.write(f"Total Reviews: **{len(df_r)}**")
        st.dataframe(df_r, use_container_width=True)

    with tab_wsd:
        st.subheader("Word Sense Disambiguation Dataset (Polysemy: 'novel')")
        df_w = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "wsd_dataset.csv"))
        st.write(f"Total WSD Sentences: **{len(df_w)}**")
        st.dataframe(df_w, use_container_width=True)

"""
Comprehensive Test Suite for Library NLP Mini Project
Verifies all 10 NLP experiments and modules end-to-end.
"""

import sys
import os

# Ensure UTF-8 output in Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pipeline():
    print("=" * 70)
    print("VERIFYING ALL 10 NLP EXPERIMENTS FOR LIBRARY BOOK RECOMMENDATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # EXP 1: Sentiment Analysis
    # -------------------------------------------------------------
    print("\n[TEST 1/10] Experiment 1 — Sentiment Analysis...")
    from modules.sentiment import sentiment_analyzer
    metrics = sentiment_analyzer.train_and_evaluate()
    print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
    
    test_rev1 = "This book is excellent and very informative."
    pred1 = sentiment_analyzer.predict(test_rev1)
    print(f"  Review: '{test_rev1}' -> Sentiment: {pred1['sentiment']} (Conf: {pred1['confidence']:.2f})")
    assert pred1["sentiment"] == "Positive", f"Expected Positive, got {pred1['sentiment']}"

    test_rev2 = "The book is boring and difficult to understand."
    pred2 = sentiment_analyzer.predict(test_rev2)
    print(f"  Review: '{test_rev2}' -> Sentiment: {pred2['sentiment']} (Conf: {pred2['confidence']:.2f})")
    assert pred2["sentiment"] == "Negative", f"Expected Negative, got {pred2['sentiment']}"
    print("  -> Exp 1 PASSED.")

    # -------------------------------------------------------------
    # EXP 2 & 3: Tokenization, Filtration, Script Validation, Stemming & Lemmatization
    # -------------------------------------------------------------
    print("\n[TEST 2 & 3/10] Experiments 2 & 3 — Preprocessing, Script Check, Stemming & Lemmatization...")
    from modules.preprocessing import preprocess_text, validate_script
    sample_query = "Find science fiction books for beginners!"
    prep_res = preprocess_text(sample_query)
    print(f"  Original: '{prep_res['original_query']}'")
    print(f"  Words: {prep_res['word_tokens']}")
    print(f"  Filtered: {prep_res['filtered_tokens']}")
    print(f"  Stopwords Removed: {prep_res['stopword_removed_tokens']}")
    print(f"  Stemmed: {prep_res['stemmed_tokens']}")
    print(f"  Lemmatized: {prep_res['lemmatized_tokens']}")
    assert prep_res["script_validation"]["is_valid"] is True
    
    # Test non-English script validation
    hindi_check = validate_script("किताब खोजें")
    print(f"  Non-English Script Test: 'किताब खोजें' -> Valid: {hindi_check['is_valid']}, Script: {hindi_check['script']}")
    assert hindi_check["is_valid"] is False
    print("  -> Exp 2 & 3 PASSED.")

    # -------------------------------------------------------------
    # EXP 4: Morphological Analysis & Word Generation
    # -------------------------------------------------------------
    print("\n[TEST 4/10] Experiment 4 — Morphological Analysis & Word Generation...")
    from modules.morphology import analyze_morphology, expand_query_morphologically
    morph1 = analyze_morphology("reading")
    print(f"  Word: 'reading' -> Root: {morph1['root']}, Suffix: {morph1['suffix']}, Forms: {morph1['generated_forms']}")
    assert morph1["root"] == "read"

    expansion = expand_query_morphologically(["reading", "books"])
    print(f"  Query Expansion: ['reading', 'books'] -> {expansion['expanded_tokens']}")
    assert "read" in expansion["expanded_tokens"]
    print("  -> Exp 4 PASSED.")

    # -------------------------------------------------------------
    # EXP 5: N-Gram Model
    # -------------------------------------------------------------
    print("\n[TEST 5/10] Experiment 5 — N-Gram Model & Next-Word Prediction...")
    from modules.ngram import ngram_model
    ngram_res = ngram_model.analyze_query("find science fiction books")
    print(f"  Unigrams: {ngram_res['unigrams_raw']}")
    print(f"  Bigrams: {ngram_res['bigrams_raw']}")
    print(f"  Trigrams: {ngram_res['trigrams_raw']}")
    suggestions = ngram_model.suggest_next_word(["science", "fiction"])
    print(f"  Suggestion for ['science', 'fiction']: {[s['word'] for s in suggestions]}")
    assert len(ngram_res["bigrams_raw"]) == 3
    print("  -> Exp 5 PASSED.")

    # -------------------------------------------------------------
    # EXP 6: POS Tagging
    # -------------------------------------------------------------
    print("\n[TEST 6/10] Experiment 6 — POS Tagging...")
    from modules.pos_tagging import tag_pos
    pos_res = tag_pos("Find interesting science books")
    for item in pos_res["structured_results"]:
        print(f"  {item['word']:12} -> {item['tag']:5} ({item['category']})")
    assert any(item["category"] in ["Noun", "Adjective", "Verb"] for item in pos_res["structured_results"])
    print("  -> Exp 6 PASSED.")

    # -------------------------------------------------------------
    # EXP 7: Chunking + Feature Selection + Training Size
    # -------------------------------------------------------------
    print("\n[TEST 7/10] Experiment 7 — Chunking & Training Size Empirical Study...")
    from modules.chunking import extract_chunks, analyze_training_size_and_features
    chunk_res = extract_chunks("Find interesting science fiction books")
    print(f"  Extracted Phrases: {[p['phrase'] for p in chunk_res['extracted_phrases']]}")
    assert any("science fiction books" in p["phrase"] for p in chunk_res["extracted_phrases"])

    study = analyze_training_size_and_features()
    print("  Training Size Study Results:")
    for row in study["comparison_table"]:
        print(f"    {row['Feature Set'][:15]:15} | Size: {row['Training Size (%)']:4} | Acc: {row['Accuracy']:.4f} | F1: {row['F1 Score']:.4f}")
    print("  -> Exp 7 PASSED.")

    # -------------------------------------------------------------
    # EXP 8: Named Entity Recognition
    # -------------------------------------------------------------
    print("\n[TEST 8/10] Experiment 8 — Named Entity Recognition (NER)...")
    from modules.ner import extract_named_entities
    ner_res = extract_named_entities("Find Harry Potter books written by J.K. Rowling in Pune.")
    for ent in ner_res["entities"]:
        print(f"  Entity: '{ent['entity']}' -> Label: {ent['label']} (Source: {ent['source']})")
    labels_found = [e["label"] for e in ner_res["entities"]]
    assert "BOOK" in labels_found or "AUTHOR" in labels_found
    print("  -> Exp 8 PASSED.")

    # -------------------------------------------------------------
    # EXP 9: Text Similarity & Recommendation
    # -------------------------------------------------------------
    print("\n[TEST 9/10] Experiment 9 — Text Similarity & Recommendation...")
    from modules.similarity import book_recommender
    rec_res = book_recommender.recommend_books("Suggest fantasy adventure books with wizards", top_k=3)
    print("  Top Recommendations:")
    for idx, b in enumerate(rec_res["recommendations"], 1):
        print(f"    {idx}. {b['title']} by {b['author']} (Score: {b['similarity_score']:.4f})")
    assert len(rec_res["recommendations"]) > 0
    print("  -> Exp 9 PASSED.")

    # -------------------------------------------------------------
    # EXP 10: Word Sense Disambiguation using LSTM
    # -------------------------------------------------------------
    print("\n[TEST 10/10] Experiment 10 — Word Sense Disambiguation using LSTM...")
    from modules.wsd import wsd_model
    wsd_metrics = wsd_model.train_and_evaluate(epochs=20)
    print(f"  LSTM Trained | Val Acc: {wsd_metrics['final_val_acc']:.4f} | Test Acc: {wsd_metrics['test_accuracy']:.4f}")

    wsd_test1 = "I borrowed a novel from the library."
    res1 = wsd_model.disambiguate(wsd_test1)
    print(f"  Sentence: '{wsd_test1}' -> Sense: {res1['predicted_sense']} (Conf: {res1['confidence']:.2f})")

    wsd_test2 = "The researcher proposed a novel method to solve the equation."
    res2 = wsd_model.disambiguate(wsd_test2)
    print(f"  Sentence: '{wsd_test2}' -> Sense: {res2['predicted_sense']} (Conf: {res2['confidence']:.2f})")
    
    assert res1["predicted_sense"] == "BOOK", f"Expected BOOK, got {res1['predicted_sense']}"
    assert res2["predicted_sense"] == "NEW", f"Expected NEW, got {res2['predicted_sense']}"
    print("  -> Exp 10 PASSED.")

    print("\n" + "=" * 70)
    print("ALL 10 NLP EXPERIMENTS VERIFIED SUCCESSFULLY AND WORKING TOGETHER!")
    print("=" * 70)

if __name__ == "__main__":
    test_pipeline()

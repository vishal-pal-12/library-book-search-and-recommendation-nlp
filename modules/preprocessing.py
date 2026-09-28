"""
Experiments 2 & 3: Preprocessing Pipeline
- Exp 2: Tokenization, Filtration & Script Validation
- Exp 3: Stop Word Removal, Porter Stemming & WordNet Lemmatization
"""

import re
import string
import unicodedata
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# Ensure required NLTK resources
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}" if resource in ["stopwords", "wordnet"] else f"taggers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def validate_script(text):
    """
    Validates if the text conforms to standard English (Latin/ASCII) script.
    Detects and flags unsupported scripts (e.g. Devanagari, Cyrillic, Arabic, CJK, etc.)
    or unusual non-ASCII characters.
    """
    unsupported_chars = []
    scripts_detected = set()

    for char in text:
        # Allow common ascii, whitespace, and punctuation
        if ord(char) < 128:
            continue
        
        name = unicodedata.name(char, "UNKNOWN")
        script = name.split()[0] if name != "UNKNOWN" else "UNKNOWN"
        scripts_detected.add(script)
        unsupported_chars.append({"char": char, "unicode": f"U+{ord(char):04X}", "name": name, "script": script})

    is_valid = len(unsupported_chars) == 0
    return {
        "is_valid": is_valid,
        "script": "Latin / English" if is_valid else f"Unsupported ({', '.join(scripts_detected)})",
        "unsupported_characters": unsupported_chars,
        "warning_message": "All characters belong to valid English script." if is_valid else f"Found {len(unsupported_chars)} non-English script character(s): {[c['char'] for c in unsupported_chars]}"
    }

def tokenize_and_filter(text):
    """
    Experiment 2: Sentence Tokenization, Word Tokenization, Filtration, and Script Validation.
    """
    script_check = validate_script(text)
    
    # 1. Sentence Tokenization
    sentences = sent_tokenize(text)
    
    # 2. Word Tokenization
    raw_word_tokens = word_tokenize(text)
    
    # 3. Filtration: lowercase, remove standalone punctuation and non-alphanumeric noise
    filtered_tokens = [
        token.lower() for token in raw_word_tokens
        if token.lower() not in string.punctuation and re.search(r"\w", token)
    ]

    return {
        "original_query": text,
        "sentence_tokens": sentences,
        "word_tokens": raw_word_tokens,
        "filtered_tokens": filtered_tokens,
        "script_validation": script_check
    }

def get_wordnet_pos(treebank_tag):
    """Maps Penn Treebank POS tag to WordNet POS tag for accurate lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    """
    Experiment 3: Stop Word Removal, Stemming, and Lemmatization.
    Connects with Experiment 2 to provide comparative tokens.
    """
    exp2_res = tokenize_and_filter(text)
    filtered_tokens = exp2_res["filtered_tokens"]

    # 1. Stop word removal
    stopword_removed_tokens = [w for w in filtered_tokens if w not in stop_words]

    # 2. Porter Stemming
    stemmed_tokens = [stemmer.stem(w) for w in stopword_removed_tokens]

    # 3. WordNet Lemmatization with POS awareness
    tagged = pos_tag(stopword_removed_tokens)
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged
    ]

    comparison_table = []
    for orig, stem, lemma in zip(stopword_removed_tokens, stemmed_tokens, lemmatized_tokens):
        comparison_table.append({
            "token": orig,
            "stemmed": stem,
            "lemmatized": lemma
        })

    return {
        **exp2_res,
        "stopword_removed_tokens": stopword_removed_tokens,
        "stemmed_tokens": stemmed_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "comparison_table": comparison_table
    }

"""
Experiment 6: Part-of-Speech (POS) Tagging
Extracts grammatical categories using NLTK Perceptron Tagger,
maps tags to clear human-readable descriptions, and formats tokens for Chunking.
"""

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Human-readable mapping for Penn Treebank POS tags
PENN_TAG_DESCRIPTIONS = {
    "CC": ("Coordinating Conjunction", "Conjunction"),
    "CD": ("Cardinal Digit", "Number"),
    "DT": ("Determiner", "Determiner"),
    "EX": ("Existential There", "Pronoun"),
    "FW": ("Foreign Word", "Other"),
    "IN": ("Preposition / Subordinating Conjunction", "Preposition"),
    "JJ": ("Adjective", "Adjective"),
    "JJR": ("Adjective, Comparative", "Adjective"),
    "JJS": ("Adjective, Superlative", "Adjective"),
    "LS": ("List Item Marker", "Other"),
    "MD": ("Modal", "Verb / Modal"),
    "NN": ("Noun, Singular", "Noun"),
    "NNS": ("Noun, Plural", "Noun"),
    "NNP": ("Proper Noun, Singular", "Noun / Proper Noun"),
    "NNPS": ("Proper Noun, Plural", "Noun / Proper Noun"),
    "PDT": ("Predeterminer", "Determiner"),
    "POS": ("Possessive Ending", "Grammatical Marker"),
    "PRP": ("Personal Pronoun", "Pronoun"),
    "PRP$": ("Possessive Pronoun", "Pronoun"),
    "RB": ("Adverb", "Adverb"),
    "RBR": ("Adverb, Comparative", "Adverb"),
    "RBS": ("Adverb, Superlative", "Adverb"),
    "RP": ("Particle", "Particle"),
    "TO": ("to", "Preposition"),
    "UH": ("Interjection", "Interjection"),
    "VB": ("Verb, Base Form", "Verb"),
    "VBD": ("Verb, Past Tense", "Verb"),
    "VBG": ("Verb, Gerund / Present Participle", "Verb / Modifier"),
    "VBN": ("Verb, Past Participle", "Verb"),
    "VBP": ("Verb, Non-3rd Person Singular Present", "Verb"),
    "VBZ": ("Verb, 3rd Person Singular Present", "Verb"),
    "WDT": ("Wh-determiner", "Determiner"),
    "WP": ("Wh-pronoun", "Pronoun"),
    "WP$": ("Possessive Wh-pronoun", "Pronoun"),
    "WRB": ("Wh-adverb", "Adverb")
}

def tag_pos(text_or_tokens):
    """
    Tags tokens with Part-of-Speech and returns structured grammatical metadata.
    """
    if isinstance(text_or_tokens, str):
        tokens = word_tokenize(text_or_tokens)
    else:
        tokens = text_or_tokens

    raw_tagged = pos_tag(tokens)
    structured_results = []

    for word, tag in raw_tagged:
        desc, category = PENN_TAG_DESCRIPTIONS.get(tag, ("Punctuation / Symbol", "Symbol"))
        structured_results.append({
            "word": word,
            "tag": tag,
            "category": category,
            "description": desc
        })

    return {
        "tagged_tuples": raw_tagged,
        "structured_results": structured_results
    }

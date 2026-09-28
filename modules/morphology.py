"""
Experiment 4: Morphological Analysis & Word Generation
Performs morphological decomposition (root, prefix, suffix),
generates inflectional and derivational word forms,
and provides query expansion for library search.
"""

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

# Common English prefixes and suffixes
COMMON_PREFIXES = [
    "un", "re", "pre", "dis", "mis", "non", "in", "im", "over", "under", "sub"
]

COMMON_SUFFIXES = [
    "ing", "ed", "es", "s", "er", "or", "able", "ible", "tion", "sion", "ment",
    "ness", "al", "ful", "less", "ly", "ist", "ism", "ize", "ise"
]

# Library domain morphological rules and generation dictionary
LIBRARY_MORPHOLOGY_RULES = {
    "read": {
        "root": "read",
        "forms": ["read", "reads", "reading", "reader", "readable", "readability", "unread"],
        "category": "verb/noun"
    },
    "search": {
        "root": "search",
        "forms": ["search", "searches", "searched", "searching", "searcher", "searchable", "research"],
        "category": "verb/noun"
    },
    "recommend": {
        "root": "recommend",
        "forms": ["recommend", "recommends", "recommended", "recommending", "recommendation", "recommendable"],
        "category": "verb"
    },
    "write": {
        "root": "write",
        "forms": ["write", "writes", "writing", "writer", "written", "rewrite"],
        "category": "verb"
    },
    "publish": {
        "root": "publish",
        "forms": ["publish", "publishes", "published", "publishing", "publisher", "publication", "unpublished"],
        "category": "verb"
    },
    "learn": {
        "root": "learn",
        "forms": ["learn", "learns", "learning", "learned", "learner", "learnable", "unlearn"],
        "category": "verb"
    },
    "borrow": {
        "root": "borrow",
        "forms": ["borrow", "borrows", "borrowed", "borrowing", "borrower"],
        "category": "verb"
    },
    "catalogue": {
        "root": "catalogue",
        "forms": ["catalogue", "catalogues", "catalogued", "cataloguing", "catalog"],
        "category": "noun/verb"
    },
    "collect": {
        "root": "collect",
        "forms": ["collect", "collects", "collected", "collecting", "collection", "collector"],
        "category": "verb"
    },
    "educate": {
        "root": "educate",
        "forms": ["educate", "educates", "educated", "educating", "education", "educator", "educational"],
        "category": "verb"
    },
    "novel": {
        "root": "novel",
        "forms": ["novel", "novels", "novelist", "novelty"],
        "category": "noun/adjective"
    },
    "author": {
        "root": "author",
        "forms": ["author", "authors", "authored", "authoring", "authorship", "coauthor"],
        "category": "noun/verb"
    }
}

def analyze_morphology(word):
    """
    Deconstructs a word into root, prefix, suffix, and identifies affix types.
    """
    word_clean = word.lower().strip()
    
    # Check domain catalog first
    for base, data in LIBRARY_MORPHOLOGY_RULES.items():
        if word_clean in data["forms"] or word_clean == base:
            # Determine prefix and suffix relative to base
            prefix = ""
            suffix = ""
            for p in sorted(COMMON_PREFIXES, key=len, reverse=True):
                if word_clean.startswith(p) and len(word_clean) > len(p) + 2:
                    prefix = p
                    break
            for s in sorted(COMMON_SUFFIXES, key=len, reverse=True):
                if word_clean.endswith(s) and len(word_clean) > len(s) + 2:
                    suffix = s
                    break

            return {
                "word": word,
                "root": base,
                "prefix": prefix if prefix else "None",
                "suffix": suffix if suffix else "None",
                "generated_forms": data["forms"],
                "category": data["category"]
            }

    # General algorithmic morphological breakdown
    detected_prefix = "None"
    detected_suffix = "None"
    stem_candidate = word_clean

    for p in sorted(COMMON_PREFIXES, key=len, reverse=True):
        if stem_candidate.startswith(p) and len(stem_candidate) > len(p) + 2:
            detected_prefix = p
            stem_candidate = stem_candidate[len(p):]
            break

    for s in sorted(COMMON_SUFFIXES, key=len, reverse=True):
        if stem_candidate.endswith(s) and len(stem_candidate) > len(s) + 2:
            detected_suffix = s
            stem_candidate = stem_candidate[:-len(s)]
            break

    # Lemmatize stem candidate
    root_verb = lemmatizer.lemmatize(stem_candidate, pos=wordnet.VERB)
    root_noun = lemmatizer.lemmatize(root_verb, pos=wordnet.NOUN)
    root = root_noun

    # Generate standard English inflections algorithmically
    generated_forms = generate_word_forms(root)

    return {
        "word": word,
        "root": root,
        "prefix": detected_prefix,
        "suffix": detected_suffix,
        "generated_forms": generated_forms,
        "category": "general"
    }

def generate_word_forms(root):
    """
    Generates regular inflectional forms for any given root word.
    """
    root = root.lower().strip()
    forms = set([root])
    
    # Plural / 3rd person singular
    if root.endswith(("s", "sh", "ch", "x", "z")):
        forms.add(root + "es")
    elif root.endswith("y") and len(root) > 1 and root[-2] not in "aeiou":
        forms.add(root[:-1] + "ies")
    else:
        forms.add(root + "s")

    # Present Participle / Gerund (-ing)
    if root.endswith("e") and not root.endswith(("ee", "oe")):
        forms.add(root[:-1] + "ing")
    elif len(root) >= 3 and root[-1] not in "aeiouwxy" and root[-2] in "aeiou" and root[-3] not in "aeiou":
        forms.add(root + root[-1] + "ing")
    else:
        forms.add(root + "ing")

    # Past Tense (-ed)
    if root.endswith("e"):
        forms.add(root + "d")
    elif root.endswith("y") and len(root) > 1 and root[-2] not in "aeiou":
        forms.add(root[:-1] + "ied")
    elif len(root) >= 3 and root[-1] not in "aeiouwxy" and root[-2] in "aeiou" and root[-3] not in "aeiou":
        forms.add(root + root[-1] + "ed")
    else:
        forms.add(root + "ed")

    # Agent noun (-er)
    if root.endswith("e"):
        forms.add(root + "r")
    else:
        forms.add(root + "er")

    return sorted(list(forms))

def expand_query_morphologically(query_tokens):
    """
    Expands query tokens to include their base roots and forms,
    improving search recall in the Library Recommendation engine.
    """
    expanded_set = set(query_tokens)
    expansion_details = {}

    for token in query_tokens:
        morph_info = analyze_morphology(token)
        root = morph_info["root"]
        forms = morph_info["generated_forms"]
        
        # Add root and closely related forms
        expanded_set.add(root)
        for form in forms[:3]:  # Top 3 variations
            expanded_set.add(form)

        expansion_details[token] = {
            "root": root,
            "expanded_to": [root] + [f for f in forms if f != token][:3]
        }

    return {
        "original_tokens": query_tokens,
        "expanded_tokens": sorted(list(expanded_set)),
        "expansion_details": expansion_details
    }

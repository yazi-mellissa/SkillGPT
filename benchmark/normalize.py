"""Turning three different response shapes into one comparable set of skill strings.

This module is the reason the benchmark is possible at all. The three backends do not
agree on what a response looks like:

    /search          a Milvus hit list, nested, each hit carrying an `entity` with
                     `expertise` and `text` fields
    /search_openai   a flat list of strings, comma-split from the model's reply
    /search_gemini   the same, but the model phrases things differently

and they do not agree on how a skill is written either: "Node.js", "NodeJS" and
"node js" are one skill. Scoring without normalising first measures formatting, not
extraction.
"""

import re
import unicodedata

# Aliases that the models and the index genuinely disagree on. Kept deliberately
# small: every entry here is a judgement that makes the scores look better, so it has
# to be defensible. Add to it only for spellings of the same skill, never for skills
# that are merely related.
ALIASES = {
    "nodejs": "node",
    "node js": "node",
    "node.js": "node",
    "js": "javascript",
    "ecmascript": "javascript",
    "ts": "typescript",
    "py": "python",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "rest api": "rest",
    "restful": "rest",
    "aws cloud": "aws",
    "amazon web services": "aws",
    "gcp": "google cloud",
    "ms sql": "sql server",
    "c sharp": "c#",
    "golang": "go",
}

_PUNCTUATION = re.compile(r"[^\w\s#+./-]")
_WHITESPACE = re.compile(r"\s+")


def normalize_skill(raw: str) -> str:
    """Lowercase, strip accents and punctuation, then apply the alias table."""
    if not raw:
        return ""
    text = unicodedata.normalize("NFKD", str(raw))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = _PUNCTUATION.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip(" .-/")
    return ALIASES.get(text, text)


def normalize_set(skills) -> set:
    """Normalise an iterable of skills and drop whatever normalises to nothing."""
    return {s for s in (normalize_skill(x) for x in skills) if s}


def extract_vector_skills(payload) -> list:
    """Pull skill strings out of whatever `/search` returned.

    pymilvus has changed this shape more than once, so the function accepts the
    nested form, the flat form, and plain strings, rather than assuming one.
    """
    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]

    skills = []
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, dict):
            entity = item.get("entity", item)
            if isinstance(entity, dict):
                value = entity.get("expertise") or entity.get("text")
                if value:
                    skills.append(value)
                    continue
            stack.extend(item.values())
        elif isinstance(item, str):
            skills.append(item)
    return skills


def extract_llm_skills(payload) -> list:
    """`/search_openai` and `/search_gemini` return a list of strings.

    A model occasionally answers with one string holding the whole comma-separated
    list, so that case is split here rather than counted as a single wrong skill.
    """
    if payload is None:
        return []
    if isinstance(payload, str):
        payload = [payload]
    skills = []
    for item in payload:
        if isinstance(item, str):
            skills.extend(part for part in item.split(",") if part.strip())
        else:
            skills.append(str(item))
    return skills

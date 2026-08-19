"""
Entity Type Description Conditioning for GLiNER.

Allows users to pass full natural-language definitions alongside entity type labels.
Validated by IBM ZeroNER (ACL 2025) and OpenBioNER (NAACL 2025) to yield +10-16% F1
on rare and novel entity types compared to short-label baselines.

Usage:
    # Option 1 — short labels (existing, unchanged)
    labels = ["person", "organization", "location"]

    # Option 2 — list of description dicts (new)
    labels = [
        {"label": "person",       "description": "a named human individual"},
        {"label": "organization", "description": "a legally incorporated company or institution"},
        {"label": "location",     "description": "a named geographical place or physical location"},
    ]

    # Option 3 — dict mapping label → description (new)
    labels = {
        "person":       "a named human individual",
        "organization": "a legally incorporated company or institution",
        "location":     "a named geographical place or physical location",
    }

    # Option 4 — use a built-in curated library
    from gliner.descriptions import ONTONOTES_DESCRIPTIONS
    entities = model.predict_entities(text, ONTONOTES_DESCRIPTIONS)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union, Optional

# ---------------------------------------------------------------------------
# Core normalisation utility
# ---------------------------------------------------------------------------

LabelInput = Union[
    List[str],
    List[Dict[str, str]],
    Dict[str, str],
]


def normalise_labels(
    labels: LabelInput,
    description_sep: str = ": ",
    max_description_length: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    Parse any supported label format and return (display_names, prompt_strings).

    display_names   — what appears as entity["label"] in the model output
    prompt_strings  — what is tokenised and encoded as the entity type sequence

    When no description is provided for a label, prompt_string == display_name
    (backward-compatible with existing code).

    Args:
        labels: One of:
            - List[str]: ["person", "organization"]  (no change)
            - List[Dict]: [{"label": "person", "description": "a named human individual"}, ...]
            - Dict[str, str]: {"person": "a named human individual", ...}
        description_sep: Separator between label and description in prompt string.
            ": " (default) validated by ZeroNER paper for DeBERTa-family models.
        max_description_length: If set, descriptions are truncated to this many characters
            before being concatenated. None = no truncation.

    Returns:
        Tuple of (display_names, prompt_strings), both List[str] of equal length.

    Examples:
        >>> normalise_labels(["person", "org"])
        (['person', 'org'], ['person', 'org'])

        >>> normalise_labels({"person": "a human individual", "org": "a company"})
        (['person', 'org'], ['person: a human individual', 'org: a company'])

        >>> normalise_labels([{"label": "person", "description": "a human individual"}])
        (['person'], ['person: a human individual'])
    """
    if isinstance(labels, dict):
        items = list(labels.items())
        display_names = [k for k, _ in items]
        descriptions  = [v for _, v in items]
    elif isinstance(labels, list):
        if not labels:
            return [], []
        if isinstance(labels[0], dict):
            display_names = [e["label"] for e in labels]
            descriptions  = [e.get("description") for e in labels]
        else:
            # Plain list of strings — no descriptions
            display_names = list(labels)
            descriptions  = [None] * len(labels)
    else:
        raise TypeError(
            f"labels must be List[str], List[Dict[str, str]], or Dict[str, str]. "
            f"Got {type(labels).__name__}."
        )

    prompt_strings = []
    for name, desc in zip(display_names, descriptions):
        if desc:
            effective = desc[:max_description_length] if max_description_length is not None else desc
            prompt_strings.append(f"{name}{description_sep}{effective}")
        else:
            prompt_strings.append(name)

    return display_names, prompt_strings


def remap_entity_labels(
    entities: List[Dict],
    prompt_to_display: Dict[str, str],
) -> List[Dict]:
    """
    Replace entity["label"] prompt strings with their display names.

    Called after prediction when descriptions were used as prompts.
    Operates in-place for efficiency; also returns the list.
    """
    for entity in entities:
        label = entity.get("label")
        if label in prompt_to_display:
            entity["label"] = prompt_to_display[label]
    return entities


# ---------------------------------------------------------------------------
# Built-in curated description libraries
# ---------------------------------------------------------------------------

# OntoNotes 18 entity types — definitions sourced from the OntoNotes guidelines
# and refined following the ZeroNER paper's appendix conventions.
ONTONOTES_DESCRIPTIONS: Dict[str, str] = {
    "PERSON":       "people, including fictional characters",
    "NORP":         "nationalities, religious groups, or political groups",
    "FAC":          "buildings, airports, highways, bridges, and other man-made structures",
    "ORG":          "companies, agencies, institutions, and other organisations",
    "GPE":          "countries, cities, states, and other geopolitical entities",
    "LOC":          "non-GPE locations: mountain ranges, bodies of water, regions",
    "PRODUCT":      "vehicles, weapons, foods, and other physical products (not services)",
    "EVENT":        "named hurricanes, battles, wars, sports events, and other events",
    "WORK_OF_ART":  "titles of books, songs, films, artworks, and similar creative works",
    "LAW":          "named laws, acts, or legal documents",
    "LANGUAGE":     "any named language",
    "DATE":         "absolute or relative dates or periods",
    "TIME":         "times smaller than a day",
    "PERCENT":      "percentage values, including the percent sign",
    "MONEY":        "monetary values, including the currency unit",
    "QUANTITY":     "measurements of weight, distance, speed, temperature, or similar",
    "ORDINAL":      "first, second, third, and other ordinal numbers",
    "CARDINAL":     "numerals that do not fall under another type",
}

# CoNLL-2003 4-type schema
CONLL_DESCRIPTIONS: Dict[str, str] = {
    "PER":  "a named person or family",
    "ORG":  "a named organisation, company, agency, or institution",
    "LOC":  "a named geographical location that is not a geopolitical entity",
    "MISC": "miscellaneous named entity that does not fit PER, ORG, or LOC",
}

# WNUT-17 6-type schema (emerging / novel entities)
WNUT_DESCRIPTIONS: Dict[str, str] = {
    "person":         "a named real or fictional individual person",
    "location":       "a named geographical place, region, or physical location",
    "corporation":    "a named company, business, or commercial organisation",
    "product":        "a named physical product, including software and services",
    "creative-work":  "a title of a creative work such as a book, film, song, or game",
    "group":          "a named group of people that is not a corporation, e.g. a band or sports team",
}

# Common biomedical types (drawn from OpenBioNER and BC5CDR)
BIOMEDICAL_DESCRIPTIONS: Dict[str, str] = {
    "disease":          "a named disease, disorder, or medical condition",
    "chemical":         "a named chemical compound, drug, or pharmaceutical substance",
    "gene":             "a named gene, protein, or gene product",
    "species":          "a named biological species, organism, or taxon",
    "mutation":         "a named genetic variant, SNP, or mutation",
    "cell_line":        "a named cell line used in biological research",
    "cell_type":        "a named type of biological cell",
    "DNA":              "a named DNA sequence, region, or domain",
    "RNA":              "a named RNA molecule or sequence",
}

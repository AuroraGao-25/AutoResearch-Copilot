from __future__ import annotations

from dataclasses import dataclass


POS_ADJ = "good"
NEG_ADJ = "bad"
NEU_ADJ = "okay"


@dataclass(frozen=True)
class MetamorphicPair:
    category: str
    relation_type: str
    source_text: str
    followup_text: str
    source_sentiment: str
    followup_sentiment: str
    aspect: str


def _normalize_sentiment(sentiment: str) -> str:
    value = sentiment.strip().lower()
    if value not in {"positive", "negative", "neutral"}:
        raise ValueError(f"Unsupported sentiment '{sentiment}'")
    return value


def _aspect_phrase(aspect: str) -> str:
    clean = aspect.strip().lower()
    if not clean:
        raise ValueError("Aspect cannot be empty")
    return clean


def make_negation_pair(aspect: str, sentiment: str) -> MetamorphicPair | None:
    source_sentiment = _normalize_sentiment(sentiment)
    if source_sentiment == "neutral":
        return None

    aspect_phrase = _aspect_phrase(aspect)
    if source_sentiment == "positive":
        src_adj, fup_adj = POS_ADJ, POS_ADJ
        followup_sentiment = "negative"
    else:
        src_adj, fup_adj = NEG_ADJ, NEG_ADJ
        followup_sentiment = "positive"

    source_text = f"The {aspect_phrase} is {src_adj}."
    followup_text = f"The {aspect_phrase} is not {fup_adj}."

    return MetamorphicPair(
        category="negation",
        relation_type="flip",
        source_text=source_text,
        followup_text=followup_text,
        source_sentiment=source_sentiment,
        followup_sentiment=followup_sentiment,
        aspect=aspect_phrase,
    )


def make_contrast_pair(aspect: str, sentiment: str) -> MetamorphicPair:
    source_sentiment = _normalize_sentiment(sentiment)
    aspect_phrase = _aspect_phrase(aspect)

    if source_sentiment == "positive":
        target_adj = POS_ADJ
        distractor_adj = NEG_ADJ
    elif source_sentiment == "negative":
        target_adj = NEG_ADJ
        distractor_adj = POS_ADJ
    else:
        target_adj = NEU_ADJ
        distractor_adj = POS_ADJ

    source_text = f"The {aspect_phrase} is {target_adj}."
    followup_text = (
        f"The ambience is {distractor_adj}, but the {aspect_phrase} is {target_adj}."
    )

    return MetamorphicPair(
        category="contrast",
        relation_type="invariant",
        source_text=source_text,
        followup_text=followup_text,
        source_sentiment=source_sentiment,
        followup_sentiment=source_sentiment,
        aspect=aspect_phrase,
    )


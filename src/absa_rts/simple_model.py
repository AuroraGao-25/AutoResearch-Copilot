from __future__ import annotations


def predict_sentiment(text: str) -> str:
    lowered = f" {text.lower()} "
    has_not = " not " in lowered
    has_but = " but " in lowered

    positive_hits = sum(token in lowered for token in (" good", " great", " excellent"))
    negative_hits = sum(token in lowered for token in (" bad", " terrible", " poor"))

    if has_but:
        clause = lowered.split(" but ")[-1]
        positive_hits = sum(token in clause for token in (" good", " great", " excellent"))
        negative_hits = sum(token in clause for token in (" bad", " terrible", " poor"))
        has_not = " not " in clause

    if positive_hits == negative_hits:
        return "neutral"

    sentiment = "positive" if positive_hits > negative_hits else "negative"
    if has_not and sentiment != "neutral":
        return "negative" if sentiment == "positive" else "positive"
    return sentiment


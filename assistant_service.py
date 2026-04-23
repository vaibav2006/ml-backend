from __future__ import annotations

"""Dynamic assistant helper for the rainfall web app.

This module enriches chatbot answers with free public sources (DuckDuckGo and Wikipedia)
and then combines those snippets with project-specific farm guidance.
"""

import json
import re
from typing import Any
from urllib import parse, request


USER_AGENT = "V2S-Rainfall-Assistant/1.0"


def _fetch_json(url: str, timeout: float = 6.0) -> dict[str, Any] | list[Any] | None:
    """Fetch JSON from a URL and return `None` on any network/parsing failure."""
    req = request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                return None
            raw = response.read().decode("utf-8", errors="ignore")
            return json.loads(raw)
    except Exception:
        return None


def _normalize_sentence(text: str, max_len: int = 460) -> str:
    """Normalize whitespace and trim long sentences to keep responses readable."""
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."


def _extract_duckduckgo_related(items: list[Any], limit: int = 2) -> list[str]:
    """Collect short snippets from DuckDuckGo `RelatedTopics` recursively."""
    snippets: list[str] = []
    for item in items:
        if len(snippets) >= limit:
            break
        if isinstance(item, dict) and isinstance(item.get("Text"), str):
            snippets.append(_normalize_sentence(item["Text"], max_len=220))
            continue
        if isinstance(item, dict) and isinstance(item.get("Topics"), list):
            nested = _extract_duckduckgo_related(item["Topics"], limit=limit - len(snippets))
            snippets.extend(nested)
    return snippets[:limit]


def fetch_duckduckgo_context(question: str) -> dict[str, Any] | None:
    """Query DuckDuckGo instant answer API and return compact context snippets."""
    query = parse.quote(question)
    url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
    payload = _fetch_json(url)
    if not isinstance(payload, dict):
        return None

    snippets: list[str] = []
    answer = payload.get("Answer")
    abstract = payload.get("AbstractText")
    if isinstance(answer, str) and answer.strip():
        snippets.append(_normalize_sentence(answer))
    if isinstance(abstract, str) and abstract.strip():
        snippets.append(_normalize_sentence(abstract))

    related = payload.get("RelatedTopics")
    if isinstance(related, list):
        snippets.extend(_extract_duckduckgo_related(related))

    snippets = [item for item in snippets if item]
    if not snippets:
        return None

    source = payload.get("AbstractURL")
    if not isinstance(source, str) or not source.strip():
        source = "https://duckduckgo.com/"

    return {
        "summary": snippets[0],
        "extra": snippets[1:3],
        "source": source
    }


def fetch_wikipedia_context(question: str) -> dict[str, Any] | None:
    """Fetch a concise Wikipedia summary for the user's question."""
    query = parse.quote(question)
    search_url = (
        "https://en.wikipedia.org/w/api.php?"
        f"action=opensearch&search={query}&limit=1&namespace=0&format=json"
    )
    search_payload = _fetch_json(search_url)
    if not isinstance(search_payload, list) or len(search_payload) < 2:
        return None
    if not isinstance(search_payload[1], list) or not search_payload[1]:
        return None

    title = str(search_payload[1][0]).strip()
    if not title:
        return None

    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{parse.quote(title)}"
    summary_payload = _fetch_json(summary_url)
    if not isinstance(summary_payload, dict):
        return None

    extract = summary_payload.get("extract")
    if not isinstance(extract, str) or not extract.strip():
        return None

    page_url = (
        summary_payload.get("content_urls", {})
        .get("desktop", {})
        .get("page")
    )
    if not isinstance(page_url, str) or not page_url.strip():
        page_url = f"https://en.wikipedia.org/wiki/{parse.quote(title.replace(' ', '_'))}"

    return {
        "summary": _normalize_sentence(extract),
        "extra": [],
        "source": page_url
    }


def _contextual_guidance(question: str, context: dict[str, Any]) -> str:
    """Build local, prediction-aware advice even when web context is limited."""
    normalized = (question or "").lower()
    crop = str(context.get("cropType") or "your crop")
    location = str(context.get("location") or "your location")
    advisory = str(context.get("advisory") or "continue regular field monitoring and planning")
    rain_probability = float(context.get("rainProbability") or 0.0)
    confidence = float(context.get("confidence") or 0.0)
    station_name = context.get("stationName")

    if rain_probability >= 0.7:
        risk_note = "Rain signal is high, so prevention should be immediate."
    elif rain_probability >= 0.45:
        risk_note = "Rain signal is moderate, so prepare while keeping operations balanced."
    else:
        risk_note = "Rain signal is low, so focus on irrigation efficiency and observation."

    domain_bundle = (
        f"Agricultural: protect {crop} through drainage, soil moisture checks, and timed inputs. "
        "Social: coordinate alerts with nearby farmers, local labor, and village groups. "
        "Physical safety: avoid electric hazards in wet fields and follow safe field-entry timing. "
        "Theoretical: prediction confidence indicates how strongly image patterns match rainfall examples. "
        "Explanatory: use confidence as a planning signal, not as the only decision input."
    )

    if any(token in normalized for token in ["health", "medical", "fever", "body"]):
        return (
            f"For health and field safety in {location}, keep hydration, heat-rest cycles, and protective gear in place. "
            "Use local medical guidance for diagnosis and treatment. "
            f"{risk_note}"
        )

    if any(token in normalized for token in ["finance", "money", "market", "price", "cost"]):
        return (
            f"For farm finance around {crop}, split spending into immediate protection, essential labor, and optional upgrades. "
            f"Current prediction confidence is {confidence:.2f}. {risk_note}"
        )

    if any(token in normalized for token in ["map", "aws", "station", "weather"]):
        station_line = f"Nearest comparison station: {station_name}. " if station_name else ""
        return (
            f"{station_line}Location-aware planning in {location} should combine radar/ground weather updates with your model output. "
            f"Current rain probability is {rain_probability:.2f}. {risk_note}"
        )

    return (
        f"For {crop} in {location}, {advisory}. {risk_note} "
        f"{domain_bundle}"
    )


def build_dynamic_assistant_response(question: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the final assistant answer plus source links used for enrichment."""
    normalized_question = re.sub(r"\s+", " ", (question or "").strip())
    if not normalized_question:
        normalized_question = "Give me a useful overview for my farm planning."

    context = context or {}

    dynamic_sources: list[str] = []
    snippets: list[str] = []

    duck_context = fetch_duckduckgo_context(normalized_question)
    if duck_context:
        snippets.append(duck_context["summary"])
        snippets.extend(duck_context.get("extra") or [])
        dynamic_sources.append(duck_context["source"])

    wiki_context = fetch_wikipedia_context(normalized_question)
    if wiki_context:
        if wiki_context["summary"] not in snippets:
            snippets.append(wiki_context["summary"])
        dynamic_sources.append(wiki_context["source"])

    guidance = _contextual_guidance(normalized_question, context)
    best_summary = snippets[0] if snippets else (
        "Here is a practical explanation built from your current prediction context and crop location."
    )
    extra_summary = " ".join(snippets[1:3]).strip()

    answer_parts = [
        f"Direct answer: {best_summary}",
        f"Applied guidance for your case: {guidance}"
    ]
    if extra_summary:
        answer_parts.append(f"Additional context: {extra_summary}")

    answer_parts.append(
        "Action checklist: verify field conditions, follow crop-stage priorities, and reassess after new images or weather shifts."
    )

    unique_sources = []
    for source in dynamic_sources:
        if source and source not in unique_sources:
            unique_sources.append(source)

    return {
        "answer": "\n\n".join(answer_parts),
        "sources": unique_sources
    }

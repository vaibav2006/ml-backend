from __future__ import annotations

"""Crop-focused advisory utilities for rainfall prediction results.

This module intentionally avoids location/AWS comparisons. It builds
advisory outputs only from:
- selected crop type
- model rain/no-rain probabilities
- model confidence
"""

from typing import Any


def _clamp_probability(value: float) -> float:
    """Clamp a numeric probability to the [0, 1] range."""
    return max(0.0, min(1.0, float(value)))


def classify_risk_band(rain_probability: float) -> str:
    """Map rain probability into interpretable risk bands."""
    probability = _clamp_probability(rain_probability)
    if probability >= 0.7:
        return "high"
    if probability >= 0.45:
        return "moderate"
    return "low"


def _confidence_narrative(rain_probability: float, model_confidence: float) -> str:
    """Create a human-readable confidence explanation."""
    band = classify_risk_band(rain_probability)
    rain_pct = rain_probability * 100.0
    model_pct = model_confidence * 100.0

    if band == "high":
        return (
            f"High rain possibility from cloud-image features ({rain_pct:.1f}% rain likelihood, "
            f"{model_pct:.1f}% model confidence). Take preventive crop actions now."
        )
    if band == "moderate":
        return (
            f"Moderate rain possibility ({rain_pct:.1f}% rain likelihood, {model_pct:.1f}% model confidence). "
            "Prepare protection measures and monitor conditions regularly."
        )
    return (
        f"Low rain possibility ({rain_pct:.1f}% rain likelihood, {model_pct:.1f}% model confidence). "
        "Continue regular crop planning with routine observation."
    )


def crop_advisory(crop_type: str, rain_probability: float) -> str:
    """Return crop-specific guidance driven only by model rain confidence."""
    crop = (crop_type or "general").strip().lower()
    band = classify_risk_band(rain_probability)

    high = {
        "rice": "Keep drainage channels open, protect nursery beds, and postpone top-dressing before the rainfall window.",
        "wheat": "Protect stored seed and fertilizer, delay foliar spray, and avoid machinery movement on saturated soil.",
        "cotton": "Protect open bolls, strengthen field drainage, and monitor for fungal infection after rain bursts.",
        "maize": "Support young plants, clear runoff paths, and avoid excessive irrigation during probable rainfall hours.",
        "sugarcane": "Check waterlogging risk, secure standing cane against lodging, and schedule field operations in dry gaps."
    }

    moderate = {
        "rice": "Maintain moderate drainage readiness and monitor canopy moisture before fertilizer or spray operations.",
        "wheat": "Keep spray plan flexible and inspect field wetness before stepping into heavy operations.",
        "cotton": "Track humidity build-up, keep fungicide plan ready, and avoid over-irrigation.",
        "maize": "Continue irrigation cautiously and inspect stalk stability if wind and cloud thickness increase.",
        "sugarcane": "Maintain balanced moisture and continue routine monitoring for lodging-prone patches."
    }

    low = {
        "rice": "Continue normal irrigation scheduling and maintain regular nutrient plan.",
        "wheat": "Proceed with standard field operations and monitor weather changes once per day.",
        "cotton": "Continue routine pest and moisture checks with normal cultivation steps.",
        "maize": "Follow normal irrigation and nutrition schedule while observing cloud variation.",
        "sugarcane": "Maintain standard crop care and irrigation intervals without emergency action."
    }

    if band == "high":
        return high.get(crop, "Rain risk is high. Protect sensitive farm operations and improve immediate field drainage.")
    if band == "moderate":
        return moderate.get(crop, "Rain risk is moderate. Keep preventive crop actions ready and monitor confidence changes.")
    return low.get(crop, "Rain risk is low. Continue routine crop management and periodic weather checks.")


def _weekly_checklist(crop_type: str, rain_probability: float) -> list[str]:
    """Generate practical checklist bullets for the selected crop and confidence band."""
    crop = (crop_type or "your crop").strip()
    band = classify_risk_band(rain_probability)

    common = [
        f"Review current field status for {crop} before each major operation.",
        "Re-run prediction when cloud conditions visibly change.",
        "Document actions and outcomes to improve next-cycle planning."
    ]

    if band == "high":
        return [
            "Open all drainage paths and remove water stagnation points.",
            "Delay non-essential spray/fertilizer operations until heavy rain risk drops.",
            "Protect harvested produce and input materials from moisture exposure.",
            *common
        ]
    if band == "moderate":
        return [
            "Keep drainage partially prepared and monitor soil saturation.",
            "Schedule farm activities in safer low-rain windows.",
            "Maintain preventive plant-protection readiness.",
            *common
        ]

    return [
        "Continue planned irrigation and avoid over-watering.",
        "Proceed with routine crop operations while observing cloud changes.",
        "Maintain standard pest and disease monitoring schedule.",
        *common
    ]


def build_confidence_chart(
    rain_probability: float,
    no_rain_probability: float,
    model_confidence: float,
    prediction_label: str
) -> dict[str, Any]:
    """Create chart-friendly confidence payload for frontend visualizations."""
    rain_p = _clamp_probability(rain_probability)
    no_rain_p = _clamp_probability(no_rain_probability)
    model_p = _clamp_probability(model_confidence)
    threshold = 0.7

    return {
        "title": "Rainfall Confidence Chart",
        "threshold": threshold,
        "predictedLabel": prediction_label,
        "points": [
            {
                "label": "No-Rain Probability",
                "value": round(no_rain_p, 6),
                "color": "#8a9aa7",
                "description": "Probability that uploaded cloud image indicates no rainfall"
            },
            {
                "label": "Rain Probability",
                "value": round(rain_p, 6),
                "color": "#29b6d1",
                "description": "Probability that uploaded cloud image indicates rainfall"
            },
            {
                "label": "Prediction Confidence",
                "value": round(model_p, 6),
                "color": "#ffd08c",
                "description": "Confidence of the selected output class"
            }
        ],
        "decision": {
            "rule": "Rain alert triggers when rain probability is greater than or equal to 70%.",
            "isAlert": rain_p >= threshold,
            "gapFromThreshold": round(rain_p - threshold, 6)
        }
    }


def build_farmer_alert(
    crop_type: str,
    prediction_label: str,
    confidence: float,
    rain_probability: float,
    no_rain_probability: float
) -> dict[str, Any]:
    """Build crop-only advisory output from prediction confidence values."""
    selected_crop = (crop_type or "general").strip() or "general"
    rain_p = _clamp_probability(rain_probability)
    no_rain_p = _clamp_probability(no_rain_probability)
    model_conf = _clamp_probability(confidence)

    should_notify = rain_p >= 0.7
    advisory = crop_advisory(selected_crop, rain_p)
    band = classify_risk_band(rain_p)
    confidence_narrative = _confidence_narrative(rain_p, model_conf)
    confidence_chart = build_confidence_chart(rain_p, no_rain_p, model_conf, prediction_label)

    return {
        "title": "Smart Crop Rain Advisory",
        "cropType": selected_crop,
        "predictionLabel": prediction_label,
        "predictionConfidence": round(model_conf, 6),
        "rainConfidence": round(rain_p, 6),
        "riskBand": band,
        "shouldNotify": should_notify,
        "alertThreshold": 0.7,
        "notificationTitle": f"Rain Advisory for {selected_crop}",
        "notificationBody": (
            f"Crop: {selected_crop}. Rain confidence: {rain_p * 100:.1f}%. {advisory}"
            if should_notify
            else f"Crop: {selected_crop}. Rain confidence: {rain_p * 100:.1f}%. Alert not triggered. {advisory}"
        ),
        "advisory": advisory,
        "confidenceNarrative": confidence_narrative,
        "weeklyChecklist": _weekly_checklist(selected_crop, rain_p),
        "userSelection": {
            "cropType": selected_crop
        },
        "sdgBadges": [
            "SDG 2 - Zero Hunger",
            "SDG 8 - Decent Work & Economic Growth",
            "SDG 13 - Climate Action"
        ],
        "confidenceChart": confidence_chart
    }

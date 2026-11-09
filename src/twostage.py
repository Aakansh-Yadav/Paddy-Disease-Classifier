"""Two-stage readout: healthy vs unhealthy, then disease among the nine."""

from __future__ import annotations

from src.config import (
    CLASS_INFO,
    CLASS_TO_IDX,
    HEALTHY_LABEL,
    HEALTHY_THRESHOLD,
    UNHEALTHY_CLASSES,
)


def split_health_and_disease(
    scores: dict[str, float],
    threshold: float = HEALTHY_THRESHOLD,
) -> dict:
    p_healthy = float(scores.get(HEALTHY_LABEL, 0.0))
    p_unhealthy = float(sum(scores.get(name, 0.0) for name in UNHEALTHY_CLASSES))
    health = "healthy" if p_healthy >= threshold else "unhealthy"
    disease_mass = {name: float(scores.get(name, 0.0)) for name in UNHEALTHY_CLASSES}
    total = sum(disease_mass.values())
    if total > 0:
        renormalized = {name: value / total for name, value in disease_mass.items()}
    else:
        renormalized = {name: 0.0 for name in UNHEALTHY_CLASSES}
    ordered = sorted(renormalized.items(), key=lambda item: item[1], reverse=True)[:5]
    disease_ranking = [
        {
            "label": name,
            "probability": score,
            "index": CLASS_TO_IDX[name],
            **CLASS_INFO[name],
        }
        for name, score in ordered
    ]
    top = disease_ranking[0] if disease_ranking else None
    disease = None
    if health == "unhealthy" and top is not None:
        disease = {
            "label": top["label"],
            "title": top["title"],
            "probability": top["probability"],
            "cause": top["cause"],
            "signs": top["signs"],
            "action": top["action"],
            "ranking": disease_ranking,
        }
    return {
        "health": health,
        "p_healthy": p_healthy,
        "p_unhealthy": p_unhealthy,
        "threshold": threshold,
        "disease": disease,
    }

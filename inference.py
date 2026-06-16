# inference.py
import logging
import os

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import CAPABILITIES, TRACK_PROFILES, TRACK_VECS

logger = logging.getLogger(__name__)

# ─── Load trained artifacts ────────────────────────────────────────────────────
_BASE = os.path.dirname(__file__)

try:
    model = joblib.load(os.path.join(_BASE, "model.pkl"))
    le    = joblib.load(os.path.join(_BASE, "label_encoder.pkl"))
except FileNotFoundError as e:
    raise RuntimeError(
        f"Required model file not found: {e}. "
        "Run training.py first to generate model.pkl and label_encoder.pkl."
    ) from e


# ─── Helpers ───────────────────────────────────────────────────────────────────

def weighted_dot_score(student_caps: dict, track_profile: dict) -> float:
    return round(
        sum(weight * student_caps.get(cap, 0.0) for cap, weight in track_profile.items()),
        4,
    )


def validate_input(student_caps: dict) -> None:
    for cap in CAPABILITIES:
        if cap not in student_caps:
            raise ValueError(f"Missing capability: '{cap}'")
        val = student_caps[cap]
        if not isinstance(val, (int, float)):
            raise ValueError(f"Capability '{cap}' must be numeric, got {type(val).__name__}")
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Capability '{cap}' must be between 0 and 1, got {val}")

    if sum(student_caps.values()) == 0:
        raise ValueError(
            "Invalid profile: all capabilities are zero. "
            "Please provide at least one non-zero capability."
        )


# ─── Local XAI ────────────────────────────────────────────────────────────────

def _cap_level(score: float) -> str:
    if score >= 0.85: return "Excellent"
    if score >= 0.70: return "Strong"
    if score >= 0.55: return "Good"
    return "Developing"


def _fit_label(required: float, student: float) -> str:
    if student >= required * 0.90: return "Strong"
    if student >= required * 0.70: return "Good"
    return "Needs improvement"


def explain_recommendation(
    track_name: str,
    capability_vector: dict,
    grades: dict,
    course_weights: dict,
) -> dict:
    """
    Generate a deterministic, local explanation for a recommended track.

    Returns:
        {
            "summary":          str,
            "top_capabilities": [{"capability", "score", "level"}, ...],
            "top_courses":      [{"course", "grade", "contributed_to"}, ...],
            "track_fit":        [{"capability", "required", "student", "fit"}, ...]
        }
    """
    track_caps = TRACK_PROFILES.get(track_name, {})

    # 1. Student's top capabilities
    top_capabilities = [
        {
            "capability": cap,
            "score":      round(score, 2),
            "level":      _cap_level(score),
        }
        for cap, score in sorted(capability_vector.items(), key=lambda x: -x[1])
        if score > 0
    ][:4]

    # 2. Courses that contributed most to this track
    course_contributions = []
    normalized_grades = {k.replace(" ", "").upper(): v for k, v in grades.items()}

    for course, grade in normalized_grades.items():
        weights = course_weights.get(course, {})
        if not weights:
            continue

        contribution = sum(
            track_caps.get(cap, 0) * weights.get(cap, 0) * (float(grade) / 100)
            for cap in track_caps
        )
        if contribution > 0:
            relevant_caps = [
                cap for cap in weights if cap in track_caps and weights[cap] > 0.10
            ]
            course_contributions.append({
                "course":        course,
                "grade":         int(grade),
                "contribution":  round(contribution, 4),
                "contributed_to": relevant_caps,
            })

    course_contributions.sort(key=lambda x: -x["contribution"])
    top_courses = [
        {"course": c["course"], "grade": c["grade"], "contributed_to": c["contributed_to"]}
        for c in course_contributions[:3]
    ]

    # 3. Per-capability fit breakdown
    track_fit = [
        {
            "capability": cap,
            "required":   round(weight, 2),
            "student":    round(capability_vector.get(cap, 0), 2),
            "fit":        _fit_label(weight, capability_vector.get(cap, 0)),
        }
        for cap, weight in sorted(track_caps.items(), key=lambda x: -x[1])
    ]

    # 4. Summary sentence
    strong_fits      = [f["capability"] for f in track_fit if f["fit"] == "Strong"]
    top_course_name  = top_courses[0]["course"] if top_courses else None

    if strong_fits:
        caps_str = " and ".join(strong_fits[:2])
        summary  = (
            f"You're recommended for the {track_name} track because your "
            f"{caps_str} capabilities are a strong match for what this track requires."
        )
        if top_course_name:
            summary += f" Your performance in '{top_course_name}' was a key factor."
    else:
        summary = (
            f"Based on your overall academic profile, {track_name} is the closest "
            f"match to your current capability set."
        )

    return {
        "summary":          summary,
        "top_capabilities": top_capabilities,
        "top_courses":      top_courses,
        "track_fit":        track_fit,
    }


# ─── Main recommendation pipeline ─────────────────────────────────────────────

def recommend(student_caps: dict, grades: dict | None = None, top_k: int = 3) -> dict:
    """
    Full pipeline: capability vector → ML model → ranked tracks → XAI explanations.

    Args:
        student_caps: pre-computed capability vector {cap: score}
        grades:       original raw grades dict (used for XAI; optional)
        top_k:        number of tracks to return

    Returns:
        {
            "student_strengths":  [str, ...],
            "recommendations":    [{track, probability, similarity, weighted_fit, explanation}, ...],
        }
    """
    validate_input(student_caps)

    from mapper import COURSE_CAPABILITY_WEIGHTS  # late import to avoid circular

    cap_values  = [student_caps[cap] for cap in CAPABILITIES]
    student_vec = np.array(cap_values).reshape(1, -1)

    # Similarity + weighted dot features for ML model
    sim_values = [
        cosine_similarity(student_vec, TRACK_VECS[track])[0, 0]
        for track in TRACK_PROFILES
    ]
    wdp_values = [
        weighted_dot_score(student_caps, profile)
        for profile in TRACK_PROFILES.values()
    ]

    x     = np.array(cap_values + sim_values + wdp_values).reshape(1, -1)
    probs = model.predict_proba(x)[0]

    top_idx = probs.argsort()[::-1][:top_k]

    # Student strengths
    top_strengths = [
        cap for cap, _ in sorted(student_caps.items(), key=lambda x: -x[1])[:3]
    ]

    results = []
    for i in top_idx:
        track      = le.classes_[i]
        similarity = cosine_similarity(student_vec, TRACK_VECS[track])[0, 0]
        wdp        = weighted_dot_score(student_caps, TRACK_PROFILES[track])

        rec = {
            "track":        track,
            "probability":  float(round(probs[i] * 100, 2)),
            "similarity":   float(round(similarity * 100, 2)),
            "weighted_fit": float(round(wdp * 100, 2)),
            "explanation":  None,
        }

        if grades is not None:
            rec["explanation"] = explain_recommendation(
                track, student_caps, grades, COURSE_CAPABILITY_WEIGHTS
            )

        results.append(rec)

    return {
        "student_strengths": top_strengths,
        "recommendations":   results,
    }


# ─── Community reranking ───────────────────────────────────────────────────────

def rerank_with_community(
    recommendations: list,
    community_scores: dict,
    weight: float = 0.1,
) -> list:
    """Blend model probability with community scores and re-sort."""
    for rec in recommendations:
        community_boost = community_scores.get(rec["track"], 0.0)
        rec["final_score"] = round(
            (1 - weight) * (rec["probability"] / 100) + weight * community_boost,
            4,
        )
    return sorted(recommendations, key=lambda x: x["final_score"], reverse=True)

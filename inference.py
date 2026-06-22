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
    track_caps = TRACK_PROFILES.get(track_name, {})

    top_capabilities = [
        {
            "capability": cap,
            "score":      round(score, 2),
            "level":      _cap_level(score),
        }
        for cap, score in sorted(capability_vector.items(), key=lambda x: -x[1])
        if score > 0
    ][:4]

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
                "course":         course,
                "grade":          int(grade),
                "contribution":   round(contribution, 4),
                "contributed_to": relevant_caps,
            })

    course_contributions.sort(key=lambda x: -x["contribution"])
    top_courses = [
        {"course": c["course"], "grade": c["grade"], "contributed_to": c["contributed_to"]}
        for c in course_contributions[:3]
    ]

    track_fit = [
        {
            "capability": cap,
            "required":   round(weight, 2),
            "student":    round(capability_vector.get(cap, 0), 2),
            "fit":        _fit_label(weight, capability_vector.get(cap, 0)),
        }
        for cap, weight in sorted(track_caps.items(), key=lambda x: -x[1])
    ]

    strong_fits     = [f["capability"] for f in track_fit if f["fit"] == "Strong"]
    top_course_name = top_courses[0]["course"] if top_courses else None

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


# ─── XAI Helpers for Full Explanation ─────────────────────────────────────────

def _explain_rejection(
    track: str,
    capability_vector: dict,
    ml_scores: dict,
    llm_scores: dict,
) -> str:
    """Explain why a track was not selected as top recommendation."""
    ml_score  = ml_scores.get(track, 0.0)
    llm_score = llm_scores.get(track, 0.0)
    track_caps = TRACK_PROFILES.get(track, {})

    weak_caps = [
        cap for cap, required in track_caps.items()
        if capability_vector.get(cap, 0.0) < required * 0.70
    ]

    reasons = []
    if ml_score < 0.10:
        reasons.append("low ML probability")
    if llm_score < 0.30:
        reasons.append("weak LLM assessment")
    if weak_caps:
        reasons.append(f"insufficient capabilities in {', '.join(weak_caps[:2])}")
    if not reasons:
        reasons.append("lower overall fit compared to selected tracks")

    return "This track was not selected due to: " + "; ".join(reasons) + "."


def _explain_llm_influence(
    ml_scores: dict,
    llm_scores: dict,
    final_scores: dict,
) -> dict:
    """Explain how LLM scores influenced the final ranking."""
    promoted = []
    demoted  = []

    ml_ranking    = sorted(ml_scores.items(),    key=lambda x: -x[1])
    final_ranking = sorted(final_scores.items(), key=lambda x: -x[1])

    for i, (track, _) in enumerate(final_ranking):
        ml_rank   = next(j for j, (t, _) in enumerate(ml_ranking) if t == track)
        llm_score = llm_scores.get(track, 0.0)

        if i < ml_rank and llm_score > ml_scores.get(track, 0):
            promoted.append({
                "track":      track,
                "moved_from": ml_rank + 1,
                "moved_to":   i + 1,
                "llm_score":  round(llm_score, 4),
                "reason":     "LLM strongly endorsed this track",
            })
        elif i > ml_rank and llm_score < ml_scores.get(track, 0):
            demoted.append({
                "track":      track,
                "moved_from": ml_rank + 1,
                "moved_to":   i + 1,
                "llm_score":  round(llm_score, 4),
                "reason":     "LLM assessment was lower than ML prediction",
            })

    return {
        "promoted_by_llm":   promoted[:3],
        "demoted_by_llm":    demoted[:3],
        "overall_influence": "High" if (promoted or demoted) else "Low",
    }


def _explain_confidence(
    final_score: float,
    ml_scores: dict,
    llm_scores: dict,
    track_name: str,
) -> dict:
    """
    Explain the confidence level of the recommendation.

    Thresholds calibrated for the 4-signal blended final_score
    (grades 35% + project 30% + competition 20% + community 15%),
    which typically lands in 0.25–0.65 rather than the 0–1 ML-only range.

      High      final >= 0.50 AND ML >= 0.35 AND LLM >= 0.45 AND agreement < 0.20
      Moderate  final >= 0.30 OR  (ML >= 0.25 AND LLM >= 0.30)
      Low       everything else
    """
    ml_score  = ml_scores.get(track_name, 0.0)
    llm_score = llm_scores.get(track_name, 0.0)

    high_final_score = final_score >= 0.50
    strong_ml        = ml_score    >= 0.35
    strong_llm       = llm_score   >= 0.45
    agreement        = abs(ml_score - llm_score) < 0.20

    if high_final_score and strong_ml and strong_llm and agreement:
        level = "High"
        explanation = (
            "The model is highly confident. Both ML and LLM strongly support "
            "this track with good agreement between the two signals."
        )
    elif final_score >= 0.30 or (ml_score >= 0.25 and llm_score >= 0.30):
        level = "Moderate"
        explanation = (
            "The model has moderate confidence. The final score is reasonable, "
            "but ML and LLM signals show some disagreement."
        )
    else:
        level = "Low"
        explanation = (
            "The model has low confidence. The final score is relatively low, "
            "suggesting this track may not be the best fit."
        )

    return {
        "level":       level,
        "final_score": round(final_score, 4),
        "ml_score":    round(ml_score, 4),
        "llm_score":   round(llm_score, 4),
        "explanation": explanation,
    }


def _explain_feature_importance(
    capability_vector: dict,
    ml_scores: dict,
) -> dict:
    """Explain which capabilities were most important."""
    sorted_caps = sorted(capability_vector.items(), key=lambda x: -x[1])

    top_features = []
    for cap, value in sorted_caps[:5]:
        if value > 0:
            impact = (
                "Very High" if value >= 0.85 else
                "High"      if value >= 0.70 else
                "Medium"    if value >= 0.50 else
                "Low"
            )
            top_features.append({
                "capability": cap,
                "value":      round(value, 4),
                "impact":     impact,
            })

    avg_capability = np.mean([v for v in capability_vector.values() if v > 0]) if capability_vector else 0
    profile_strength = (
        "Strong"    if avg_capability >= 0.70 else
        "Moderate"  if avg_capability >= 0.50 else
        "Developing"
    )

    return {
        "top_features":         top_features,
        "profile_strength":     profile_strength,
        "avg_capability_score": round(avg_capability, 4),
        "explanation": (
            f"Your profile shows {profile_strength.lower()} capabilities overall. "
            f"The most influential factors are your "
            f"{', '.join([f['capability'] for f in top_features[:3]])} capabilities."
        ),
    }


def explain_full_recommendation(
    track_name: str,
    capability_vector: dict,
    grades: dict,
    course_weights: dict,
    ml_scores: dict,
    llm_scores: dict,
    final_scores: dict,
) -> dict:
    """
    Full XAI explanation covering:
    1. Why this track was chosen
    2. Why other tracks were rejected
    3. How LLM influenced the final ranking
    4. Confidence level explanation
    5. Feature importance breakdown
    """
    track_explanation = explain_recommendation(
        track_name, capability_vector, grades, course_weights
    )

    other_tracks = [
        {
            "track":       track,
            "final_score": score,
            "ml_score":    ml_scores.get(track, 0),
            "llm_score":   llm_scores.get(track, 0),
            "reason":      _explain_rejection(track, capability_vector, ml_scores, llm_scores),
        }
        for track, score in sorted(final_scores.items(), key=lambda x: -x[1])
        if track != track_name
    ]

    return {
        "track_explanation": track_explanation,
        "rejected_tracks":   other_tracks[:3],
        "llm_influence":     _explain_llm_influence(ml_scores, llm_scores, final_scores),
        "confidence":        _explain_confidence(
                                 final_scores[track_name], ml_scores, llm_scores, track_name
                             ),
        "feature_importance": _explain_feature_importance(capability_vector, ml_scores),
    }


# ─── Main recommendation pipeline ─────────────────────────────────────────────

def recommend(student_caps: dict, grades: dict | None = None, top_k: int = 3) -> dict:
    validate_input(student_caps)

    from mapper import COURSE_CAPABILITY_WEIGHTS

    cap_values  = [student_caps[cap] for cap in CAPABILITIES]
    student_vec = np.array(cap_values).reshape(1, -1)

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

    ml_scores_normalized: dict[str, float] = {
        le.classes_[i]: float(round(probs[i], 4))
        for i in range(len(le.classes_))
    }

    top_idx = probs.argsort()[::-1][:top_k]

    top_strengths = [
        cap for cap, _ in sorted(student_caps.items(), key=lambda x: -x[1])[:3]
    ]

    results = []
    for i in top_idx:
        track      = le.classes_[i]
        similarity = cosine_similarity(student_vec, TRACK_VECS[track])[0, 0]
        wdp        = weighted_dot_score(student_caps, TRACK_PROFILES[track])

        rec = {
            "track":               track,
            "probability":         float(round(probs[i] * 100, 2)),
            "ml_score_normalized": ml_scores_normalized[track],
            "similarity":          float(round(similarity * 100, 2)),
            "weighted_fit":        float(round(wdp * 100, 2)),
            "explanation":         None,
        }

        if grades is not None:
            rec["explanation"] = explain_recommendation(
                track, student_caps, grades, COURSE_CAPABILITY_WEIGHTS
            )

        results.append(rec)

    return {
        "student_strengths": top_strengths,
        "recommendations":   results,
        "ml_scores":         ml_scores_normalized,
    }


# ─── Soft voting: ML + LLM ────────────────────────────────────────────────────

def soft_vote(
    ml_scores: dict[str, float],
    llm_scores: dict[str, float],
    ml_weight: float = 0.40,
    top_k: int = 3,
) -> list[dict]:
    llm_weight = 1.0 - ml_weight
    all_tracks = set(ml_scores) | set(llm_scores)
    blended = []

    for track in all_tracks:
        ml  = ml_scores.get(track, 0.0)
        llm = llm_scores.get(track, 0.0)

        if llm == 0.0:
            final = round(ml * 0.6, 4)
        else:
            final = round(ml_weight * ml + llm_weight * llm, 4)

        blended.append({
            "track":       track,
            "ml_score":    round(ml, 4),
            "llm_score":   round(llm, 4),
            "final_score": final,
        })

    blended.sort(key=lambda x: -x["final_score"])
    blended = [b for b in blended if b["final_score"] >= 0.05]
    return blended[:top_k]


# rerank_with_community was removed.
# Community signal is now one of the 4 weighted inputs to get_llm_track_scores()
# (weight=0.15) and is blended before soft_vote — not applied as post-processing.

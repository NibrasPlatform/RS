# routes.py
import logging
import os

from flask import Blueprint, jsonify, request

from inference import recommend, soft_vote, rerank_with_community
from llm_scorer import get_llm_track_scores, get_community_scores
from mapper import (
    COURSE_CAPABILITY_WEIGHTS,
    add_course_to_weights,
    grades_to_capabilities,
)

logger = logging.getLogger(__name__)

recommend_bp = Blueprint("recommend", __name__)


# ─── POST /recommend ──────────────────────────────────────────────────────────

@recommend_bp.route("/recommend", methods=["POST"])
def recommend_api():
    """
    Accept either:
      { "grades": {"CS103": 90, ...} }
    or:
      { "capabilities": {"Math": 0.9, ...} }

    Optional LLM signal fields (any combination):
      "top_comment":          str              — community comment text
      "correct_answers":      [str, ...]       — correct quiz answer strings
      "problem_type_ranks":   {"Graph": 120}   — problem type → count/rank

    Optional tuning:
      "ml_weight":  float (default 0.60)  — weight for ML in soft voting
                    LLM weight = 1 - ml_weight
    """
    data = request.get_json(silent=True)

    if not data or ("grades" not in data and "capabilities" not in data):
        return jsonify({"status": "error", "message": "Missing 'grades' or 'capabilities' field"}), 400

    # ── Resolve capability vector ─────────────────────────────────────────────
    grades = None

    if "grades" in data:
        grades = data["grades"]
        if not grades:
            return jsonify({"status": "error", "message": "Grades cannot be empty"}), 400

        for course, grade in grades.items():
            if grade is None:
                continue
            if not isinstance(grade, (int, float)):
                return jsonify({
                    "status": "error",
                    "message": f"Invalid grade for '{course}': must be a number"
                }), 400
            if not (0 <= grade <= 100):
                return jsonify({
                    "status": "error",
                    "message": f"Invalid grade for '{course}': must be between 0 and 100, got {grade}"
                }), 400

        caps, unknown = grades_to_capabilities(grades)
    else:
        caps    = data["capabilities"]
        unknown = []

    if all(v == 0 for v in caps.values()):
        return jsonify({"status": "error", "message": "All capabilities are zero"}), 400

    # ── ML recommendation ─────────────────────────────────────────────────────
    try:
        ml_result = recommend(caps, grades=grades)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    ml_scores    = ml_result["ml_scores"]          # {track: 0–1}
    track_names  = [r["track"] for r in ml_result["recommendations"]]

    # ── Collect LLM signals ───────────────────────────────────────────────────
    top_comment        = data.get("top_comment", "").strip()
    correct_answers    = data.get("correct_answers") or []
    problem_type_ranks = data.get("problem_type_ranks") or {}

    has_llm_signal = bool(top_comment or correct_answers or problem_type_ranks)

    # Dynamic ml_weight: more signals = more trust in LLM = lower ml_weight
    signal_count = sum([
        1 if top_comment else 0,
        1 if correct_answers else 0,
        1 if problem_type_ranks else 0,
    ])
    ml_weight = {0: 1.0, 1: 0.75, 2: 0.60, 3: 0.50}[signal_count]

    llm_scores: dict[str, float] = {}
    llm_signal_used: list[str]   = []

    if has_llm_signal:
        llm_scores = get_llm_track_scores(
            top_comment=top_comment,
            correct_answers=correct_answers if correct_answers else None,
            problem_type_ranks=problem_type_ranks if problem_type_ranks else None,
            recommended_tracks=track_names,
        )
        if top_comment:        llm_signal_used.append("community_comment")
        if correct_answers:    llm_signal_used.append("quiz_answers")
        if problem_type_ranks: llm_signal_used.append("problem_solving")

    # ── Soft voting ───────────────────────────────────────────────────────────
    if llm_scores and any(v > 0 for v in llm_scores.values()):
        voted = soft_vote(ml_scores, llm_scores, ml_weight=ml_weight, top_k=3)

        # Attach ML explanation to voted tracks
        from inference import explain_recommendation
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        import numpy as np
        from config import CAPABILITIES, TRACK_PROFILES, TRACK_VECS

        ml_recs_by_track = {r["track"]: r for r in ml_result["recommendations"]}
        student_vec = np.array([caps[c] for c in CAPABILITIES]).reshape(1, -1)

        final_recommendations = []
        for i, v in enumerate(voted):
            track  = v["track"]
            ml_rec = ml_recs_by_track.get(track)

            if ml_rec:
                # Track was in ML top_k — use its data directly
                probability  = ml_rec.get("probability")
                similarity   = ml_rec.get("similarity")
                weighted_fit = ml_rec.get("weighted_fit")
                explanation  = ml_rec.get("explanation")
            else:
                # Track was promoted by LLM — compute its stats on the fly
                from inference import weighted_dot_score
                probability  = round(ml_scores.get(track, 0.0) * 100, 2)
                similarity   = round(float(cos_sim(student_vec, TRACK_VECS[track])[0, 0]) * 100, 2)
                weighted_fit = round(weighted_dot_score(caps, TRACK_PROFILES[track]) * 100, 2)
                explanation  = explain_recommendation(
                    track, caps, grades or {}, COURSE_CAPABILITY_WEIGHTS
                ) if grades else None

            final_recommendations.append({
                "rank":           i + 1,
                "track":          track,
                "final_score":    v["final_score"],
                "ml_score":       v["ml_score"],
                "llm_score":      v["llm_score"],
                "probability":    probability,
                "similarity":     similarity,
                "weighted_fit":   weighted_fit,
                "explanation":    explanation,
                "llm_promoted":   ml_rec is None,   # flag: LLM رفع التراك ده
            })

        scoring_method = "soft_voting"
    else:
        # No LLM signal — return pure ML result
        final_recommendations = [
            {
                "rank":          i + 1,
                "track":         rec["track"],
                "final_score":   rec["ml_score_normalized"],
                "ml_score":      rec["ml_score_normalized"],
                "llm_score":     None,
                "probability":   rec["probability"],
                "similarity":    rec["similarity"],
                "weighted_fit":  rec["weighted_fit"],
                "explanation":   rec.get("explanation"),
            }
            for i, rec in enumerate(ml_result["recommendations"])
        ]
        scoring_method = "ml_only"

    # ── Build response ────────────────────────────────────────────────────────
    top = final_recommendations[0] if final_recommendations else {}

    response = {
        "student_summary": {
            "strengths":      ml_result["student_strengths"],
            "top_capability": ml_result["student_strengths"][0] if ml_result["student_strengths"] else None,
        },
        "top_recommendation": {
            "track":     top.get("track"),
            "score":     top.get("final_score"),
            "why":       (top.get("explanation") or {}).get("summary"),
        },
        "recommendations": final_recommendations,
        "meta": {
            "scoring_method":   scoring_method,
            "ml_weight":        ml_weight if has_llm_signal else 1.0,
            "llm_weight":       round(1 - ml_weight, 2) if has_llm_signal else 0.0,
            "llm_signals_used": llm_signal_used,
        },
        "insights": {
            "confidence_level": "High" if (top.get("final_score") or 0) > 0.35 else "Low",
        },
    }

    if llm_scores:
        response["llm_track_scores"] = llm_scores

    if unknown:
        response["warnings"] = {
            "unknown_courses": unknown,
            "message": "These courses were not found in course_weights.json and were ignored.",
        }

    return jsonify(response), 200


# ─── GET /courses ─────────────────────────────────────────────────────────────

@recommend_bp.route("/courses", methods=["GET"])
def list_courses():
    """Return all known courses and their capability weights."""
    return jsonify({"courses": list(COURSE_CAPABILITY_WEIGHTS.keys())}), 200


# ─── POST /courses/add ────────────────────────────────────────────────────────

@recommend_bp.route("/courses/add", methods=["POST"])
def add_course():
    """
    Dynamically add a new course by generating its weights via GPT-4o-mini.

    Body: { "course_name": "Advanced Computer Vision" }
    """
    data = request.get_json(silent=True)

    if not data or "course_name" not in data:
        return jsonify({"error": "Missing 'course_name'"}), 400

    course_name = data["course_name"].strip()
    if not course_name:
        return jsonify({"error": "'course_name' cannot be empty"}), 400

    normalized = course_name.replace(" ", "").upper()
    if normalized in COURSE_CAPABILITY_WEIGHTS:
        return jsonify({
            "message": "Course already exists",
            "weights": COURSE_CAPABILITY_WEIGHTS[normalized],
        }), 200

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY is not set on the server"}), 500

    try:
        weights = add_course_to_weights(course_name, api_key)
        return jsonify({
            "course":  normalized,
            "weights": weights,
            "message": "Weights generated and saved successfully",
        }), 201
    except Exception as e:
        logger.exception("Failed to generate weights for course '%s'", course_name)
        return jsonify({"error": str(e)}), 500

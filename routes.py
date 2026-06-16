# routes.py
import logging
import os

from flask import Blueprint, jsonify, request

from inference import recommend, rerank_with_community
from llm_scorer import get_community_scores
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

    Optional fields:
      "top_comment": str   — free-text community signal for reranking
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

        # Validate every grade value before processing
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

    # ── Model recommendation ──────────────────────────────────────────────────
    try:
        result = recommend(caps, grades=grades)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    # ── Community reranking (optional) ────────────────────────────────────────
    top_comment = data.get("top_comment", "").strip()
    if top_comment:
        track_names      = [rec["track"] for rec in result["recommendations"]]
        community_scores = get_community_scores(top_comment, recommended_tracks=track_names)

        if any(score > 0 for score in community_scores.values()):
            result["recommendations"] = rerank_with_community(
                result["recommendations"], community_scores, weight=0.1
            )
            result["community_scores"] = community_scores

    # ── Build response ────────────────────────────────────────────────────────
    top = result["recommendations"][0] if result["recommendations"] else {}

    response = {
        "student_summary": {
            "strengths":      result["student_strengths"],
            "top_capability": result["student_strengths"][0] if result["student_strengths"] else None,
        },
        "top_recommendation": {
            "track":     top.get("track"),
            "match_pct": top.get("probability"),
            "why":       (top.get("explanation") or {}).get("summary"),
        },
        "recommendations": [
            {
                "rank":        i + 1,
                "track":       rec["track"],
                "probability": rec["probability"],
                "similarity":  rec["similarity"],
                "weighted_fit":rec["weighted_fit"],
                "explanation": rec.get("explanation"),
            }
            for i, rec in enumerate(result["recommendations"])
        ],
        "insights": {
            "confidence_level": "High" if top.get("probability", 0) > 60 else "Low",
        },
    }

    if unknown:
        response["warnings"] = {
            "unknown_courses": unknown,
            "message": "These courses were not found in course_weights.json and were ignored.",
        }

    if "community_scores" in result:
        response["community_scores"] = result["community_scores"]

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
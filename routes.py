# routes.py
import logging
import os

from flask import Blueprint, jsonify, request

from inference import recommend, soft_vote, rerank_with_community
from llm_scorer import get_llm_track_scores, get_community_scores, explain_recommendation_to_student
from mapper import (
    COURSE_CAPABILITY_WEIGHTS,
    add_course_to_weights,
    grades_to_capabilities,
    project_grades_to_capabilities,
)

logger = logging.getLogger(__name__)

recommend_bp = Blueprint("recommend", __name__)


# ─── POST /recommend ──────────────────────────────────────────────────────────

@recommend_bp.route("/recommend", methods=["POST"])
def recommend_api():
    """
    Required fields:
      "grades":             {course: grade (0-100)}   — student course grades
      "project_grades":     {project_title: grade}    — student project/GP grades
      "competition_ranks":  {problem_type: count}     — problems solved per type
                            (raw platform tags supported: "dp", "pwn", etc.)
      "top_comment":        str                       — top upvoted community comment

    Optional:
      "competition_rating": {"platform": "codeforces", "value": 1850}
                             or {"percentile": 92}
                            — actual competition rank/rating, used to scale
                              the competition_ranks signal's confidence.

    All four required fields are needed. The system is designed for students
    who have completed at least two years and have activity across all
    signal sources.
    """
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"status": "error", "message": "Request body is empty"}), 400

    # All four signals are required
    missing = []
    if "grades" not in data and "capabilities" not in data:
        missing.append("grades")
    if not data.get("top_comment", "").strip():
        missing.append("top_comment")
    if not data.get("project_grades"):
        missing.append("project_grades")
    if not data.get("competition_ranks"):
        missing.append("competition_ranks")

    if missing:
        return jsonify({
            "status": "error",
            "message": f"Missing required fields: {', '.join(missing)}",
            "required": ["grades", "project_grades", "competition_ranks", "top_comment"]
        }), 400

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
    top_comment         = data.get("top_comment", "").strip()
    project_grades      = data.get("project_grades") or {}
    competition_ranks   = data.get("competition_ranks") or {}
    competition_rating  = data.get("competition_rating") or None

    # All 3 LLM signals are always present.
    # LLM weight > ML weight because ML is trained on synthetic data.
    ml_weight = 0.40

    llm_scores: dict[str, float] = {}
    llm_signal_used: list[str]   = []

    # Pre-warm the project capability cache (auto-generates weights for any
    # project title not seen before) so the XAI explanation has real focus
    # areas to reference, and so we can surface "unknown" projects too.
    unknown_projects: list[str] = []
    if project_grades:
        api_key = os.getenv("OPENAI_API_KEY")
        _, unknown_projects = project_grades_to_capabilities(project_grades, api_key=api_key)

    # grades are always passed to LLM for contextual interpretation
    has_llm_signal = bool(top_comment or project_grades or competition_ranks or grades)

    if has_llm_signal:
        llm_scores = get_llm_track_scores(
            top_comment=top_comment,
            project_grades=project_grades if project_grades else None,
            competition_ranks=competition_ranks if competition_ranks else None,
            competition_rating=competition_rating,
            grades=grades,
            recommended_tracks=track_names,
        )
        if top_comment:        llm_signal_used.append("community_comment")
        if project_grades:     llm_signal_used.append("project_grades")
        if competition_ranks:  llm_signal_used.append("competition_rank")
        if grades:              llm_signal_used.append("grades")

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

            # ── Track fit health check ───────────────────────────────────────
            fit_warning = None
            if explanation:
                track_fit  = explanation.get("track_fit", [])
                weak_caps  = [
                    f["capability"] for f in track_fit
                    if f["fit"] == "Needs improvement" and f["required"] >= 0.15
                ]
                if weak_caps:
                    fit_warning = (
                        f"Missing key capabilities for this track: "
                        f"{', '.join(weak_caps)}. "
                        f"Consider strengthening these areas before committing to this path."
                    )

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
                "llm_promoted":   ml_rec is None,
                "fit_warning":    fit_warning,
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

    # ── Generate student-facing XAI for top recommendation ──────────────────
    top_rec = final_recommendations[0] if final_recommendations else {}
    student_xai = None
    if top_rec:
        student_xai = explain_recommendation_to_student(
            track_name=top_rec["track"],
            grades=grades or {},
            top_comment=top_comment,
            project_grades=project_grades,
            competition_ranks=competition_ranks,
            competition_rating=competition_rating,
            ml_score=top_rec["ml_score"],
            llm_score=top_rec["llm_score"],
            final_score=top_rec["final_score"],
        )

    # ── Build response ────────────────────────────────────────────────────────
    top = top_rec

    response = {
        "student_summary": {
            "strengths":      ml_result["student_strengths"],
            "top_capability": ml_result["student_strengths"][0] if ml_result["student_strengths"] else None,
        },
        "top_recommendation": {
            "track":     top.get("track"),
            "score":     top.get("final_score"),
            "why":       (top.get("explanation") or {}).get("summary"),
            "xai":       student_xai,
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

    warnings = {}
    if unknown:
        warnings["unknown_courses"] = unknown
    if unknown_projects:
        warnings["unknown_projects"] = unknown_projects
    if warnings:
        warnings["message"] = (
            "Courses are ignored if not found in course_weights.json. "
            "Projects listed in 'unknown_projects' could not be auto-scored "
            "(e.g. OPENAI_API_KEY not set) and were ignored."
        )
        response["warnings"] = warnings

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

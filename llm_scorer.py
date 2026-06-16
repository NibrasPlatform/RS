# llm_scorer.py
import json
import logging
import os
import re

from openai import OpenAI

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────

CAPABILITIES = [
    "Programming", "Algorithms", "Math", "Theory",
    "Data", "Systems", "Hardware", "AI",
    "UX", "Security", "Graphics", "Biology",
]

TRACKS = [
    "Artificial Intelligence",
    "Systems",
    "Theory",
    "Human-Computer Interaction",
    "Visual Computing",
    "Computer Engineering",
    "Information Track",
    "Computational Biology",
]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None


# ─── Shared helpers ────────────────────────────────────────────────────────────

def _call_openai(prompt: str) -> str | None:
    """Single-point OpenAI call. Returns raw text or None on failure."""
    if client is None:
        logger.warning("No OPENAI_API_KEY set.")
        return None
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    except Exception as e:
        logger.error("OpenAI API call failed: %s", e)
        return None


def _zero_track_scores() -> dict[str, float]:
    return {t: 0.0 for t in TRACKS}


def _parse_track_scores(raw: str) -> dict[str, float]:
    """Parse and clamp a JSON dict of {track: score}."""
    try:
        scores: dict = json.loads(raw)
        return {
            t: float(max(0.0, min(1.0, scores.get(t, 0.0))))
            for t in TRACKS
        }
    except (json.JSONDecodeError, TypeError) as e:
        logger.error("Failed to parse track scores: %s | raw=%r", e, raw)
        return _zero_track_scores()


# ─── 1. Community signal scorer ────────────────────────────────────────────────

def get_community_scores(
    top_comment: str,
    recommended_tracks: list[str] | None = None,
) -> dict[str, float]:
    """
    Score a student's top community comment against each CS track (0.0–1.0).

    Args:
        top_comment:         Free-text comment from the community feature.
        recommended_tracks:  If provided, scores for unrelated tracks are zeroed.

    Returns:
        {track_name: score} for all TRACKS.
    """
    if not top_comment or not top_comment.strip():
        logger.warning("Empty comment; returning zero scores.")
        return _zero_track_scores()

    if client is None:
        return _zero_track_scores()

    relevance_note = ""
    if recommended_tracks:
        relevance_note = f"""
The model already recommended these tracks: {', '.join(recommended_tracks)}.
- If the comment is unrelated to these tracks → return all scores = 0.
- Only boost tracks the comment clearly signals.
"""

    track_list = "\n".join(f"- {t}" for t in TRACKS)
    prompt = f"""You are a track-scoring system for a CS education platform.

Given a student comment, assign a score (0.0 to 1.0) for EACH track below
reflecting how strongly the comment signals interest or strength in that track.

Rules:
- Use soft scoring (not binary). Multiple tracks can score high.
- Do NOT assign 1.0 unless extremely certain.
- If the comment is irrelevant, spam, or has no academic signal → all scores = 0.
- Output valid JSON only. No explanation, no markdown.
{relevance_note}
Tracks:
{track_list}

Student comment:
{top_comment}"""

    raw = _call_openai(prompt)
    return _parse_track_scores(raw) if raw else _zero_track_scores()


# ─── 2. Quiz answer analyzer ───────────────────────────────────────────────────

def get_quiz_scores(correct_answers: list[str]) -> dict[str, float]:
    """
    Analyze a student's correct quiz answers via NLU and return track scores.

    The LLM acts as NLU: it infers what CS concepts each correct answer
    demonstrates, then maps that understanding to track relevance scores.

    Args:
        correct_answers: List of correct answer strings from quizzes
                         e.g. ["O(n log n) because merge sort divides...",
                               "A deadlock occurs when two processes..."]

    Returns:
        {track_name: score (0.0–1.0)} for all TRACKS.
    """
    if not correct_answers:
        return _zero_track_scores()

    answers_block = "\n".join(
        f"{i+1}. {ans}" for i, ans in enumerate(correct_answers[:20])  # cap at 20
    )
    track_list = "\n".join(f"- {t}" for t in TRACKS)

    prompt = f"""You are a CS education analyst performing NLU on student quiz answers.

Step 1 – Understand the concepts demonstrated:
  Read each correct answer and identify the CS concepts, skills, or knowledge areas shown.

Step 2 – Map to tracks:
  For each track below, assign a score (0.0 to 1.0) reflecting how strongly
  the demonstrated concepts align with that track's focus.

Rules:
- Base scores ONLY on what the answers clearly demonstrate. Do not guess.
- Use soft, continuous scoring. Multiple tracks can score high.
- If answers show no clear signal for a track → score = 0.
- Output valid JSON only: {{"track_name": score, ...}}. No explanation.

Tracks:
{track_list}

Student's correct quiz answers:
{answers_block}"""

    raw = _call_openai(prompt)
    return _parse_track_scores(raw) if raw else _zero_track_scores()


# ─── 3. Problem-solving rank analyzer ─────────────────────────────────────────

def get_problem_solving_scores(
    problem_type_ranks: dict[str, int | float],
) -> dict[str, float]:
    """
    Convert problem-solving activity ranks into track scores.

    Args:
        problem_type_ranks: {problem_type: rank_or_count}
            e.g. {"Graph": 120, "DP": 95, "System Design": 40, "Math": 30}
            Higher value = more problems solved of that type.

    Returns:
        {track_name: score (0.0–1.0)} for all TRACKS.
    """
    if not problem_type_ranks:
        return _zero_track_scores()

    # Normalize counts to relative proportions so LLM sees signal strength
    total = sum(problem_type_ranks.values()) or 1
    proportions = {k: round(v / total, 3) for k, v in problem_type_ranks.items()}

    track_list = "\n".join(f"- {t}" for t in TRACKS)
    prompt = f"""You are a CS track advisor analyzing a student's problem-solving activity.

Below are the types of problems the student solves most (as proportions of total activity):
{json.dumps(proportions, indent=2)}

For each track, assign a score (0.0 to 1.0) reflecting how well this
problem-solving pattern aligns with the track's focus.

Rules:
- Higher proportion = stronger signal. Weight scores accordingly.
- Use soft scoring. Multiple tracks can be high.
- Output valid JSON only: {{"track_name": score, ...}}. No explanation.

Tracks:
{track_list}"""

    raw = _call_openai(prompt)
    return _parse_track_scores(raw) if raw else _zero_track_scores()


# ─── 4. Aggregate LLM scorer (combines all signals) ───────────────────────────

def get_llm_track_scores(
    top_comment: str = "",
    correct_answers: list[str] | None = None,
    problem_type_ranks: dict[str, int | float] | None = None,
    recommended_tracks: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Aggregate all LLM-based signals into a single per-track score (0.0–1.0).

    Each available signal is scored independently, then combined via
    weighted average so the result stays in the same [0, 1] range as
    the ML model's normalized probabilities.

    Args:
        top_comment:          Community comment text.
        correct_answers:      List of correct quiz answer strings.
        problem_type_ranks:   {problem_type: count/rank}.
        recommended_tracks:   Tracks already suggested by ML (used to
                              filter community score).
        weights:              Override default signal weights.
                              Keys: "community", "quiz", "problem_solving".

    Returns:
        {track_name: aggregated_score (0.0–1.0)}
    """
    default_weights = {"community": 0.35, "quiz": 0.45, "problem_solving": 0.20}
    w = {**default_weights, **(weights or {})}

    scores_list: list[tuple[dict, float]] = []  # (score_dict, weight)

    if top_comment and top_comment.strip():
        community = get_community_scores(top_comment, recommended_tracks)
        scores_list.append((community, w["community"]))
        logger.info("Community signal collected.")

    if correct_answers:
        quiz = get_quiz_scores(correct_answers)
        scores_list.append((quiz, w["quiz"]))
        logger.info("Quiz signal collected.")

    if problem_type_ranks:
        ps = get_problem_solving_scores(problem_type_ranks)
        scores_list.append((ps, w["problem_solving"]))
        logger.info("Problem-solving signal collected.")

    if not scores_list:
        return _zero_track_scores()

    # Weighted average over only the signals that were provided
    total_w = sum(weight for _, weight in scores_list)
    aggregated = {}
    for track in TRACKS:
        aggregated[track] = round(
            sum(s[track] * wt for s, wt in scores_list) / total_w,
            4,
        )

    return aggregated

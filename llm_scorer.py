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

# ─── Problem type → Track mapping ─────────────────────────────────────────────
# Each problem type maps to the tracks it signals, with a weight (0–1).
# Used to give the LLM explicit context instead of guessing.

PROBLEM_TYPE_TRACK_MAP: dict[str, dict[str, float]] = {
    # Artificial Intelligence
    "Machine Learning":         {"Artificial Intelligence": 0.90, "Information Track": 0.10},
    "Neural Networks":          {"Artificial Intelligence": 0.95},
    "Search & Optimization":    {"Artificial Intelligence": 0.70, "Theory": 0.30},
    "NLP Problems":             {"Artificial Intelligence": 0.85, "Information Track": 0.15},
    "Computer Vision":          {"Visual Computing": 0.70, "Artificial Intelligence": 0.30},

    # Theory
    "Dynamic Programming":      {"Theory": 0.70, "Artificial Intelligence": 0.20, "Information Track": 0.10},
    "Graph Algorithms":         {"Theory": 0.60, "Artificial Intelligence": 0.20, "Systems": 0.20},
    "Math / Proofs":            {"Theory": 0.80, "Artificial Intelligence": 0.20},
    "Combinatorics":            {"Theory": 0.85, "Computational Biology": 0.15},
    "Automata / Complexity":    {"Theory": 0.95},

    # Systems
    "OS Concepts":              {"Systems": 0.90, "Computer Engineering": 0.10},
    "Memory Management":        {"Systems": 0.70, "Computer Engineering": 0.30},
    "Concurrency":              {"Systems": 0.85, "Computer Engineering": 0.15},
    "Networking":               {"Systems": 0.75, "Information Track": 0.25},
    "Compilers":                {"Systems": 0.80, "Theory": 0.20},

    # Computer Engineering
    "Circuit Design":           {"Computer Engineering": 0.95},
    "Low-level Programming":    {"Computer Engineering": 0.70, "Systems": 0.30},
    "Hardware Architecture":    {"Computer Engineering": 0.90, "Systems": 0.10},
    "Embedded Systems":         {"Computer Engineering": 0.85, "Systems": 0.15},

    # Visual Computing
    "Geometry / Computational Geometry": {"Visual Computing": 0.90, "Theory": 0.10},
    "Image Processing":         {"Visual Computing": 0.85, "Artificial Intelligence": 0.15},
    "Graphics Rendering":       {"Visual Computing": 0.95},
    "Simulation":               {"Visual Computing": 0.60, "Artificial Intelligence": 0.40},

    # Human-Computer Interaction
    "UI / UX Problems":         {"Human-Computer Interaction": 0.95},
    "Accessibility":            {"Human-Computer Interaction": 0.90},
    "Human Factors":            {"Human-Computer Interaction": 0.85},

    # Information Track
    "Database & SQL":           {"Information Track": 0.90, "Systems": 0.10},
    "Data Analysis":            {"Information Track": 0.70, "Artificial Intelligence": 0.30},
    "Cryptography / Security":  {"Information Track": 0.60, "Systems": 0.40},
    "Information Retrieval":    {"Information Track": 0.80, "Artificial Intelligence": 0.20},

    # Computational Biology
    "Sequence Alignment":       {"Computational Biology": 0.95},
    "Bioinformatics":           {"Computational Biology": 0.90, "Information Track": 0.10},
    "Genomics / Statistics":    {"Computational Biology": 0.80, "Artificial Intelligence": 0.20},

    # General (signal to multiple tracks)
    "Data Structures":          {"Theory": 0.40, "Systems": 0.30, "Artificial Intelligence": 0.30},
    "Sorting / Searching":      {"Theory": 0.50, "Artificial Intelligence": 0.30, "Information Track": 0.20},
    "Probability & Statistics": {"Artificial Intelligence": 0.50, "Computational Biology": 0.30, "Information Track": 0.20},
}


# ─── Platform tag → Standard problem type mapping ─────────────────────────────
# Converts raw tags from Codeforces, LeetCode, TryHackMe to standard types.
# Unknown tags are passed to the LLM as-is for inference.

PLATFORM_TAG_MAP: dict[str, str] = {
    # ── Codeforces tags ────────────────────────────────────────────────────────
    "dp":                        "Dynamic Programming",
    "dynamic programming":       "Dynamic Programming",
    "graphs":                    "Graph Algorithms",
    "graph":                     "Graph Algorithms",
    "shortest paths":            "Graph Algorithms",
    "trees":                     "Graph Algorithms",
    "dfs and similar":           "Graph Algorithms",
    "math":                      "Math / Proofs",
    "number theory":             "Math / Proofs",
    "combinatorics":             "Combinatorics",
    "probabilities":             "Probability & Statistics",
    "geometry":                  "Geometry / Computational Geometry",
    "strings":                   "Data Structures",
    "data structures":           "Data Structures",
    "sorting":                   "Sorting / Searching",
    "binary search":             "Sorting / Searching",
    "greedy":                    "Sorting / Searching",
    "brute force":               "Sorting / Searching",
    "two pointers":              "Data Structures",
    "implementation":            "Data Structures",
    "bitmasks":                  "Data Structures",
    "constructive algorithms":   "Automata / Complexity",
    "games":                     "Theory",
    "flows":                     "Graph Algorithms",
    "matrices":                  "Math / Proofs",
    "fft":                       "Math / Proofs",
    "hashing":                   "Cryptography / Security",

    # ── LeetCode topics ────────────────────────────────────────────────────────
    "dynamic programming":       "Dynamic Programming",
    "tree":                      "Graph Algorithms",
    "graph":                     "Graph Algorithms",
    "depth-first search":        "Graph Algorithms",
    "breadth-first search":      "Graph Algorithms",
    "union find":                "Graph Algorithms",
    "topological sort":          "Graph Algorithms",
    "array":                     "Data Structures",
    "string":                    "Data Structures",
    "hash table":                "Data Structures",
    "stack":                     "Data Structures",
    "queue":                     "Data Structures",
    "heap (priority queue)":     "Data Structures",
    "linked list":               "Data Structures",
    "binary search":             "Sorting / Searching",
    "sorting":                   "Sorting / Searching",
    "two pointers":              "Data Structures",
    "sliding window":            "Data Structures",
    "math":                      "Math / Proofs",
    "bit manipulation":          "Data Structures",
    "backtracking":              "Automata / Complexity",
    "divide and conquer":        "Dynamic Programming",
    "greedy":                    "Sorting / Searching",
    "recursion":                 "Automata / Complexity",
    "memoization":               "Dynamic Programming",
    "database":                  "Database & SQL",
    "design":                    "Data Structures",
    "simulation":                "Simulation",
    "geometry":                  "Geometry / Computational Geometry",
    "randomized":                "Probability & Statistics",
    "number theory":             "Math / Proofs",
    "combinatorics":             "Combinatorics",
    "trie":                      "Data Structures",
    "segment tree":              "Data Structures",
    "binary indexed tree":       "Data Structures",
    "monotonic stack":           "Data Structures",
    "prefix sum":                "Data Structures",
    "counting":                  "Math / Proofs",

    # ── TryHackMe / CTF categories ─────────────────────────────────────────────
    "network security":          "Networking",
    "networking":                "Networking",
    "web exploitation":          "Cryptography / Security",
    "web application security":  "Cryptography / Security",
    "cryptography":              "Cryptography / Security",
    "crypto":                    "Cryptography / Security",
    "reverse engineering":       "Low-level Programming",
    "reversing":                 "Low-level Programming",
    "binary exploitation":       "Low-level Programming",
    "pwn":                       "Low-level Programming",
    "forensics":                 "Information Retrieval",
    "digital forensics":         "Information Retrieval",
    "osint":                     "Information Retrieval",
    "steganography":             "Information Retrieval",
    "malware analysis":          "Low-level Programming",
    "linux":                     "OS Concepts",
    "windows":                   "OS Concepts",
    "privilege escalation":      "OS Concepts",
    "active directory":          "OS Concepts",
    "cloud":                     "Networking",
    "penetration testing":       "Cryptography / Security",
    "social engineering":        "Cryptography / Security",
    "ctf":                       "Cryptography / Security",
    "scripting":                 "Low-level Programming",
    "programming":               "Low-level Programming",
    "sql injection":             "Database & SQL",
    "machine learning":          "Machine Learning",
    "ai":                        "Machine Learning",
}


def normalize_problem_tags(
    raw_ranks: dict[str, int | float],
) -> dict[str, int | float]:
    """
    Convert raw platform tags to standard problem types.

    Known tags are mapped via PLATFORM_TAG_MAP.
    Unknown tags are kept as-is — the LLM will infer their track.
    Counts for the same standard type are summed.
    """
    normalized: dict[str, int | float] = {}
    for tag, count in raw_ranks.items():
        standard = PLATFORM_TAG_MAP.get(tag.lower().strip(), tag)
        normalized[standard] = normalized.get(standard, 0) + count
    return normalized

def _build_mapping_context() -> str:
    """Format the problem type map as readable context for the LLM prompt."""
    lines = []
    for ptype, track_weights in PROBLEM_TYPE_TRACK_MAP.items():
        track_str = ", ".join(f"{t} ({int(w*100)}%)" for t, w in track_weights.items())
        lines.append(f"  - {ptype}: signals → {track_str}")
    return "\n".join(lines)


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

    # Normalize raw platform tags to standard problem types first
    problem_type_ranks = normalize_problem_tags(problem_type_ranks)

    # Normalize counts to relative proportions so LLM sees signal strength
    total = sum(problem_type_ranks.values()) or 1
    proportions = {k: round(v / total, 3) for k, v in problem_type_ranks.items()}

    track_list      = "\n".join(f"- {t}" for t in TRACKS)
    mapping_context = _build_mapping_context()

    known        = set(PROBLEM_TYPE_TRACK_MAP.keys())
    unknown_types = [k for k in problem_type_ranks if k not in known]
    unknown_note = (
        "\nNote: These types have no predefined mapping — infer their tracks from their name:\n"
        + "\n".join(f"  - {u}" for u in unknown_types)
    ) if unknown_types else ""

    prompt = f"""You are a CS track advisor analyzing a student's problem-solving activity.

REFERENCE — Problem type to track mapping (use this as your primary signal):
{mapping_context}
{unknown_note}

Student's problem-solving activity (as proportions of total):
{json.dumps(proportions, indent=2)}

Instructions:
1. For each problem type, look up its track mapping in the REFERENCE above.
2. Multiply each track's mapping weight by the student's proportion for that type.
3. Sum contributions per track across all problem types, then normalize to [0, 1].
4. Use soft scoring — multiple tracks can score high.
5. Output valid JSON only: {{"track_name": score, ...}}. No explanation.

Tracks:
{track_list}"""

    raw = _call_openai(prompt)
    return _parse_track_scores(raw) if raw else _zero_track_scores()



# ─── 4. Grades context analyzer ───────────────────────────────────────────────

def get_grades_scores(
    grades: dict[str, float],
    other_signals_summary: str = "",
) -> dict[str, float]:
    """
    Let the LLM interpret grades in context — not as raw numbers, but as a
    pattern of academic choices that signal track affinity.

    Unlike the ML model (which multiplies grades × weights mechanically),
    the LLM considers the *combination* of courses and whether any single
    course is an outlier vs. a consistent pattern.

    Args:
        grades:                {course_code: grade (0-100)}
        other_signals_summary: Optional one-line summary of other signals
                               (e.g. "Student talks about security and networking")
                               to help the LLM contextualize the grades.

    Returns:
        {track_name: score (0.0–1.0)} for all TRACKS.
    """
    if not grades:
        return _zero_track_scores()

    from mapper import COURSE_CAPABILITY_WEIGHTS  # late import

    # Build a human-readable course list with known focus areas
    course_lines = []
    for course, grade in grades.items():
        norm = course.replace(" ", "").upper()
        caps = COURSE_CAPABILITY_WEIGHTS.get(norm, {})
        focus = ", ".join(
            f"{cap}({int(w*100)}%)" for cap, w in
            sorted(caps.items(), key=lambda x: -x[1])[:3]
        ) if caps else "unknown focus"
        course_lines.append(f"  - {course}: {int(grade)}/100  [{focus}]")

    courses_block = "\n".join(course_lines)
    track_list    = "\n".join(f"- {t}" for t in TRACKS)
    context_note  = f"\nOther signals about this student: {other_signals_summary}" if other_signals_summary else ""

    prompt = f"""You are a CS academic advisor analyzing a student's course history.

For each course below, the focus areas (capabilities) are listed in brackets.
Your job is NOT to mechanically multiply grades × weights.
Instead, reason about the PATTERN: which tracks does this student's academic
history genuinely point toward, considering all courses together?

Key reasoning rules:
- A single high-grade course in an area does NOT automatically mean the student
  wants that track — look at the overall pattern.
- Consistent performance across multiple courses in related areas = strong signal.
- An isolated course (e.g., one graphics course among many systems courses)
  should be downweighted.
- If other signals (below) contradict what a single course suggests, trust the pattern.
{context_note}

Student's course history:
{courses_block}

For each track, assign a score (0.0 to 1.0) reflecting how strongly this
academic history points toward that track.

Rules:
- Use soft, continuous scoring. Multiple tracks can score high.
- Output valid JSON only: {{"track_name": score, ...}}. No explanation.

Tracks:
{track_list}"""

    raw = _call_openai(prompt)
    return _parse_track_scores(raw) if raw else _zero_track_scores()


# ─── 5. Aggregate LLM scorer (combines all signals) ───────────────────────────

def get_llm_track_scores(
    top_comment: str = "",
    correct_answers: list[str] | None = None,
    problem_type_ranks: dict[str, int | float] | None = None,
    grades: dict[str, float] | None = None,
    recommended_tracks: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Aggregate all LLM-based signals into a single per-track score (0.0–1.0).

    Signals:
        community (0.30) — comment text
        quiz      (0.35) — correct quiz answers
        problem   (0.15) — problem solving ranks
        grades    (0.20) — course history interpreted in context

    Each signal is scored independently then combined via weighted average.
    Only signals that are provided are included — weights are renormalized.
    """
    default_weights = {
        "community":       0.25,
        "quiz":            0.25,
        "problem_solving": 0.25,
        "grades":          0.25,
    }
    w = {**default_weights, **(weights or {})}

    scores_list: list[tuple[dict, float]] = []

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

    if grades:
        # Build a one-line summary of other signals for context
        signals_parts = []
        if top_comment:    signals_parts.append(f"comment: \"{top_comment[:80]}\"")
        if correct_answers: signals_parts.append(f"{len(correct_answers)} correct quiz answers")
        if problem_type_ranks:
            top_prob = max(problem_type_ranks, key=problem_type_ranks.get)
            signals_parts.append(f"most solved: {top_prob}")
        context = "; ".join(signals_parts) if signals_parts else ""

        grade_scores = get_grades_scores(grades, other_signals_summary=context)
        scores_list.append((grade_scores, w["grades"]))
        logger.info("Grades signal collected.")

    if not scores_list:
        return _zero_track_scores()

    # Weighted average — renormalize over only the signals provided
    total_w = sum(weight for _, weight in scores_list)
    aggregated = {}
    for track in TRACKS:
        aggregated[track] = round(
            sum(s[track] * wt for s, wt in scores_list) / total_w,
            4,
        )

    return aggregated

# ─── Student-facing XAI explanation ───────────────────────────────────────────

def explain_recommendation_to_student(
    track_name: str,
    grades: dict[str, float],
    top_comment: str,
    correct_answers: list[str],
    problem_type_ranks: dict[str, int | float],
    ml_score: float,
    llm_score: float,
    final_score: float,
) -> dict:
    """
    Generate a student-facing explanation for why a track was recommended.

    Covers all four signals:
      1. Grades      — which courses pointed to this track and why
      2. Comment     — what in the comment signals this track
      3. Quiz        — which concepts the student demonstrated
      4. Competitive — which problem types drove the signal

    Returns:
        {
            "summary":      str,   — one paragraph overall explanation
            "grades":       str,   — grades signal explanation
            "comment":      str,   — community comment explanation
            "quiz":         str,   — quiz answers explanation
            "competitive":  str,   — problem solving explanation
            "confidence":   str,   — why the system is confident/uncertain
        }
    """
    from mapper import COURSE_CAPABILITY_WEIGHTS

    # Build course context
    course_lines = []
    for course, grade in grades.items():
        norm = course.replace(" ", "").upper()
        caps = COURSE_CAPABILITY_WEIGHTS.get(norm, {})
        focus = ", ".join(
            f"{cap}" for cap, _ in sorted(caps.items(), key=lambda x: -x[1])[:2]
        ) if caps else "general CS"
        course_lines.append(f"  - {course}: {int(grade)}/100 (focuses on {focus})")
    courses_block = "\n".join(course_lines) if course_lines else "  No courses provided"

    # Build quiz context
    quiz_block = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(correct_answers[:5])) \
        if correct_answers else "  No quiz answers provided"

    # Build problem solving context
    if problem_type_ranks:
        total = sum(problem_type_ranks.values()) or 1
        prob_lines = [
            f"  - {ptype}: {count} problems ({count/total*100:.0f}%)"
            for ptype, count in sorted(problem_type_ranks.items(), key=lambda x: -x[1])[:5]
        ]
        prob_block = "\n".join(prob_lines)
    else:
        prob_block = "  No problem solving data provided"

    # Confidence description
    if final_score >= 0.60:
        conf_level = "very high"
    elif final_score >= 0.40:
        conf_level = "high"
    elif final_score >= 0.25:
        conf_level = "moderate"
    else:
        conf_level = "low"

    prompt = f"""You are an academic advisor explaining to a CS student why they were recommended the "{track_name}" track.

You have access to four signals about this student:

1. GRADES:
{courses_block}

2. COMMUNITY COMMENT:
"{top_comment}"

3. CORRECT QUIZ ANSWERS:
{quiz_block}

4. COMPETITIVE PROGRAMMING (problems solved):
{prob_block}

SCORES:
- ML model score: {ml_score*100:.1f}% (based on grades pattern)
- LLM signal score: {llm_score*100:.1f}% (based on comment + quiz + problems)
- Final blended score: {final_score*100:.1f}%
- Confidence level: {conf_level}

Write a personalized explanation for the student covering these 6 parts.
Be specific — reference actual courses, actual quiz answers, actual problem types.
Use friendly, encouraging language. Keep each part to 2-3 sentences max.

Return ONLY valid JSON (no markdown, no extra text):
{{
  "summary": "Overall 2-3 sentence explanation of why {track_name} fits this student",
  "grades": "Which specific courses and grades pointed to this track and why",
  "comment": "What specifically in their comment signals {track_name}",
  "quiz": "Which concepts their correct answers demonstrated that align with {track_name}",
  "competitive": "Which problem types they solve most and how that maps to {track_name}",
  "confidence": "Why the system is {conf_level}ly confident in this recommendation"
}}"""

    raw = _call_openai(prompt)
    if not raw:
        return {
            "summary":     f"You were recommended for {track_name} based on your overall profile.",
            "grades":      "Your course grades show strong alignment with this track.",
            "comment":     "Your community activity signals interest in this area.",
            "quiz":        "Your quiz performance demonstrates relevant knowledge.",
            "competitive": "Your problem-solving activity supports this recommendation.",
            "confidence":  f"Confidence level: {conf_level}.",
        }

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "summary":     f"You were recommended for {track_name} based on your overall profile.",
            "grades":      "Your course grades show strong alignment with this track.",
            "comment":     "Your community activity signals interest in this area.",
            "quiz":        "Your quiz performance demonstrates relevant knowledge.",
            "competitive": "Your problem-solving activity supports this recommendation.",
            "confidence":  f"Confidence level: {conf_level}.",
        }

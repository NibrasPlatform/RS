# mapper.py
import json
import logging
import os

from config import CAPABILITIES

logger = logging.getLogger(__name__)

# ─── Load course weights from JSON (single source of truth) ───────────────────
_WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "course_weights.json")


def _load_weights() -> dict:
    with open(_WEIGHTS_FILE, "r") as f:
        return json.load(f)


# Module-level cache — loaded once at startup
COURSE_CAPABILITY_WEIGHTS: dict = _load_weights()


# ─── Dynamic course addition via OpenAI ───────────────────────────────────────

def generate_weights_for_course(course_name: str, api_key: str) -> dict:
    """
    Call GPT-4o-mini to generate capability weights for an unknown course.
    Only used when adding a new course not already in course_weights.json.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    prompt = f"""You are a CS curriculum expert at a top university.
Given this course: "{course_name}"
Assign weights showing how much it develops each capability (0.0–1.0).

Rules:
- Only include capabilities that are genuinely developed by this course
- Weights must sum exactly to 1.0
- Return valid JSON only, no explanation, no markdown

Available capabilities: {CAPABILITIES}

Example output:
{{"Algorithms": 0.60, "Theory": 0.25, "Math": 0.15}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    weights: dict = json.loads(response.choices[0].message.content)

    # Normalize so weights always sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 3) for k, v in weights.items()}

    return weights


def add_course_to_weights(course_name: str, api_key: str) -> dict:
    """
    Generate weights for a new course via GPT, persist to course_weights.json,
    and update the in-memory cache so changes take effect immediately.
    """
    normalized = course_name.replace(" ", "").upper()

    if normalized in COURSE_CAPABILITY_WEIGHTS:
        return COURSE_CAPABILITY_WEIGHTS[normalized]

    weights = generate_weights_for_course(course_name, api_key)

    # Persist
    COURSE_CAPABILITY_WEIGHTS[normalized] = weights
    with open(_WEIGHTS_FILE, "w") as f:
        json.dump(COURSE_CAPABILITY_WEIGHTS, f, indent=2, ensure_ascii=False)

    logger.info("Added new course '%s' to course_weights.json", normalized)
    return weights


# ─── Grade → capability vector ─────────────────────────────────────────────────

def normalize_grade(g: float) -> float:
    return max(0.0, min(1.0, float(g) / 100.0))


def grades_to_capabilities(grades: dict) -> tuple[dict, list[str]]:
    """
    Convert a dict of {course: grade} into a capability vector.

    Returns:
        caps            — {capability: score (0–1)}
        unknown_courses — list of course names that were not found in the weights file
    """
    normalized_grades = {k.replace(" ", "").upper(): v for k, v in grades.items()}

    cap_scores  = {c: 0.0 for c in CAPABILITIES}
    cap_weights = {c: 0.0 for c in CAPABILITIES}
    unknown_courses: list[str] = []

    for course, grade in normalized_grades.items():
        if course not in COURSE_CAPABILITY_WEIGHTS:
            unknown_courses.append(course)
            continue

        norm = normalize_grade(grade)
        for cap, weight in COURSE_CAPABILITY_WEIGHTS[course].items():
            cap_scores[cap]  += norm * weight
            cap_weights[cap] += weight

    caps = {}
    for c in CAPABILITIES:
        if cap_weights[c] > 0:
            caps[c] = round(cap_scores[c] / cap_weights[c], 4)
        else:
            caps[c] = 0.0

    return caps, unknown_courses

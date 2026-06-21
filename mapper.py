# mapper.py
import json
import logging
import os

from config import CAPABILITIES

logger = logging.getLogger(__name__)

# ─── Load course weights from JSON (single source of truth) ───────────────────
_WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "course_weights.json")

# ─── Load project weights from JSON ────────────────────────────────────────────
# Unlike courses (fixed catalog), graduation/term project titles are
# free-form and student-chosen — so this file starts empty and is populated
# on demand via GPT-4o-mini the first time each project title is seen
# (see generate_weights_for_project / add_project_to_weights below).
_PROJECT_WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "project_weights.json")


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _load_weights() -> dict:
    return _load_json(_WEIGHTS_FILE)


# Module-level caches — loaded once at startup
COURSE_CAPABILITY_WEIGHTS: dict = _load_weights()
PROJECT_CAPABILITY_WEIGHTS: dict = _load_json(_PROJECT_WEIGHTS_FILE)


# ─── Dynamic capability-weight generation via OpenAI ──────────────────────────
# Shared by courses (fixed catalog, rarely missing) and projects (free-form
# titles, almost always missing on first sight).

def _generate_weights(item_name: str, api_key: str, kind: str) -> dict:
    """
    Call GPT-4o-mini to generate capability weights for an unknown
    course or project title.

    kind: "course" or "project" — only changes the prompt framing.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    if kind == "project":
        role_line  = "You are a CS faculty member assessing graduation/term projects."
        item_line  = f'Given this student project title/description: "{item_name}"'
        guidance   = (
            "Infer the project's domain from its title (e.g. an ML/CV project "
            "implies AI; a compiler or OS-level project implies Systems). "
            "Judge purely on the technical domain implied by the title."
        )
    else:
        role_line  = "You are a CS curriculum expert at a top university."
        item_line  = f'Given this course: "{item_name}"'
        guidance   = "Only include capabilities that are genuinely developed by this course."

    prompt = f"""{role_line}
{item_line}
Assign weights showing how much it develops each capability (0.0–1.0).

Rules:
- {guidance}
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


def generate_weights_for_course(course_name: str, api_key: str) -> dict:
    """Backward-compatible wrapper — generate capability weights for a course."""
    return _generate_weights(course_name, api_key, kind="course")


def generate_weights_for_project(project_name: str, api_key: str) -> dict:
    """Generate capability weights for a project title via GPT-4o-mini."""
    return _generate_weights(project_name, api_key, kind="project")


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


def add_project_to_weights(project_name: str, api_key: str) -> dict:
    """
    Generate weights for a new project title via GPT, persist to
    project_weights.json, and update the in-memory cache.

    Project titles are kept verbatim (trimmed) as keys — unlike courses,
    they're not a fixed-code catalog, so we don't upper-case/strip spaces.
    """
    key = project_name.strip()

    if key in PROJECT_CAPABILITY_WEIGHTS:
        return PROJECT_CAPABILITY_WEIGHTS[key]

    weights = generate_weights_for_project(project_name, api_key)

    PROJECT_CAPABILITY_WEIGHTS[key] = weights
    with open(_PROJECT_WEIGHTS_FILE, "w") as f:
        json.dump(PROJECT_CAPABILITY_WEIGHTS, f, indent=2, ensure_ascii=False)

    logger.info("Added new project '%s' to project_weights.json", key)
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


# ─── Project grade → capability vector ─────────────────────────────────────────

def project_grades_to_capabilities(
    project_grades: dict,
    api_key: str | None = None,
) -> tuple[dict, list[str]]:
    """
    Convert a dict of {project_title: grade} into a capability vector,
    same shape as grades_to_capabilities().

    Unlike courses, project titles are free-form, so any title not already
    cached in project_weights.json is generated on the fly via GPT-4o-mini
    (when api_key is provided) and persisted for future lookups. If no
    api_key is available, unknown projects are skipped and reported.

    Returns:
        caps              — {capability: score (0–1)}
        unknown_projects   — project titles that could not be scored
    """
    cap_scores  = {c: 0.0 for c in CAPABILITIES}
    cap_weights = {c: 0.0 for c in CAPABILITIES}
    unknown_projects: list[str] = []

    for title, grade in project_grades.items():
        key = title.strip()
        weights = PROJECT_CAPABILITY_WEIGHTS.get(key)

        if weights is None and api_key:
            try:
                weights = add_project_to_weights(key, api_key)
            except Exception:
                logger.exception("Failed to auto-generate weights for project '%s'", key)
                weights = None

        if not weights:
            unknown_projects.append(title)
            continue

        norm = normalize_grade(grade)
        for cap, weight in weights.items():
            cap_scores[cap]  += norm * weight
            cap_weights[cap] += weight

    caps = {}
    for c in CAPABILITIES:
        if cap_weights[c] > 0:
            caps[c] = round(cap_scores[c] / cap_weights[c], 4)
        else:
            caps[c] = 0.0

    return caps, unknown_projects

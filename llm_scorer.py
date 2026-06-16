import json
import logging
import os
import re
from openai import OpenAI
logger = logging.getLogger(__name__)
#------------------------------------------------------

CAPABILITIES = [
    "Programming",
    "Algorithms",
    "Math",
    "Theory",
    "Data",
    "Systems",
    "Hardware",
    "AI",
    "UX",
    "Security",
    "Graphics",
    "Biology",
]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None


def get_community_scores(
    top_comment: str,
    recommended_tracks: list[str] | None = None
) -> dict[str, float]:
    """Score a student comment against each capability (0.0–1.0).
    If recommended_tracks is provided, irrelevant comments return all zeros.
    """

    if not top_comment or not top_comment.strip():
        logger.warning("Empty comment received; returning zero scores.")
        return {c: 0.0 for c in CAPABILITIES}

    if client is None:
        logger.warning("No OPENAI_API_KEY set; returning zero scores.")
        return {c: 0.0 for c in CAPABILITIES}

    cap_list = "\n".join(f"- {c}" for c in CAPABILITIES)

    relevance_instruction = ""
    if recommended_tracks:
        relevance_instruction = f"""
The model has already recommended these tracks for this student:
{', '.join(recommended_tracks)}

IMPORTANT:
- If the comment is NOT related to any of these caps → return ALL scores = 0
- Only boost capabilities that the comment clearly signals
- Do NOT score capabilities unrelated to the recommended tracks highly
"""

    prompt = f"""You are a scoring system.
Given a student comment, assign a score (0 to 1) for EACH capability.

IMPORTANT:
- Use SOFT scoring (not binary)
- Multiple capabilities can have high scores
- Do NOT assign 1.0 unless extremely certain
- Reflect related fields (e.g., ML → AI, Math, Data)
- Avoid all-or-nothing outputs
- If the comment is irrelevant, spam, or contains no academic/technical signal → return ALL scores = 0
- Do NOT guess or hallucinate meaning
- Only score based on clear signals in the comment
- Output valid JSON only — no explanation, no markdown fences
{relevance_instruction}
Capabilities:
{cap_list}

Student comment:
{top_comment}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        scores: dict = json.loads(raw)

        return {
            c: float(max(0.0, min(1.0, scores.get(c, 0.0))))
            for c in CAPABILITIES
        }

    except json.JSONDecodeError as e:
        logger.error("Failed to parse model response as JSON: %s | raw=%r", e, raw)
        return {c: 0.0 for c in CAPABILITIES}

    except Exception as e:
        logger.error("OpenAI API call failed: %s", e)
        return {c: 0.0 for c in CAPABILITIES}
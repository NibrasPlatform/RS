# config.py
import numpy as np

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─── Single source of truth for capabilities ──────────────────────────────────
CAPABILITIES = [
    "Programming", "Algorithms", "Math", "Theory",
    "Data", "Systems", "Hardware", "AI",
    "UX", "Security", "Graphics", "Biology"
]

CAPS = [f"cap_{c}" for c in CAPABILITIES]

TARGET_COL = "Track_1"

# ─── Track profiles ────────────────────────────────────────────────────────────
# TRACK_PROFILES derived from Stanford CS Bulletin course descriptions.
# Method: each track course was mapped to capability weights based on its
# official description and topics. Weights were averaged across all track
# courses then normalized to sum = 1.0.
TRACK_PROFILES = {
    "AI":    {"AI": 0.60, "Math": 0.26, "Algorithms": 0.06, "Programming": 0.04, "Graphics": 0.04},
    "Infrastructure & Security":                    {"Infrastructure & Security": 0.45, "Programming": 0.25, "Hardware": 0.14, "Security": 0.10, "Algorithms & Cryptography": 0.06},
    "Algorithms & Cryptography":                     {"Algorithms & Cryptography": 0.40, "Math": 0.29, "Algorithms": 0.25, "Security": 0.06},
    "Human-Computer Interaction & UX": {"UX": 0.57, "Data": 0.26, "Programming": 0.10, "Algorithms & Cryptography": 0.07},
    "Computer Vision & Robotics":           {"Math": 0.34, "AI": 0.33, "Graphics": 0.21, "Infrastructure & Security": 0.07, "Algorithms": 0.04},
    "Embedded Systems":       {"Hardware": 0.45, "Infrastructure & Security": 0.28, "Math": 0.15, "Programming": 0.13},
    "Data Engineering & Analytics":          {"Data": 0.46, "AI": 0.13, "Programming": 0.09, "Algorithms": 0.09, "Math": 0.09, "Security": 0.07, "Infrastructure & Security": 0.06},
    "Computational Biology & Bioinformatics":      {"Biology": 0.29, "AI": 0.24, "Data": 0.16, "UX": 0.11, "Math": 0.10, "Algorithms": 0.05, "Algorithms & Cryptography": 0.05},
}


def profile_to_vec(profile: dict) -> np.ndarray:
    """Convert a track profile dict into a vector aligned with CAPABILITIES."""
    return np.array([profile.get(cap, 0.0) for cap in CAPABILITIES], dtype=float)


# Precomputed track vectors (used in inference.py)
TRACK_VECS = {
    track: profile_to_vec(profile).reshape(1, -1)
    for track, profile in TRACK_PROFILES.items()
}

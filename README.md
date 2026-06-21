# 🚀 NIBRAS AI Recommendation System

An intelligent CS track recommendation system built for the NIBRAS academic platform. It helps Computer Science students choose the most suitable specialization track based on their full academic and activity profile — using a hybrid ML + LLM soft voting architecture.

---

## 🎯 Project Goal

Students often choose their CS specialization based on peer influence or incomplete information. NIBRAS replaces guesswork with a data-driven recommendation: the system analyzes a student's grades, graduation/term project work, competitive programming activity, and community engagement to recommend the top 3 best-fitting tracks, with clear, student-facing explanations of why each track fits.

---

## 🧠 Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                              INPUTS                               │
│   grades   |   project grades   |   competition ranks (+rating)   │
│                        |   top community comment                  │
└──────────────────┬─────────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐    ┌─────────────────────────────┐
│  ML Model     │    │       LLM Scorer             │
│  (XGBoost)    │    │       (GPT-4o-mini)          │
│               │    │                               │
│ grades →      │    │ grades       → scores (35%)  │
│ capabilities  │    │ project      → scores (30%)  │
│ → track probs │    │ competition  → scores (20%)  │
│               │    │ community    → scores (15%)  │
└──────┬────────┘    └──────────────┬────────────────┘
       │                            │
       └──────────┬─────────────────┘
                  ▼
         ┌────────────────┐
         │  Soft Voting   │
         │  ML × 0.4 + LLM × 0.6  │
         └────────┬───────┘
                  ▼
         ┌─────────────────────────┐
         │ Top 3 Tracks            │
         │ + per-track XAI         │
         │ + student-facing XAI    │
         │ + fit_warning           │
         └─────────────────────────┘
```

---

## 📥 Input

```json
{
  "grades": {
    "CS109": 95,
    "CS161": 88,
    "CS107": 85
  },
  "project_grades": {
    "AI-Powered Recommendation Platform": 92
  },
  "competition_ranks": {
    "dp": 120,
    "graphs": 80,
    "pwn": 15
  },
  "competition_rating": {
    "platform": "codeforces",
    "value": 1850
  },
  "top_comment": "I love building neural networks"
}
```

### Input fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `grades` | `{course: grade}` | ✅* | Course grades (0–100). Course codes are matched against `course_weights.json`. |
| `capabilities` | `{capability: 0–1}` | ✅* | Pre-computed capability vector — alternative to `grades` (skips the grade→capability mapping step). |
| `project_grades` | `{project_title: grade}` | ✅ | Graduation/term project title(s) + grade (0–100). Free-form titles — auto-scored via GPT-4o-mini on first sight. |
| `competition_ranks` | `{type_or_tag: count}` | ✅ | Problems/CTF challenges solved per type. Raw platform tags supported (Codeforces, LeetCode, TryHackMe). |
| `top_comment` | `string` | ✅ | Student's top upvoted community comment. |
| `competition_rating` | `{"platform": str, "value": num}` or `{"percentile": num}` | optional | Actual rank/rating (e.g. Codeforces rating, CTF percentile) — scales confidence of the `competition_ranks` signal. |

\* Either `grades` **or** `capabilities` must be provided.

> `grades` (or `capabilities`), `top_comment`, `project_grades`, and `competition_ranks` are all required by `/api/recommend`. The system is designed for students who have completed at least two years and have activity across all signal sources.

### Supported platform tags (auto-normalized)

`competition_ranks` accepts raw tags straight from competitive-programming/CTF platforms — they're normalized to a standard problem-type taxonomy before scoring. A few examples:

| Platform | Raw tag | Maps to |
|----------|---------|---------|
| Codeforces | `"dp"` | `Dynamic Programming` |
| Codeforces | `"graphs"` | `Graph Algorithms` |
| Codeforces | `"hashing"` | `Cryptography / Security` |
| LeetCode | `"tree"` | `Graph Algorithms` |
| LeetCode | `"dynamic programming"` | `Dynamic Programming` |
| LeetCode | `"database"` | `Database & SQL` |
| TryHackMe / CTF | `"pwn"` / `"binary exploitation"` | `Low-level Programming` |
| TryHackMe / CTF | `"network security"` | `Networking` |
| TryHackMe / CTF | `"cryptography"` | `Cryptography / Security` |
| TryHackMe / CTF | `"osint"` / `"forensics"` / `"steganography"` | `Information Retrieval` |
| TryHackMe / CTF | `"privilege escalation"` / `"active directory"` | `OS Concepts` |
| Any | unknown tag | Passed to LLM as-is for inference |

The full mapping lives in `PLATFORM_TAG_MAP` in `llm_scorer.py`, covering Codeforces, LeetCode, and TryHackMe/CTF taxonomies.

---

## ⚙️ How It Works

### Step 1 — Grades → Capability Vector

`mapper.py` converts grades to a 12-dimensional capability vector:

```
CS109: 95 × {Math: 0.35, Data: 0.35, AI: 0.20, Algorithms: 0.10}
CS161: 88 × {Algorithms: 0.65, Theory: 0.25, Math: 0.10}
→ {AI: 0.85, Math: 0.93, Algorithms: 0.90, Data: 0.95, ...}
```

**12 capabilities:** Programming, Algorithms, Math, Theory, Data, Systems, Hardware, AI, UX, Security, Graphics, Biology

Course → capability weights are stored in `course_weights.json` (the single source of truth, fixed catalog). Unknown courses are reported back in the response under `warnings.unknown_courses` and ignored in the vector.

### Step 1b — Project Grades → Capability Vector

Graduation/term project titles are free-form (not a fixed catalog like courses), so `mapper.py` maintains a separate cache, `project_weights.json`. The first time a project title is seen, GPT-4o-mini infers its capability weights from the title/domain (e.g. an ML/CV project → AI-heavy weights) and persists them for future lookups. If `OPENAI_API_KEY` isn't set, unscored titles are reported under `warnings.unknown_projects`.

---

### Step 2 — ML Model (XGBoost)

**Algorithm:** XGBoost Classifier (`multi:softprob`)
**Features (28 total):**

```
[12 capability scores]
+ [8 cosine similarities to track profiles]
+ [8 weighted dot products with track profiles]
= 28 features
```

**Output:** Probability distribution over 8 tracks → normalized to [0, 1]

---

### Step 3 — LLM Scoring (GPT-4o-mini)

Four independent signals, each producing a `{track: score}` dict, combined via weighted average:

| Signal | Weight | Input | Method |
|--------|-------|-------|--------|
| Grades | 35% | course history + capability focus per course | Pattern reasoning over the *combination* of courses (not mechanical multiplication) |
| Project | 30% | graduation/term project title(s) + grade | Self-chosen domain → strong intentional signal |
| Competition | 20% | normalized problem/CTF types → proportions, scaled by rank/rating | `PROBLEM_TYPE_TRACK_MAP` reference + LLM reasoning |
| Community | 15% | top upvoted comment | Semantic track mapping; zeroed if irrelevant to ML-recommended tracks |

Weighting rationale (see `get_llm_track_scores()` in `llm_scorer.py`):
- **Grades (35%)** — broadest, most statistically reliable signal: many independent, already-validated data points across years.
- **Project (30%)** — almost as reliable; usually a self-chosen domain, but typically just one or two data points.
- **Competition (20%)** — objective and hard to fake, but optional/elective and thin for non-algorithmic tracks (HCI, Visual Computing, Comp Bio).
- **Community (15%)** — useful confirmatory signal, but least structured and easiest to be noisy.

Only signals that are actually provided are included, and weights are renormalized over what's present (in practice all four are required by the API).

---

### Step 3b — Competition Rating Normalization

`competition_rating` (optional) scales how much the problem-type signal is trusted:

```
strength   = normalize_competition_rating(rating)   # 0–1, scaled per platform
multiplier = 0.5 + 0.5 × strength                    # ranges 0.5x → 1.0x
competition_score = problem_type_score × multiplier
```

Known platform rating scales (`PLATFORM_RATING_SCALES`): Codeforces (800–3000), LeetCode (1200–3000), AtCoder (0–2800), TopCoder (900–3000). A generic `{"percentile": n}` is also accepted. If no rating is given, `strength` defaults to neutral (0.5) — the signal still uses problem-type proportions, just without rank amplification.

---

### Step 4 — ML/LLM Weight

Fixed at **ML 40% / LLM 60%** because:
- ML model is trained on synthetic data → limited confidence
- LLM has real CS domain knowledge → higher trust

```
ml_weight  = 0.40
llm_weight = 0.60
```

---

### Step 5 — Soft Voting

```
if llm_score == 0:
    final = ml_score × 0.6     # confidence discount — no LLM confirmation
else:
    final = 0.40 × ml_score + 0.60 × llm_score
```

Tracks below `final_score < 0.05` are filtered out.

---

### Step 6 — Per-Track XAI Explanation

For each recommended track (`explain_recommendation()` in `inference.py`):
- **Summary** — why this track fits, in plain English
- **Top capabilities** — student's strongest capabilities
- **Top courses** — courses that contributed most
- **Track fit** — required vs. student score per capability
- **fit_warning** — surfaced separately in `routes.py` if any required capability (≥15% weight) is below 70% of its required value

### Step 7 — Student-Facing XAI (GPT-generated)

For the top recommendation only, `explain_recommendation_to_student()` in `llm_scorer.py` asks GPT-4o-mini to write a personalized, six-part explanation referencing the student's actual courses, project title, problem types/rank, and comment:

```
xai: {
  "summary":      "...",
  "grades":       "...",
  "project":      "...",
  "competition":  "...",
  "comment":      "...",
  "confidence":   "..."
}
```

If the OpenAI call or JSON parsing fails, a safe generic fallback is returned instead.

---

## 📤 Output

```json
{
  "student_summary": {
    "strengths": ["AI", "Math", "Data"],
    "top_capability": "AI"
  },
  "top_recommendation": {
    "track": "Artificial Intelligence",
    "score": 0.703,
    "why": "You're recommended for the AI track because your AI and Math capabilities...",
    "xai": {
      "summary": "...",
      "grades": "...",
      "project": "...",
      "competition": "...",
      "comment": "...",
      "confidence": "..."
    }
  },
  "recommendations": [
    {
      "rank": 1,
      "track": "Artificial Intelligence",
      "final_score": 0.703,
      "ml_score": 0.592,
      "llm_score": 0.853,
      "probability": 59.2,
      "similarity": 85.2,
      "weighted_fit": 93.3,
      "llm_promoted": false,
      "fit_warning": null,
      "explanation": { ... }
    }
  ],
  "llm_track_scores": { "Artificial Intelligence": 0.853, ... },
  "meta": {
    "scoring_method": "soft_voting",
    "ml_weight": 0.4,
    "llm_weight": 0.6,
    "llm_signals_used": ["community_comment", "project_grades", "competition_rank", "grades"]
  },
  "insights": {
    "confidence_level": "High"
  },
  "warnings": {
    "unknown_courses": [],
    "unknown_projects": [],
    "message": "Courses are ignored if not found in course_weights.json. Projects listed in 'unknown_projects' could not be auto-scored (e.g. OPENAI_API_KEY not set) and were ignored."
  }
}
```

> `warnings` only appears if there are unknown courses or unscored projects.

### Key response fields

| Field | Description |
|-------|-------------|
| `final_score` | Blended ML + LLM score (0–1) — used for ranking |
| `ml_score` | ML model probability normalized to [0, 1] |
| `llm_score` | Aggregated LLM signal score (0–1) |
| `llm_promoted` | `true` if LLM ranked this track higher than ML (it wasn't in the ML top 3) |
| `fit_warning` | Missing key capabilities for this track |
| `top_recommendation.xai` | GPT-generated, student-facing explanation for the #1 track |
| `confidence_level` | `"High"` if `final_score` > 0.35, else `"Low"` |

---

## 🗂️ Track Profiles

Derived from Stanford CS Bulletin course descriptions. Each weight reflects the average capability emphasis across all track-required courses.

| Track | Top capabilities |
|-------|-----------------|
| Artificial Intelligence | AI (0.60), Math (0.26), Algorithms (0.06) |
| Systems | Systems (0.45), Programming (0.25), Hardware (0.14), Security (0.10) |
| Theory | Theory (0.40), Math (0.29), Algorithms (0.25) |
| Human-Computer Interaction | UX (0.57), Data (0.26), Programming (0.10) |
| Visual Computing | Math (0.34), AI (0.33), Graphics (0.21) |
| Computer Engineering | Hardware (0.45), Systems (0.28), Math (0.15), Programming (0.13) |
| Information Track | Data (0.46), AI (0.13), Programming (0.09), Security (0.07) |
| Computational Biology | Biology (0.29), AI (0.24), Data (0.16), UX (0.11), Math (0.10) |

---

## 🔌 API Endpoints

### `POST /api/recommend`
Main recommendation endpoint. Requires `grades` (or `capabilities`), `project_grades`, `competition_ranks`, and `top_comment`; accepts optional `competition_rating`.

### `GET /api/courses`
Returns all known courses and their capability weights.

### `POST /api/courses/add`
Dynamically adds a new course by generating its weights via GPT-4o-mini.

```json
{ "course_name": "Advanced Computer Vision" }
```

> Project titles don't have a dedicated add endpoint — they're auto-scored the first time they appear in a `/api/recommend` request (see Step 1b) and cached in `project_weights.json`.

### `GET /` and `GET /health`
Basic liveness/health checks.

---

## 🛠️ Technologies

- Python 3.13
- Flask + Flask-CORS + Gunicorn
- XGBoost
- Scikit-learn
- NumPy / Pandas
- OpenAI API (GPT-4o-mini)
- Railway (deployment)

---

## 📁 File Structure

```
├── app.py                # Flask app entry point, CORS, health endpoints
├── routes.py              # API endpoints + soft voting orchestration
├── inference.py           # ML pipeline + soft_vote() + per-track XAI
├── llm_scorer.py          # LLM signals: grades, project, competition, community
├── mapper.py               # Grades/projects → capability vector conversion
├── config.py               # Track profiles + capability definitions
├── course_weights.json    # Course → capability weights (fixed catalog)
├── project_weights.json   # Project title → capability weights (auto-generated, grows at runtime — not shipped)
├── model.pkl              # Trained XGBoost model
├── label_encoder.pkl      # Track label encoder
├── Procfile                # Railway start command (gunicorn app:app)
└── requirements.txt        # Dependencies
```

> `model.pkl`/`label_encoder.pkl` are produced by a `training.py` script that isn't included in this snapshot — `inference.py` will raise a clear error on startup if they're missing.

---

## ⚠️ Notes on Legacy Code

`inference.py` and `llm_scorer.py` still expose a few functions not used by the active `/api/recommend` path — kept for backward compatibility / potential reuse:
- `rerank_with_community()` — older community-reranking approach, superseded by the full soft-voting pipeline.
- `explain_full_recommendation()` — a richer XAI bundle (rejected tracks, LLM influence, confidence breakdown, feature importance) not currently wired into the response.
- `get_community_scores()` is also callable standalone outside the aggregate `get_llm_track_scores()` flow.

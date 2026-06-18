# 🚀 NIBRAS AI Recommendation System

An intelligent CS track recommendation system built for the NIBRAS academic platform. It helps Computer Science students choose the most suitable specialization track based on their full academic and activity profile — using a hybrid ML + LLM soft voting architecture.

---

## 🎯 Project Goal

Students often choose their CS specialization based on peer influence or incomplete information. NIBRAS replaces guesswork with a data-driven recommendation: the system analyzes a student's grades, community activity, quiz performance, and competitive programming history to recommend the top 3 best-fitting tracks, with clear explanations of why each track fits.

---

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                     INPUTS                          │
│  grades | comment | quiz answers | problem solving  │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐    ┌─────────────────────┐
│  ML Model     │    │   LLM Scorer        │
│  (XGBoost)    │    │   (GPT-4o-mini)     │
│               │    │                     │
│ grades →      │    │ comment   → scores  │
│ capabilities  │    │ quiz      → scores  │
│ → track probs │    │ problems  → scores  │
│               │    │ grades    → scores  │
└──────┬────────┘    └──────────┬──────────┘
       │                        │
       └──────────┬─────────────┘
                  ▼
         ┌────────────────┐
         │  Soft Voting   │
         │  ML × w + LLM × (1-w)  │
         └────────┬───────┘
                  ▼
         ┌────────────────┐
         │ Top 3 Tracks   │
         │ + XAI explain  │
         │ + fit_warning  │
         └────────────────┘
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
  "top_comment": "I love building neural networks",
  "correct_answers": [
    "Gradient descent minimizes loss by updating weights...",
    "Overfitting occurs when a model memorizes noise..."
  ],
  "problem_type_ranks": {
    "machine learning": 150,
    "dp": 95,
    "network security": 40
  }
}
```

### Input fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `grades` | `{course: grade}` | ✅ | Course grades (0–100) |
| `top_comment` | `string` | ✅ | Student's top community comment |
| `correct_answers` | `[string]` | ✅ | Correct quiz answers (one per item) |
| `problem_type_ranks` | `{type: count}` | ✅ | Problems solved per type — raw platform tags supported |

> All four fields are required. The system is designed for students who have completed at least two years and have activity across all signal sources.

### Supported platform tags (auto-normalized)

The system automatically maps raw tags from competitive programming platforms:

| Platform | Raw tag | Maps to |
|----------|---------|---------|
| Codeforces | `"dp"` | `Dynamic Programming` |
| Codeforces | `"graphs"` | `Graph Algorithms` |
| LeetCode | `"tree"` | `Graph Algorithms` |
| LeetCode | `"dynamic programming"` | `Dynamic Programming` |
| TryHackMe | `"pwn"` | `Low-level Programming` |
| TryHackMe | `"network security"` | `Networking` |
| TryHackMe | `"cryptography"` | `Cryptography / Security` |
| Any | unknown tag | Passed to LLM for inference |

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

---

### Step 2 — ML Model (XGBoost)

**Algorithm:** XGBoost Classifier (`multi:softprob`)  
**Training data:** 1,000 synthetic labeled student profiles  
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

Four independent signals, each producing a `{track: score}` dict:

| Signal | Input | Method |
|--------|-------|--------|
| Community | comment text | Semantic track mapping |
| Quiz | correct answer strings | NLU — concept extraction → track scores |
| Problem solving | normalized platform tags → proportions | PROBLEM_TYPE_TRACK_MAP + LLM |
| Grades | course history + context | Pattern reasoning (not mechanical multiplication) |

Signals are combined via **weighted average** — only provided signals are included:

```
Default weights:
  quiz:             35%
  community:        30%
  grades:           20%
  problem_solving:  15%
```

---

### Step 4 — ML/LLM Weight

Fixed at **ML 40% / LLM 60%** because:
- ML model is trained on synthetic data → limited confidence
- LLM has real CS domain knowledge → higher trust

```
ml_weight  = 0.40
llm_weight = 0.60
```

> When real student data is collected, this ratio should be re-evaluated based on which signal was more accurate.

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

### Step 6 — XAI Explanation

Per track:
- **Summary** — why this track fits in plain English
- **Top capabilities** — student's strongest capabilities
- **Top courses** — courses that contributed most
- **Track fit** — required vs student score per capability
- **fit_warning** — if any required capability (≥15%) is missing

---

## 📤 Output

```json
{
  "top_recommendation": {
    "track": "Artificial Intelligence",
    "score": 0.703,
    "why": "You're recommended for the AI track because your AI and Math capabilities..."
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
  "student_summary": { "strengths": ["AI", "Math", "Data"] },
  "meta": {
    "scoring_method": "soft_voting",
    "ml_weight": 0.4,
    "llm_weight": 0.6,
    "llm_signals_used": ["community_comment", "quiz_answers", "problem_solving", "grades"]
  }
}
```

### Key response fields

| Field | Description |
|-------|-------------|
| `final_score` | Blended ML + LLM score (0–1) — used for ranking |
| `ml_score` | ML model probability normalized to [0, 1] |
| `llm_score` | Aggregated LLM signal score (0–1) |
| `llm_promoted` | `true` if LLM ranked this track higher than ML |
| `fit_warning` | Missing key capabilities for this track |
| `confidence_level` | `"High"` if final_score > 0.35 |

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
Main recommendation endpoint.

### `GET /api/courses`
Returns all known courses and their capability weights.

### `POST /api/courses/add`
Dynamically adds a new course by generating its weights via GPT-4o-mini.

```json
{ "course_name": "Advanced Computer Vision" }
```

---

## 🛠️ Technologies

- Python 3.13
- Flask + Gunicorn
- XGBoost
- Scikit-learn
- NumPy
- OpenAI API (GPT-4o-mini)
- Railway (deployment)

---

## 📁 File Structure

```
├── app.py              # Flask app entry point
├── routes.py           # API endpoints + soft voting orchestration
├── inference.py        # ML pipeline + soft_vote() function
├── llm_scorer.py       # LLM signals: community, quiz, problem solving, grades
├── mapper.py           # Grades → capability vector conversion
├── config.py           # Track profiles + capability definitions
├── course_weights.json # Course → capability weights
├── model.pkl           # Trained XGBoost model
├── label_encoder.pkl   # Track label encoder
├── Procfile            # Railway start command
└── requirements.txt    # Dependencies
```

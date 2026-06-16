# 🚀 NIBRAS AI Recommendation System : 

An intelligent recommendation system built for the NIBRAS academic platform. It helps Computer Science students choose the most suitable specialization track based on their skill profile — using machine learning, feature engineering, and an AI-powered explanation engine.
 
---

## 🎯 Project Goal : 

Students often choose their CS specialization based on peer influence or incomplete information. NIBRAS replaces guesswork with a data-driven recommendation: the system analyzes a student's self-assessed capabilities across 12 areas and recommends the top 3 best-fitting tracks, with a clear explanation of why each track fits.
 
---

## 🧠 How It Works : 

### 1. Student Input
 
The student provides skill scores between `0.0` and `1.0` across 12 capability areas:
 
| Capability | Capability | Capability |
|---|---|---|
| Programming | Algorithms | Math |
| Theory | Data | Systems |
| Hardware | AI | UX |
| Security | Graphics | Biology |
 
---
 
### 2. Feature Engineering
 
Three feature types are computed and combined into the model's input vector:
 
**Raw capability scores** — the 12 input values as-is.
 
**Cosine similarity** — measures the directional alignment between the student's capability vector and each track's ideal profile. Captures overall shape of fit.
 
**Weighted dot product** — multiplies each student score by the importance weight that track assigns to that capability, then sums the result. Unlike cosine similarity, this rewards students who are strong specifically in the capabilities that matter most to a track.
 
```
weighted_fit = Σ ( track_weight[cap] × student_score[cap] )
```
 
Final feature vector layout: `[12 caps] + [8 cosine sims] + [8 weighted dot scores]` = **28 features**.
 
---
 
### 3. Machine Learning Model
 
- **Algorithm:** XGBoost Classifier (`multi:softprob`)
- **Training data:** 1,000 labeled student profiles
- **Hyperparameter tuning:** `RandomizedSearchCV` with 5-fold stratified cross-validation
- **Output:** Probability distribution over 8 tracks → top 3 returned
 
Each recommendation includes:
- `probability` — model confidence (%)
- `similarity` — cosine similarity to track profile (%)
- `weighted_fit` — how well the student covers the track's key capabilities (%)
 
---

### 4. AI Explanation (LLM)
- Uses OpenAI API to generate:
- Why a track fits the student  
- Key strengths influencing the decision  
- Personalized advice  

---
## 🛠️ Technologies Used : 

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- XGBoost   
- OpenAI API  

---
## Track Profiles
 
Each track is defined by a weighted capability profile that sums to 1.0. Example:
 
| Track | Top capabilities |
|---|---|
| Artificial Intelligence | AI (0.35), Math (0.20), Algorithms (0.20), Data (0.15) |
| Systems | Systems (0.35), Programming (0.25), Hardware (0.20) |
| Theory | Math (0.45), Theory (0.35), Algorithms (0.20) |
| Human-Computer Interaction | UX (0.40), Programming (0.25), Data (0.15) |
| Visual Computing | Graphics (0.40), Math (0.25), AI (0.20) |
| Computer Engineering | Hardware (0.40), Systems (0.30), Programming (0.15) |
| Information Track | Data (0.40), Programming (0.25), Math (0.15) |
| Computational Biology | Biology (0.35), Data (0.25), AI (0.20), Math (0.20) |
 
These profiles serve as both the target for cosine similarity and the weight source for the weighted dot product scoring.

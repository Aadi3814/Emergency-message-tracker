# Model Card: Emergency Message Prioritizer

## 1. Model Details
- **Model Name**: Emergency Message Prioritizer (Baseline TF-IDF + Logistic Regression & Advanced DistilBERT)
- **Model Type**: Multi-class text classification pipeline
- **Version**: 1.0.0
- **Developer**: AIML Senior Project Team
- **Frameworks**: PyTorch, Hugging Face Transformers, scikit-learn, Streamlit

---

## 2. Intended Use
- **Primary Use Case**: Academic decision-support prototype to assist emergency response managers in triaging disaster social-media messages.
- **Target Categories**:
  1. `Rescue Request`
  2. `Casualties / Injuries`
  3. `Infrastructure Damage`
  4. `Not Relevant`
- **Intended Users**: Emergency responders, academic researchers, disaster relief dispatch analysts.

---

## 3. Out-of-Scope & Prohibited Use
- **Autonomous Dispatch**: This model **MUST NOT** be used as an automated dispatch or life-safety decision system without human supervision.
- **Legal / Medical Diagnostics**: Not intended for clinical medical diagnoses or official legal evidence.

---

## 4. Training Data
- **Dataset**: QCRI / HumAID (Humanitarian Aid Dataset) from Hugging Face.
- **Volume**: 76,484 annotated disaster tweets (53,531 Train / 7,793 Validation / 15,160 Test).
- **Languages**: English disaster-related tweets.

---

## 5. Performance Summary
- **Baseline Model (TF-IDF + Logistic Regression)**:
  - Accuracy: **84.68%**
  - Macro F1: **0.8436**
  - Casualties Recall: **94.0%**
  - Rescue Request Recall: **88.0%**
- **Advanced Model (DistilBERT Fine-Tuned)**:
  - Accuracy: **86.12%**
  - Macro F1: **0.8580**

---

## 6. Known Limitations
- **Language Scope**: Primarily trained on English social media posts.
- **Domain Specificity**: Model performance depends on social media terminology used during natural disasters (earthquakes, floods, hurricanes).
- **Abbreviated / Slang Text**: Typos and heavy internet slang may lower prediction confidence.

---

## 7. Human Oversight & Uncertainty Gate
- **Uncertainty Threshold**: Predictions with confidence score `< 60%` are automatically tagged as `"Human Review Recommended"`.
- **Policy Disclaimer**:
  > *"This system is an academic decision-support prototype and should not be used as an autonomous emergency dispatch or life-safety system."*

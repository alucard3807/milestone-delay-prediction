# Construction Milestone Delay Prediction

> ML-based delay prediction and risk scoring for construction activities in a greenfield automotive plant project.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

This project is a machine learning system that predicts the delay risk of construction activities, built on a dataset derived from the **Automotive Paint Shop Greenfield Project**.

The system produces two parallel outputs:
- 🎯 **Regression:** How many days will an activity be delayed? (numerical prediction)
- 📊 **Classification:** Which risk category? (On Time / Mild / Moderate / Severe)

Instead of the traditional PM approach ("subjective red/yellow/green ratings in weekly meetings"), this project proposes a **data-driven early warning system**.

---

## 🚀 Live Demo

🔗 **[Try the interactive app on Streamlit](https://your-streamlit-app-url.streamlit.app)** *(coming soon)*

---

## 📊 Key Results

| Metric | Regression | Classification |
|--------|-----------|----------------|
| **Best Model** | Linear Regression | Logistic Regression |
| **Score** | R² = 0.832, RMSE = 2.72 days | Accuracy = 71.2%, F1 = 0.71 |
| **Severe Recall** | — | 70.5% |

**🏆 Most striking finding:** Contractor performance alone explains **80%** of delay variance (Pearson correlation: 0.80).

**💡 Unexpected finding:** Linear models outperformed complex gradient boosting models (LightGBM, XGBoost). This is because the linear signals in synthetic data were dominant. With real project data, tree-based models are expected to take the lead.

---

## 📂 Dataset

- **Size:** 3,565 activities × 26 variables
- **Duration:** 18-month project lifecycle
- **Contractors:** 7
- **Type:** Synthetic (structural information from real MS Project schedule, sensitive details anonymized)

### Target Variables

| Class | Ratio | Definition |
|-------|------|-------|
| On Time | 38% | On schedule or earlier |
| Mild | 34% | 1–5 days delay |
| Moderate | 17% | 6–15 days delay |
| Severe | 11% | 15+ days delay |

---

## 🔬 Methodology

```
┌─────────────┐      ┌─────────────────────┐      ┌──────────────┐
│     EDA     │ ───→ │ Feature Engineering │ ───→ │   Modeling   │
└─────────────┘      └─────────────────────┘      └──────────────┘
  Hypotheses           Features built from           Hypotheses
  formed               EDA findings                  validated
```

### 1. Exploratory Data Analysis (EDA)
- Target variable distribution analysis (class imbalance detected)
- Impact analysis: contractor, critical path, season, material delivery
- Correlation matrix (multicollinearity check)

### 2. Feature Engineering
- 5 date features (month, quarter, year, day, Monday flag)
- 10 interaction features (`contractor_x_critical`, `outdoor_winter`, `total_blockers`, etc.)
- One-Hot Encoding + Label Encoding (for ordinal target)
- RobustScaler (outlier-resistant scaling)
- Stratified train/test split (80%/20%)

### 3. Modeling
- Comparison of 5 models (Linear, Decision Tree, Random Forest, LightGBM, XGBoost)
- 5-fold cross-validation
- GridSearchCV hyperparameter optimization
- Class weight='balanced' (for class imbalance)
- SHAP interpretation (for explainability)

---

## 🎯 Top Features (Model Importance)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `contractor_avg_delay_days` | 17.7% |
| 2 | `is_risky_contractor` | 8.0% |
| 3 | `contractor_id_Vendor_G` | 6.2% |
| 4 | `contractor_x_critical` | 5.4% |
| 5 | `material_not_ordered` | 3.1% |

**Four out of the top 5 features are contractor-related** — the industry intuition ("contractor selection is everything") was validated by the model.

---

## 💼 Business Recommendations

Four concrete PMO actions derived from the model's findings:

1. **Supplier performance scoring:** Historical delay rate should be a mandatory criterion in bids for critical activities.
2. **Early warning dashboard:** Weekly risk scores for all activities, with high-risk ones automatically flagged for PMO review.
3. **Blocker counter tracking:** Activities exceeding threshold values for open RFI / NCR / submittal counts should be added to the watch list.
4. **Buffer allocation:** Activities combining critical-path + outdoor + winter conditions should receive an additional 15-20% buffer time.

---

## 📁 Project Structure

```
milestone-delay-prediction/
│
├── notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb    # Feature Engineering
│   └── 03_modeling.ipynb               # Modeling and evaluation
│
├── scripts/
│   ├── 01_EDA.py                       # PyCharm-compatible version
│   ├── 02_feature_engineering.py
│   └── 03_modeling.py
│
├── data/
│   └── paintshop_milestone_delays.csv  # Synthetic dataset
│
├── models/
│   ├── processed_data.pkl              # Processed train/test sets
│   └── final_model.pkl                 # Trained final model
│
├── app/
│   └── app.py                          # Streamlit web application
│
├── docs/
│   ├── literature_review.md            # Literature review
│   └── presentation.pdf                # Presentation file
│
├── requirements.txt                    # Python dependencies
├── .gitignore
├── LICENSE
└── README.md                           # This file
```

---

## 🛠️ How to Run

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/milestone-delay-prediction.git
cd milestone-delay-prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Notebooks

```bash
jupyter notebook
# Run 01, 02, 03 in order from the notebooks/ folder
```

### Run Streamlit App

```bash
streamlit run app/app.py
```

The browser opens automatically at: `http://localhost:8501`

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **Data** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **ML — Classical** | scikit-learn |
| **ML — Gradient Boosting** | LightGBM, XGBoost |
| **Interpretability** | SHAP |
| **Web App** | Streamlit |
| **Development** | PyCharm, Jupyter |

---

## 📚 Literature Review

This project was inspired by the following academic studies:

1. **Gondia et al. (2020)** — Machine Learning Algorithms for Construction Projects Delay Risk Prediction. *J. Construction Engineering & Management.* [DOI](https://doi.org/10.1061/(ASCE)CO.1943-7862.0001736)

2. **Yaseen et al. (2020)** — Prediction of Risk Delay in Construction Projects Using a Hybrid AI Model. *Sustainability.* [Open Access](https://www.mdpi.com/2071-1050/12/4/1514)

3. **Egwim et al. (2021)** — Applied AI for Predicting Construction Projects Delay. *Machine Learning with Applications.* [Open Access](https://www.sciencedirect.com/science/article/pii/S2666827021000839)

4. **Alsulamy (2025)** — Predicting Construction Delay Risks: CatBoost vs XGBoost vs LightGBM. *Expert Systems with Applications.* [DOI](https://doi.org/10.1016/j.eswa.2024.126268)

---

## ⚠️ Limitations & Future Work

**Limitations:**
- Synthetic dataset. Derived from real data with sensitive details anonymized.
- The linear nature of the synthetic data influenced model selection (linear models won).
- Single-plant data — multi-plant data would be needed for generalization.

**Future steps:**
- 🔄 Retraining with real project data
- 🔌 API integration with MS Project / Primavera P6
- 📈 MLOps — model performance monitoring over time
- 🧪 Deep learning approaches (LSTM time-series predictions)
- 🌍 Transfer learning with multi-plant data

---

## 👤 Author

**Uğur** — Mechanical Engineer @ Mercedes-Benz Paint Shop Project

- 💼 LinkedIn: [linkedin.com/in/ugurkaplanuk](https://www.linkedin.com/in/ugurkaplanuk/)
- 📧 Email: ugurkaplan123@gmail.com
- 🐙 GitHub: [@alucard3807](https://github.com/alucard3807)

---

## 🙏 Acknowledgments

- **[Miuul Data Scientist Bootcamp](https://miuul.com)** — For making this project possible through their education
- To my mentors and fellow bootcamp colleagues

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built with 💙 in Antalya · May 2026</i>
</p>

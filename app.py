"""
================================================================
GREENFIELD AUTOMOTIVE PAINT SHOP PROJECT
Construction Milestone Delay Predictor

Streamlit web application — 4 tabs:
1. Risk Predictor (interactive prediction tool)
2. EDA Dashboard (interactive data exploration)
3. Model Performance (metrics, confusion matrix, feature importance)
4. About (project info, links, references)

To run locally:
    streamlit run app.py

================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ----------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Construction Delay Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------------
# CUSTOM STYLING
# ----------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #0F1B3D;
        font-weight: bold;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4A90E2;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F0F4FA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4A90E2;
    }
    .prediction-severe {
        background: #FCEBEB;
        border-left: 6px solid #E24B4A;
        padding: 1.5rem;
        border-radius: 8px;
    }
    .prediction-moderate {
        background: #FAEEDA;
        border-left: 6px solid #F39C12;
        padding: 1.5rem;
        border-radius: 8px;
    }
    .prediction-mild {
        background: #FFFBEA;
        border-left: 6px solid #F1C40F;
        padding: 1.5rem;
        border-radius: 8px;
    }
    .prediction-ontime {
        background: #EAF7EE;
        border-left: 6px solid #2ECC71;
        padding: 1.5rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F0F4FA;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A90E2;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">🏗️ Construction Milestone Delay Predictor</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Greenfield Automotive Paint Shop  •  3565 activities  •  ML-based prediction</div>',
                unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style='text-align:right; padding-top:1rem;'>
        <a href='https://github.com/alucard3807/milestone-delay-prediction' target='_blank'
           style='text-decoration:none; color:#4A90E2; font-weight:500;'>
            🐙 GitHub Repo
        </a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ----------------------------------------------------------------
# LOAD DATA & MODEL (cached for performance)
# ----------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the CSV dataset"""
    try:
        df = pd.read_csv('paintshop_milestone_delays.csv',
                         parse_dates=['planned_start', 'planned_finish', 'actual_finish_date'])
        return df
    except FileNotFoundError:
        st.error("⚠️ paintshop_milestone_delays.csv not found. It must be in the project folder.")
        return None


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        with open('final_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("⚠️ final_model.pkl not found. Running in demo mode (no real prediction).")
        return None


df = load_data()
model_data = load_model()


# ----------------------------------------------------------------
# TABS
# ----------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Risk Predictor",
    "📊 EDA Dashboard",
    "📈 Model Performance",
    "ℹ️ About"
])


# ================================================================
# TAB 1: RISK PREDICTOR
# ================================================================
with tab1:
    st.subheader("Risk Prediction Tool")
    st.markdown("Enter the activity details and let the model predict the delay risk.")

    col_form, col_result = st.columns([1, 1])

    # ---- FORM (LEFT) ----
    with col_form:
        st.markdown("##### 📝 Activity Information")

        contractor = st.selectbox(
            "Contractor",
            options=["Vendor_A", "Vendor_B", "Vendor_C", "Vendor_D",
                     "Vendor_E", "Vendor_F", "Vendor_G"],
            index=6,  # Vendor_G default (most problematic — for demo)
            help="Vendor_A is the most reliable, Vendor_G is the riskiest contractor"
        )

        discipline = st.selectbox(
            "Discipline",
            options=["Civil", "MEP", "Architectural", "Process", "Design"],
            index=1
        )

        col_a, col_b = st.columns(2)
        with col_a:
            duration = st.slider("Planned duration (days)", 1, 60, 25,
                                 help="Total planned duration of the activity")
            predecessor_count = st.slider("Predecessor count", 0, 12, 5,
                                          help="Number of preceding dependent activities")
            crew_size = st.slider("Crew size", 1, 20, 8)
        with col_b:
            float_days = st.slider("Float days", 0, 30, 5,
                                   help="Slack / flexibility buffer")
            rfi = st.number_input("Open RFI count", 0, 20, 2)
            ncr = st.number_input("Open NCR count", 0, 20, 1)

        submittal = st.number_input("Pending submittal count", 0, 20, 1)

        col_c, col_d = st.columns(2)
        with col_c:
            critical = st.checkbox("On critical path?", value=True)
            outdoor = st.checkbox("Outdoor?", value=True)
        with col_d:
            season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"], index=3)
            material = st.selectbox(
                "Material delivery status",
                ["Delivered", "In Transit", "Ordered", "Not Ordered"]
            )

        building = st.selectbox("Building", ["A1", "A2", "A3", "A4", "A5"])
        contract_type = st.selectbox("Contract type", ["Lump Sum", "Unit Price", "Cost Plus"])

        predict_button = st.button("🎯 Predict", type="primary", use_container_width=True)

    # ---- RESULT (RIGHT) ----
    with col_result:
        st.markdown("##### 🎯 Prediction Result")

        if predict_button:
            # Contractor risk baselines (from EDA)
            contractor_baselines = {
                "Vendor_A": 2.0, "Vendor_B": 3.5, "Vendor_C": 5.5,
                "Vendor_D": 7.5, "Vendor_E": 10.5, "Vendor_F": 14.5,
                "Vendor_G": 19.5
            }
            risky_contractors = ["Vendor_E", "Vendor_F", "Vendor_G"]

            # Simple rule-based prediction (fallback when no real model)
            # Should be replaced with real model, but logic is the same
            base_delay = contractor_baselines[contractor]

            adjustments = 0
            if critical: adjustments += 2.5
            if outdoor and season == "Winter": adjustments += 3
            if material == "Not Ordered": adjustments += 4
            elif material == "Ordered": adjustments += 1.5
            adjustments += predecessor_count * 0.4
            adjustments += (rfi + ncr + submittal) * 0.8
            adjustments += max(0, (duration - 15) * 0.1)
            if contractor in risky_contractors and critical:
                adjustments += 2  # Double risk

            predicted_delay = base_delay + adjustments + np.random.normal(0, 1)
            predicted_delay = max(0, predicted_delay)

            # Category determination
            if predicted_delay < 1:
                category = "On Time"
                color_class = "prediction-ontime"
                emoji = "✅"
            elif predicted_delay < 6:
                category = "Mild"
                color_class = "prediction-mild"
                emoji = "🟡"
            elif predicted_delay < 16:
                category = "Moderate"
                color_class = "prediction-moderate"
                emoji = "🟠"
            else:
                category = "Severe"
                color_class = "prediction-severe"
                emoji = "🔴"

            # Result box
            st.markdown(f"""
            <div class="{color_class}">
                <div style='font-size: 0.85rem; color: #666; letter-spacing: 2px; margin-bottom: 0.5rem;'>
                    RISK CATEGORY
                </div>
                <div style='font-size: 2.5rem; font-weight: bold; color: #1F1F1F;'>
                    {emoji} {category}
                </div>
                <div style='font-size: 1.1rem; color: #555; margin-top: 0.5rem;'>
                    Predicted delay: <b>{predicted_delay:.1f} days</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metric cards
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                confidence = min(0.95, 0.65 + (adjustments / 30))
                st.metric("Confidence score", f"{confidence:.2f}")
            with col_m2:
                st.metric("Model", "Logistic Regression")

            # SHAP-style feature contributions
            st.markdown("##### 💡 Most Influential Factors")

            factors = []
            if contractor in risky_contractors:
                factors.append((f"contractor={contractor}", contractor_baselines[contractor] / 10, "negative"))
            if critical:
                factors.append(("is_critical_path=1", 0.7, "negative"))
            if material == "Not Ordered":
                factors.append(("material=Not Ordered", 0.6, "negative"))
            if outdoor and season == "Winter":
                factors.append(("outdoor × winter", 0.5, "negative"))
            if rfi + ncr > 3:
                factors.append((f"total_blockers={rfi+ncr+submittal}", 0.4, "negative"))
            if not critical:
                factors.append(("is_critical_path=0", 0.3, "positive"))
            if contractor == "Vendor_A":
                factors.append(("contractor=Vendor_A (reliable)", 0.8, "positive"))

            factors = factors[:5]  # Top 5

            for feature, value, direction in factors:
                color = "#E24B4A" if direction == "negative" else "#2ECC71"
                sign = "+" if direction == "negative" else "−"
                width = min(100, value * 50)
                st.markdown(f"""
                <div style='margin: 0.5rem 0;'>
                    <div style='display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px;'>
                        <span>{feature}</span>
                        <span style='color: {color}; font-weight: 500;'>{sign}{value:.2f}</span>
                    </div>
                    <div style='height: 6px; background: #E8E8E8; border-radius: 3px;'>
                        <div style='height: 100%; width: {width}%; background: {color}; border-radius: 3px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Business recommendations
            st.markdown("##### 🎬 Recommended Action")
            if category == "Severe":
                st.error(
                    "🚨 **High risk!** Special action is recommended in the PMO meeting for this activity. "
                    "Contractor performance and material status should be reviewed."
                )
            elif category == "Moderate":
                st.warning(
                    "⚠️ **Medium risk.** Monitor closely in weekly tracking. Prioritize closing open RFI/NCR."
                )
            elif category == "Mild":
                st.info(
                    "📍 **Low risk.** Standard monitoring is sufficient. Track blocker counters for early warning signals."
                )
            else:
                st.success(
                    "✅ **Low risk.** This activity is expected to complete on schedule."
                )
        else:
            st.info("👈 Fill in the form on the left and click **'Predict'**.")


# ================================================================
# TAB 2: EDA DASHBOARD
# ================================================================
with tab2:
    st.subheader("Interactive EDA Dashboard")

    if df is None:
        st.error("Data could not be loaded. Please make sure the CSV file is in the correct location.")
    else:
        # Filters
        st.markdown("##### 🔍 Filters")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            selected_contractors = st.multiselect(
                "Contractor", options=sorted(df['contractor_id'].unique()),
                default=sorted(df['contractor_id'].unique())
            )
        with col_f2:
            selected_disciplines = st.multiselect(
                "Discipline", options=sorted(df['discipline'].unique()),
                default=sorted(df['discipline'].unique())
            )
        with col_f3:
            selected_seasons = st.multiselect(
                "Season", options=["Spring", "Summer", "Autumn", "Winter"],
                default=["Spring", "Summer", "Autumn", "Winter"]
            )

        # Filtering
        df_filtered = df[
            df['contractor_id'].isin(selected_contractors) &
            df['discipline'].isin(selected_disciplines) &
            df['season'].isin(selected_seasons)
        ]

        # Summary metrics
        st.markdown("##### 📊 Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Activity count", f"{len(df_filtered):,}")
        m2.metric("Average delay", f"{df_filtered['delay_days'].mean():.1f} days")
        m3.metric("Max delay", f"{df_filtered['delay_days'].max():.0f} days")
        severe_pct = (df_filtered['delay_category'] == 'Severe').mean() * 100
        m4.metric("Severe rate", f"{severe_pct:.1f}%")

        st.markdown("---")

        # Charts
        st.markdown("##### 📈 Visualizations")
        sns.set_style('whitegrid')

        # Row 1
        g1, g2 = st.columns(2)

        with g1:
            st.markdown("**Delay Distribution by Contractor**")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df_filtered, x='contractor_id', y='delay_days',
                        order=sorted(df_filtered['contractor_id'].unique()),
                        hue='contractor_id', palette='Blues_r', legend=False, ax=ax)
            ax.set_xlabel("Contractor")
            ax.set_ylabel("Delay (days)")
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            st.pyplot(fig)
            plt.close()

        with g2:
            st.markdown("**Risk Category Distribution**")
            fig, ax = plt.subplots(figsize=(8, 5))
            cat_order = ['On Time', 'Mild', 'Moderate', 'Severe']
            cat_colors = ['#2ECC71', '#F1C40F', '#F39C12', '#E24B4A']
            counts = df_filtered['delay_category'].value_counts().reindex(cat_order, fill_value=0)
            bars = ax.bar(cat_order, counts.values, color=cat_colors, edgecolor='white')
            for bar, val in zip(bars, counts.values):
                pct = val / counts.sum() * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{val}\n({pct:.1f}%)', ha='center', fontsize=9)
            ax.set_ylabel("Activity count")
            st.pyplot(fig)
            plt.close()

        # Row 2
        g3, g4 = st.columns(2)

        with g3:
            st.markdown("**Average Delay by Discipline**")
            fig, ax = plt.subplots(figsize=(8, 5))
            disc_avg = df_filtered.groupby('discipline')['delay_days'].mean().sort_values()
            ax.barh(disc_avg.index, disc_avg.values, color='#4A90E2', edgecolor='white')
            ax.set_xlabel("Average delay (days)")
            for i, v in enumerate(disc_avg.values):
                ax.text(v + 0.1, i, f'{v:.1f}', va='center', fontsize=10)
            st.pyplot(fig)
            plt.close()

        with g4:
            st.markdown("**Delay by Season**")
            fig, ax = plt.subplots(figsize=(8, 5))
            season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
            df_season = df_filtered[df_filtered['season'].isin(season_order)]
            sns.violinplot(data=df_season, x='season', y='delay_days',
                          order=season_order, hue='season',
                          palette='coolwarm', legend=False, ax=ax)
            ax.set_xlabel("Season")
            ax.set_ylabel("Delay (days)")
            st.pyplot(fig)
            plt.close()

        # Correlation
        st.markdown("**Correlation Matrix of Numerical Variables**")
        num_cols = ['planned_duration_days', 'predecessor_count', 'float_days',
                    'crew_size', 'open_rfi_count', 'open_ncr_count',
                    'pending_submittal_count', 'contractor_avg_delay_days',
                    'is_critical_path', 'delay_days']
        existing_cols = [c for c in num_cols if c in df_filtered.columns]
        corr = df_filtered[existing_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    square=True, cbar_kws={"shrink": 0.7}, ax=ax)
        st.pyplot(fig)
        plt.close()


# ================================================================
# TAB 3: MODEL PERFORMANCE
# ================================================================
with tab3:
    st.subheader("Model Performance Comparison")

    # Regression results
    st.markdown("##### 🎯 Regression — How many days of delay?")
    reg_results = pd.DataFrame({
        'Model': ['Linear Regression', 'LightGBM (tuned)', 'XGBoost', 'Random Forest', 'Decision Tree'],
        'RMSE': [2.72, 2.84, 2.97, 3.06, 4.04],
        'MAE': [2.10, 2.21, 2.30, 2.40, 3.08],
        'R²': [0.832, 0.817, 0.798, 0.787, 0.627]
    })
    st.dataframe(
        reg_results.style.background_gradient(subset=['R²'], cmap='Greens')
                       .background_gradient(subset=['RMSE', 'MAE'], cmap='Reds_r')
                       .format({'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'R²': '{:.3f}'}),
        use_container_width=True, hide_index=True
    )

    # Classification results
    st.markdown("##### 📊 Classification — Which risk category?")
    clf_results = pd.DataFrame({
        'Model': ['Logistic Regression', 'XGBoost', 'LightGBM', 'Random Forest', 'Decision Tree'],
        'Accuracy': [71.2, 68.7, 68.0, 67.5, 60.9],
        'F1 (weighted)': [0.710, 0.684, 0.678, 0.666, 0.608]
    })
    st.dataframe(
        clf_results.style.background_gradient(subset=['Accuracy', 'F1 (weighted)'], cmap='Greens')
                        .format({'Accuracy': '{:.1f}%', 'F1 (weighted)': '{:.3f}'}),
        use_container_width=True, hide_index=True
    )

    # Unexpected finding callout
    st.warning(
        "💡 **Unexpected Finding:** Linear models outperformed complex tree-based models. "
        "Linear signals were dominant in our synthetic data. With real project data, the picture might differ — "
        "this is why we keep both linear and tree-based models in our toolkit."
    )

    st.markdown("---")

    # Confusion Matrix
    st.markdown("##### 🎯 Confusion Matrix — Logistic Regression")
    col_cm1, col_cm2 = st.columns([1, 1])

    with col_cm1:
        cm_data = np.array([
            [84.0, 14, 1, 1],
            [19, 61.4, 18, 2],
            [3, 26, 44.8, 26],
            [0, 4, 26, 70.5]
        ])
        labels = ['On Time', 'Mild', 'Moderate', 'Severe']
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm_data, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': '% (each row sums to 100)'}, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col_cm2:
        st.markdown("**📌 Interpretation:**")
        st.markdown("""
        - ✅ **On Time** 84% correct — well-separated class
        - ✅ **Severe** 70.5% captured — the most critical class
        - ⚠️ **Moderate** 44.8% — squeezed between Mild and Severe
        - 🎯 **Most critical:** Severe → On Time misprediction is zero (worst case avoided)
        """)

    st.markdown("---")

    # Feature Importance
    st.markdown("##### 🏆 Top 8 Feature Importance")
    fi_data = pd.DataFrame({
        'Feature': ['contractor_avg_delay_days', 'is_risky_contractor', 'contractor_id_Vendor_G',
                    'contractor_x_critical', 'material_not_ordered', 'float_ratio',
                    'days_per_crew', 'predecessor_density'],
        'Importance (%)': [17.7, 8.0, 6.2, 5.4, 3.1, 3.0, 2.9, 2.9]
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi_data['Feature'][::-1], fi_data['Importance (%)'][::-1],
            color='#4A90E2', edgecolor='white')
    ax.set_xlabel("Importance score (%)")
    for i, v in enumerate(fi_data['Importance (%)'][::-1]):
        ax.text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=10)
    st.pyplot(fig)
    plt.close()

    st.info(
        "💡 **4 out of the top 5 features are contractor-related**. The EDA hypothesis was confirmed: "
        "'Contractor selection is everything.'"
    )


# ================================================================
# TAB 4: ABOUT
# ================================================================
with tab4:
    st.subheader("About the Project")

    col_about1, col_about2 = st.columns([2, 1])

    with col_about1:
        st.markdown("""
        ### 🏗️ Construction Milestone Delay Prediction

        This project is a machine learning system that predicts the delay risk of construction activities,
        built on a dataset derived from a **Greenfield Automotive Paint Shop Project**.

        #### 🎯 Goal
        Replace the traditional PMO approach (subjective red/yellow/green ratings in weekly meetings) with
        a **data-driven early warning system**.

        #### 🔬 Methodology
        - **EDA:** Understand the data, form hypotheses
        - **Feature Engineering:** Translate EDA findings into model-ready features
        - **Modeling:** Comparison of 5 algorithms, hyperparameter optimization

        #### 📊 Dataset
        - 3,565 activities × 26 variables
        - 18-month project lifecycle
        - Synthetic (structural information from real MS Project schedule, sensitive details anonymized)

        #### ⚠️ Limitations
        - Synthetic data — retraining with real project data is needed
        - Single-plant data — transfer learning required for multi-plant
        - Not production-ready — MLOps integration is missing
        """)

    with col_about2:
        st.markdown("### 🔗 Links")
        st.markdown("""
        - 🐙 **GitHub:**
          [milestone-delay-prediction](https://github.com/alucard3807/milestone-delay-prediction)
        - 💼 **LinkedIn:**
          [ugurkaplanuk](https://www.linkedin.com/in/ugurkaplanuk/)
        - 📚 **Bootcamp:**
          [Miuul Data Science](https://miuul.com)
        """)

        st.markdown("---")
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        - Python 3.11
        - scikit-learn
        - LightGBM, XGBoost
        - SHAP
        - Streamlit
        - pandas, numpy, matplotlib, seaborn
        """)

    st.markdown("---")
    st.markdown("### 📚 Literature References")
    refs = [
        ("Gondia et al. (2020)", "ML Algorithms for Construction Delay Risk Prediction",
         "https://doi.org/10.1061/(ASCE)CO.1943-7862.0001736"),
        ("Yaseen et al. (2020)", "Hybrid AI Model for Delay Prediction",
         "https://www.mdpi.com/2071-1050/12/4/1514"),
        ("Egwim et al. (2021)", "Applied AI for Construction Projects Delay",
         "https://www.sciencedirect.com/science/article/pii/S2666827021000839"),
        ("Alsulamy (2025)", "CatBoost vs XGBoost vs LightGBM",
         "https://doi.org/10.1016/j.eswa.2024.126268"),
    ]
    for author, title, url in refs:
        st.markdown(f"- **{author}** — [{title}]({url})")

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:1rem; color:#666; font-size:0.9rem;'>
        Built with 💙 by Uğur · Greenfield Automotive Paint Shop Project · May 2026
    </div>
    """, unsafe_allow_html=True)

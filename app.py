
import streamlit as st
import pandas as pd

from prediction import predict_diabetes, feature_order

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #0e1117;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #20212d;
}

.hero {
    background: linear-gradient(120deg, #25204c, #172b50);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #39356e;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 10px;
}

.hero p {
    color: #c5c9e5;
    font-size: 15px;
}

.section-card {
    background: #191e2b;
    border: 1px solid #30394b;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}

.result-card {
    background: #191e2b;
    border: 1px solid #39445c;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: #5746e8;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    min-height: 45px;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #7164f5;
    color: white;
}

h2, h3 {
    color: #f8fafc;
}

.small-muted {
    color: #a5afc2;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("# 🩺 Diabetes AI")

    st.caption("Health Assessment System")

    st.divider()

    page = st.radio(
        "Navigation",
        ["Diabetes Risk Assessment", "About"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### About")

    st.write(
        "This application uses Machine Learning "
        "to estimate diabetes probability based "
        "on health indicators from BRFSS 2015."
    )

    st.caption(
        "Powered by Random Forest Classifier"
    )


# ==========================================
# FEATURE LABELS
# ==========================================

feature_labels = {
    "BMI": "Body Mass Index (BMI)",
    "Age": "Age Group",
    "GenHlth": "General Health",
    "Income": "Annual Income",
    "PhysHlth": "Physical Health (Days)",
    "HighBP": "High Blood Pressure",
    "Education": "Education Level",
    "MentHlth": "Mental Health (Days)",
    "HighChol": "High Cholesterol",
    "Fruits": "Daily Fruit Consumption"
}


# ==========================================
# INPUT FORM
# ==========================================

if page == "Diabetes Risk Assessment":

    st.markdown("""
    <div class="hero">
        <h1>🩺 Diabetes Risk Assessment</h1>
        <p>
        Understand your health indicators and
        evaluate your diabetes probability using
        Artificial Intelligence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📋 Input Your Health Information")

    st.caption(
        "Please complete all measurements below."
    )

    user_input = {}

    with st.form("diabetes_form"):

        left, right = st.columns(2)

        # ------------------------------
        # LEFT COLUMN
        # ------------------------------

        with left:

            st.markdown("### 👤 Personal Information")

            user_input["BMI"] = st.number_input(
                "Body Mass Index (BMI)",
                min_value=12.0,
                max_value=80.0,
                value=25.0,
                step=0.1,
                help="Enter your BMI value."
            )

            age_options = {
                1: "18–24 years",
                2: "25–29 years",
                3: "30–34 years",
                4: "35–39 years",
                5: "40–44 years",
                6: "45–49 years",
                7: "50–54 years",
                8: "55–59 years",
                9: "60–64 years",
                10: "65–69 years",
                11: "70–74 years",
                12: "75–79 years",
                13: "80 years or older"
            }

            age_value = st.selectbox(
                "Age Group",
                options=list(age_options.keys()),
                format_func=lambda x: age_options[x],
                index=4
            )

            user_input["Age"] = age_value

            health_options = {
                1: "Excellent",
                2: "Very Good",
                3: "Good",
                4: "Fair",
                5: "Poor"
            }

            health_value = st.selectbox(
                "General Health",
                options=list(health_options.keys()),
                format_func=lambda x: health_options[x],
                index=2
            )

            user_input["GenHlth"] = health_value

            income_options = {
                1: "Less than $10,000",
                2: "$10,000 – $14,999",
                3: "$15,000 – $19,999",
                4: "$20,000 – $24,999",
                5: "$25,000 – $34,999",
                6: "$35,000 – $49,999",
                7: "$50,000 – $74,999",
                8: "$75,000 or more"
            }

            income_value = st.selectbox(
                "Annual Income",
                options=list(income_options.keys()),
                format_func=lambda x: income_options[x],
                index=4
            )

            user_input["Income"] = income_value

            education_options = {
                1: "Never attended school",
                2: "Elementary school",
                3: "Some high school",
                4: "High school graduate",
                5: "Some college / technical school",
                6: "College graduate"
            }

            education_value = st.selectbox(
                "Education Level",
                options=list(education_options.keys()),
                format_func=lambda x: education_options[x],
                index=5
            )

            user_input["Education"] = education_value

        # ------------------------------
        # RIGHT COLUMN
        # ------------------------------

        with right:

            st.markdown("### ❤️ Health Indicators")

            user_input["PhysHlth"] = st.slider(
                "Physical Health (Days)",
                min_value=0,
                max_value=30,
                value=0,
                help="Number of days physical health was not good during the past 30 days."
            )

            user_input["MentHlth"] = st.slider(
                "Mental Health (Days)",
                min_value=0,
                max_value=30,
                value=0,
                help="Number of days mental health was not good during the past 30 days."
            )

            bp_options = {
                0: "No",
                1: "Yes"
            }

            bp_value = st.selectbox(
                "High Blood Pressure",
                options=[0, 1],
                format_func=lambda x: bp_options[x]
            )

            user_input["HighBP"] = bp_value

            cholesterol_value = st.selectbox(
                "High Cholesterol",
                options=[0, 1],
                format_func=lambda x: bp_options[x]
            )

            user_input["HighChol"] = cholesterol_value

            fruit_value = st.selectbox(
                "Daily Fruit Consumption",
                options=[0, 1],
                format_func=lambda x: (
                    "Yes" if x == 1 else "No"
                )
            )

            user_input["Fruits"] = fruit_value

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Analyze Health Metrics",
            use_container_width=True
        )


    # ==========================================
    # PREDICTION RESULT
    # ==========================================

    if submitted:

        try:
            # Follow the exact feature order
            ordered_input = {
                feature: user_input[feature]
                for feature in feature_order
            }

            label, probability = predict_diabetes(
                ordered_input
            )

            st.divider()

            st.markdown("## 📊 Analysis Results")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True
                )

                st.markdown("### Prediction Result")

                if label == "Diabetes":
                    st.error("Diabetes Predicted")
                else:
                    st.success("No Diabetes Predicted")

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True
                )

                st.markdown("### Diabetes Probability")

                st.markdown(
                    f"<h1>{probability * 100:.2f}%</h1>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            st.progress(
                min(max(float(probability), 0.0), 1.0)
            )

            st.caption(
                "Probability represents the model's estimated "
                "probability for the positive class, not a "
                "clinical diagnosis or medically validated risk score."
            )

            # ------------------------------
            # KEY FINDINGS
            # ------------------------------

            st.markdown("## 🎯 Key Findings")

            if label == "Diabetes":

                st.warning(
                    "The model predicts the positive diabetes class. "
                    "Consider discussing your health indicators "
                    "with a healthcare professional."
                )

            else:

                st.success(
                    "The model predicts the negative diabetes class. "
                    "Continue maintaining healthy lifestyle habits "
                    "and routine health checkups."
                )

            # ------------------------------
            # INPUT SUMMARY
            # ------------------------------

            with st.expander("View Your Health Information"):

                summary = pd.DataFrame({
                    "Health Indicator": [
                        feature_labels[f]
                        for f in feature_order
                    ],
                    "Input Value": [
                        user_input[f]
                        for f in feature_order
                    ]
                })

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True
                )

            st.info(
                "This application is intended for educational "
                "and research purposes. It does not replace "
                "professional medical advice or diagnostic testing."
            )

        except Exception as e:

            st.error(f"Prediction Error: {e}")


# ==========================================
# ABOUT PAGE
# ==========================================

else:

    st.markdown("""
    <div class="hero">
        <h1>About Diabetes Detection</h1>
        <p>
        An intelligent health assessment application
        powered by Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🤖 About the Model")

    st.write(
        "This application uses a Random Forest Classifier "
        "trained on the BRFSS 2015 Diabetes Health Indicators Dataset."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Algorithm", "Random Forest")
    col2.metric("Features", "10")
    col3.metric("Dataset", "BRFSS 2015")

    st.markdown("## 📌 Model Features")

    for feature in feature_order:
        st.write(
            f"• {feature_labels.get(feature, feature)}"
        )

    st.markdown("## 🎯 Purpose")

    st.write(
        "The system demonstrates how machine learning "
        "can analyze health indicators and estimate "
        "the probability of diabetes based on historical data."
    )

    st.caption(
        "Developed for educational and research purposes."
    )

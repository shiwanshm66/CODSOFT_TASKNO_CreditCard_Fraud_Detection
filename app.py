
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write(
    "An AI-powered system that identifies potentially "
    "fraudulent credit card transactions."
)

MODEL_PATH = "models/best_fraud_detection_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please run train.py first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.success("Fraud detection model loaded successfully!")

st.subheader("Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing transaction records",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)

        if "is_fraud" in data.columns:
            actual_labels = data["is_fraud"]
            features = data.drop(columns=["is_fraud"])
        else:
            actual_labels = None
            features = data.copy()

        unnecessary_columns = [
            "Unnamed: 0",
            "trans_num",
            "cc_num",
            "first",
            "last",
            "street",
            "dob"
        ]

        features = features.drop(
            columns=unnecessary_columns,
            errors="ignore"
        )

        if not hasattr(model, "feature_names_in_"):
            st.error("Model feature information is unavailable.")
            st.stop()

        expected_columns = list(model.feature_names_in_)

        missing_columns = [
            col for col in expected_columns
            if col not in features.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        features = features[expected_columns]

        if st.button("Detect Fraud", type="primary"):

            with st.spinner("Analyzing transactions..."):

                predictions = model.predict(features)
                probabilities = model.predict_proba(features)[:, 1]

                results = data.copy()

                results["Fraud_Prediction"] = [
                    "Fraudulent" if value == 1 else "Legitimate"
                    for value in predictions
                ]

                results["Fraud_Probability"] = probabilities

                total = len(predictions)
                fraud_count = int((predictions == 1).sum())
                legitimate_count = total - fraud_count

            st.success("Transaction analysis completed!")

            st.subheader("Analysis Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Transactions", f"{total:,}")
            col2.metric("Legitimate Transactions", f"{legitimate_count:,}")
            col3.metric("Potential Fraud", f"{fraud_count:,}")

            st.subheader("Prediction Results")

            st.dataframe(
                results,
                use_container_width=True
            )

            st.subheader("Fraud Detection Distribution")

            chart_data = pd.DataFrame({
                "Transaction Type": ["Legitimate", "Fraudulent"],
                "Count": [legitimate_count, fraud_count]
            })

            st.bar_chart(
                chart_data.set_index("Transaction Type")
            )

            if actual_labels is not None:
                st.subheader("Model Evaluation")

                if len(actual_labels) == len(predictions):

                    from sklearn.metrics import (
                        accuracy_score,
                        precision_score,
                        recall_score,
                        f1_score
                    )

                    accuracy = accuracy_score(
                        actual_labels, predictions
                    )

                    precision = precision_score(
                        actual_labels, predictions,
                        zero_division=0
                    )

                    recall = recall_score(
                        actual_labels, predictions,
                        zero_division=0
                    )

                    f1 = f1_score(
                        actual_labels, predictions,
                        zero_division=0
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
                    c2.metric("Precision", f"{precision * 100:.2f}%")
                    c3.metric("Recall", f"{recall * 100:.2f}%")
                    c4.metric("F1 Score", f"{f1 * 100:.2f}%")

            csv_data = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_data,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.divider()

st.caption(
    "Credit Card Fraud Detection System | CodSoft Machine Learning Internship"
)

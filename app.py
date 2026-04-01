import streamlit as st
import pickle
import pandas as pd

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- HYBRID PREDICTION FUNCTION ----------------
def predict_meaning(text):
    s = text.lower()

    # RULE-BASED OVERRIDE
    if any(word in s for word in ["river", "water", "shore", "lake", "stream"]):
        return "river"

    if any(word in s for word in ["money", "account", "loan", "deposit", "cash", "balance"]):
        return "financial"

    # ML MODEL
    X = vectorizer.transform([text])
    return model.predict(X)[0]


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="WSD ML System",
    layout="wide"
)

# ---------------- HEADER ----------------
st.title("Word Sense Disambiguation System")
st.caption("Hybrid Machine Learning and Rule-Based Semantic Analysis")

st.markdown("""
---
### Student Details

**Name:** Mohan Narayanapuram  
**Register Number:** RA2311056010126  
**Course:** 21CSE356T - Natural Language Processing  
**Faculty:** P. Kanmani  
**Institution:** SRM Institute of Science and Technology  

---
""")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Select Module",
    ["Home", "Single Prediction", "Dataset Testing", "Model Info"]
)

# ---------------- HOME ----------------
if option == "Home":
    st.subheader("Project Overview")

    st.markdown("""
    This application implements Word Sense Disambiguation (WSD) using a hybrid approach.

    ### Key Features
    - Machine Learning based classification (Naive Bayes)
    - TF-IDF feature extraction
    - Rule-based refinement for improved accuracy
    - Real-time user input prediction
    - Dataset testing using Excel upload

    ### Methodology
    1. Input sentence is processed  
    2. TF-IDF converts text into numerical features  
    3. Naive Bayes predicts meaning  
    4. Rule-based logic refines prediction  
    5. Final result is displayed  
    """)

# ---------------- SINGLE PREDICTION ----------------
elif option == "Single Prediction":
    st.subheader("Single Sentence Prediction")

    user_input = st.text_area("Enter a sentence", height=120)

    if st.button("Predict Meaning"):
        if user_input.strip() != "":
            prediction = predict_meaning(user_input)

            # ML confidence (for display only)
            X = vectorizer.transform([user_input])
            probs = model.predict_proba(X)[0]

            st.markdown("#### Prediction Result")
            st.success(f"Predicted Meaning: {prediction}")

            st.markdown("#### Confidence Scores (ML Model)")
            prob_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probs
            })
            st.dataframe(prob_df, use_container_width=True)

        else:
            st.warning("Please enter a valid sentence.")

# ---------------- DATASET TESTING ----------------
elif option == "Dataset Testing":
    st.subheader("Dataset Testing (Excel Upload)")

    uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            st.markdown("#### Uploaded Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)

            if "sentence/context" in df.columns:

                if st.button("Run Prediction on Dataset"):

                    results = []

                    for sentence in df["sentence/context"]:
                        pred = predict_meaning(str(sentence))
                        results.append(pred)

                    df["Predicted Meaning"] = results

                    st.markdown("#### Prediction Results")
                    st.dataframe(df, use_container_width=True)

                    # Download option
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="wsd_results.csv",
                        mime="text/csv"
                    )

            else:
                st.error("Required column 'sentence/context' not found.")

        except Exception as e:
            st.error(f"Error reading file: {e}")

# ---------------- MODEL INFO ----------------
elif option == "Model Info":
    st.subheader("Model Information")
    st.markdown("### Model Performance")
    st.metric(label="Accuracy", value="85.71%")

    st.markdown("""
    ### Model Details

    **Algorithm:** Multinomial Naive Bayes  
    **Feature Extraction:** TF-IDF Vectorizer  
    **Approach:** Hybrid (Machine Learning + Rule-Based)

    ### Why Hybrid Approach?
    - Machine Learning captures general patterns  
    - Rule-based logic improves accuracy for ambiguous contexts  

    ### Pipeline
    1. Sentence input  
    2. TF-IDF vectorization  
    3. Naive Bayes prediction  
    4. Rule-based refinement  
    5. Final output  

    ### Notes
    - Model trained on dataset-derived labels  
    - Excel dataset used for evaluation  
    """)

    st.markdown("#### Available Classes")
    st.write(model.classes_)
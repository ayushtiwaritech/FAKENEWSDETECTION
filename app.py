import streamlit as st
import joblib
import pandas as pd

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Fake News Detector", layout="centered")

# -------------------- LOAD DATASET --------------------
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# -------------------- AI THEME --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title */
h1 {
    text-align: center;
    color: #00ffe7;
}

/* Input box */
textarea {
    background-color: #1e1e1e !important;
    color: white !important;
    border: 2px solid #00ffe7 !important;
    border-radius: 10px !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #00ffe7, #00c3ff);
    color: black;
    border-radius: 10px;
    font-weight: bold;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<h1>🤖 AI Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyze news using Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Model Info")
st.sidebar.markdown("""
- Algorithm: Logistic Regression  
- Vectorizer: TF-IDF  
- Accuracy: 98.6%  
""")

# -------------------- LOAD MODEL --------------------
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# -------------------- SESSION STATE --------------------
if "news_input" not in st.session_state:
    st.session_state.news_input = ""

# -------------------- EXAMPLE BUTTONS --------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🟢 Try Real Example"):
        random_real = true_df.sample(1)["text"].values[0]
        st.session_state.news_input = random_real[:500]

with col2:
    if st.button("🔴 Try Fake Example"):
        random_fake = fake_df.sample(1)["text"].values[0]
        st.session_state.news_input = random_fake[:500]

# -------------------- INPUT --------------------
news_input = st.text_area("📝 Enter News Article:", value=st.session_state.news_input, height=200)

# -------------------- WORD COUNT --------------------
if news_input:
    st.write(f"Word Count: {len(news_input.split())}")

# -------------------- PREDICTION --------------------
if st.button("🚀 Analyze News"):
    if news_input.strip():

        if len(news_input.split()) < 5:
            st.warning("⚠️ Please enter more detailed text.")
        else:
            transform_input = vectorizer.transform([news_input])

            prediction = model.predict(transform_input)
            proba = model.predict_proba(transform_input)

            # RESULT
            if prediction[0] == 1:
                confidence = proba[0][1]
                st.markdown(f"""
                <div style='padding:15px; border-radius:10px; background:#0f5132; border:2px solid #00ff88;'>
                ✅ REAL NEWS<br><br>
                Confidence: {confidence*100:.2f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence = proba[0][0]
                st.markdown(f"""
                <div style='padding:15px; border-radius:10px; background:#58151c; border:2px solid #ff4d4d;'>
                ❌ FAKE NEWS<br><br>
                Confidence: {confidence*100:.2f}%
                </div>
                """, unsafe_allow_html=True)

            # PROGRESS BAR
            st.progress(int(confidence * 100))

    else:
        st.warning("Please enter some text.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<center style='color:#00ffe7;'>Made with ❤️ by Ayush, Manjeet & Sagar</center>",
    unsafe_allow_html=True
)

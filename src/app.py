# src/app.py
import streamlit as st
import joblib
import numpy as np

# Load model
bundle = joblib.load(r"C:\Users\Admin\Desktop\ML Project (Stress Detector)\stress-detection\models\stress_model.joblib")
model = bundle["model"]
le = bundle["label_encoder"]

# Page config
st.set_page_config(page_title="Stress Detector", page_icon="😰", layout="wide")

# Custom CSS for phenomenal UI
st.markdown("""
<style>
/* Animated gradient background */
body {
    background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #a1c4fd, #c2e9fb);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    color: #333333;
}

/* Gradient animation */
@keyframes gradientBG {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

/* Title styling */
h1, h2, h3 {
    color: #ff4b4b;
    font-family: 'Helvetica', sans-serif;
    text-align: center;
}

/* Sliders styling */
.stSlider > div[data-baseweb="slider"] > div {
    background: rgba(255,255,255,0.8);
    border-radius: 12px;
}

/* Buttons styling */
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    font-size: 16px;
    height: 50px;
    border-radius: 15px;
    font-weight: bold;
}

/* Prediction card */
.pred-card {
    background-color: rgba(255,255,255,0.9);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 3px 6px 25px rgba(0,0,0,0.2);
    text-align: center;
    margin-bottom: 25px;
}

/* Probability container */
.prob-container {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
    width: 100%;
    background: rgba(255,255,255,0.5);
    border-radius: 15px;
    padding: 5px;
}

/* Inner colored bar */
.prob-bar-inner {
    height: 35px;
    border-radius: 18px;
    color: white;
    font-weight: bold;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# App title
st.title("Stress Level Detector from Daily Logs")

# Layout sliders
c1, c2 = st.columns(2)
with c1:
    sleep = st.slider("Sleep hours", 3.0, 10.0, 7.0, 0.5)
    work = st.slider("Work/Study hours", 0.0, 16.0, 8.0, 0.5)
    exercise = st.slider("Exercise hours", 0.0, 4.0, 1.0, 0.25)
with c2:
    screen = st.slider("Screen time (hrs)", 0.0, 16.0, 5.0, 0.5)
    water = st.slider("Water intake (L)", 0.0, 6.0, 2.5, 0.25)
    mood = st.slider("Mood rating (1-10)", 1, 10, 5)

# Predict button
if st.button("Detect"):
    X_row = np.array([[sleep, work, exercise, screen, water, mood]])
    proba = model.predict_proba(X_row)[0]
    idx = int(proba.argmax())
    label = le.inverse_transform([idx])[0]

    prediction_text = f"{label} Stress"

    # Prediction card
    st.markdown(f"""
    <div class="pred-card">
        <h2>Predicted stress level: <b>{prediction_text}</b></h2>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars with alignment
    st.subheader("Prediction Probabilities")
    color_map = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
    
    for lab, p in zip(le.classes_, proba):
        percent = int(p * 100)
        bar_width = max(percent, 5)  # minimum width for visibility
        st.markdown(f"""
        <div class="prob-container">
            <div class="prob-bar-inner" style="width:{bar_width}%; background-color:{color_map.get(lab,'#3498db')}">
                {percent}%
            </div>
            <span style="margin-left:10px; font-weight:bold;">{lab} Stress</span>
        </div>
        """, unsafe_allow_html=True)

    # Suggestions box
    if label == "High":
        st.warning("High stress detected. Suggestions: sleep more, short exercise breaks, reduce screen time, hydrate.")
    elif label == "Medium":
        st.info("Medium stress. Try short mindfulness breaks and optimize work schedule.")
    else:
        st.success("Low stress — keep up the healthy habits!")

st.markdown("---")
st.caption("Adjust sliders and click 'Detect' to see your stress level.")
st.caption("By Atharva Kanase")
st.caption("ML Project")


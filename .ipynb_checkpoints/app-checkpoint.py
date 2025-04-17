import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Custom CSS for vibrant look
st.markdown("""
    <style>
        body {
            background-color: #f3f4f6;
        }
        .title {
            font-size: 3rem;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            background: linear-gradient(to right, #4ade80, #22d3ee);
            padding: 1rem;
            border-radius: 20px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
            margin-bottom: 2rem;
        }
        .textbox {
            padding: 10px;
            border: 2px solid #cbd5e1;
            border-radius: 12px;
        }
        .predict-btn {
            background-color: #6366f1;
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            border-radius: 8px;
            cursor: pointer;
        }
        .predict-btn:hover {
            background-color: #4f46e5;
        }
        .result {
            font-size: 1.2rem;
            margin-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📩 Spam Message Classifier</div>', unsafe_allow_html=True)

msg = st.text_area("✍️ Enter your message below:", placeholder="Type or paste a message...", height=150)

if st.button("🔍 Predict", help="Click to classify the message"):
    if msg.strip() == "":
        st.warning("⚠️ Please enter a message first!")
    else:
        data = vectorizer.transform([msg])
        prediction = model.predict(data)[0]

        if prediction == 1:
            st.error("⚠️ This is a **Spam** message.", icon="🚫")
        else:
            st.success("✅ This is a **Ham (Not Spam)** message.", icon="✅")

st.markdown("---")
st.markdown("Made with ❤️ by Bhavana", unsafe_allow_html=True)

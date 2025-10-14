import streamlit as st
import joblib
import requests

# ========================
# Load ML model & vectorizer
# ========================
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# ========================
# App UI
# ========================
st.set_page_config(page_title="AI Fake News Detector for Students", page_icon="📰", layout="wide")
st.title("📰 AI-Powered Fake News Detector for Students")

# About Tool Section
st.markdown("""
<div style='
    background-color:#dbeafe; 
    color:#1e3a8a; 
    padding:15px; 
    border-radius:10px; 
    border:1px solid #93c5fd;
    animation: slideIn 1s ease-out;
'>
    <h4 style='margin:0;'>🛠️ About This Tool</h4>
    <p style='margin:5px 0 0 0;'>
        This AI-powered Fake News Detector helps students critically evaluate online news articles.  
        Enter a news article on the left, and the tool provides a <strong>clear verdict</strong> (REAL or FAKE) with <strong>dynamic AI reasoning</strong>.
    </p>
</div>

<style>
@keyframes slideIn {
    0% { transform: translateY(-20px); opacity: 0; }
    100% { transform: translateY(0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# ========================
# ML Prediction
# ========================
def predict_fake_news(article):
    transformed = vectorizer.transform([article])
    return model.predict(transformed)[0]

# ========================
# Hugging Face Explanation (Dynamic)
# ========================
@st.cache_data(show_spinner=False)
def explain_with_huggingface(article, result, max_tokens=500, temperature=0.7):
    """
    Returns (reasoning_text, is_dynamic)
    """
    try:
        prompt = f"""
        Analyze the following news article and determine if it is {"REAL" if result == 1 else "FAKE"}.
        Provide reasoning in points under each section:

        1. Summary 📄
        2. Tone ✍️
        3. Credibility 🔍
        4. Evidence 📊
        5. Conclusion 🎯

        Article:
        {article}

        Format your response in points under each section.
        """
        API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
        headers = {"Authorization": f"Bearer {st.secrets.get('HF_API_KEY', '')}"}

        if not headers["Authorization"]:
            raise Exception("HF_API_KEY not found in Streamlit secrets!")

        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}},
            timeout=60
        )

        data = response.json()
        if isinstance(data, dict) and "error" in data:
            raise Exception(data["error"])

        return data[0]["generated_text"], True  # dynamic reasoning

    except Exception as e:
        st.warning(f"Hugging Face API failed: {e}")
        # Fallback reasoning
        if result == 1:
            reasoning = (
                "1. Summary 📄: ✅ Article is factually correct.\n"
                "2. Tone ✍️: Neutral.\n"
                "3. Credibility 🔍: Reliable sources.\n"
                "4. Evidence 📊: Supported by data.\n"
                "5. Conclusion 🎯: Likely real; trustworthy."
            )
        else:
            reasoning = (
                "1. Summary 📄: 🚨 Misleading claims.\n"
                "2. Tone ✍️: Sensational.\n"
                "3. Credibility 🔍: Unreliable sources.\n"
                "4. Evidence 📊: Unsupported.\n"
                "5. Conclusion 🎯: Likely fake; verify before trusting."
            )
        return reasoning, False  # fallback

# ========================
# Layout: left input, right verdict + AI reasoning
# ========================
col_input, col_result = st.columns([2.2, 1.3])

# -------- Left: Text Input --------
with col_input:
    news_text = st.text_area("📝 Enter News Article Here:", "", height=300)
    if st.button("🔍 Check News"):
        if not news_text.strip():
            st.warning("⚠️ Please enter a news article to analyze.")

# -------- Right: Verdict + AI Reasoning --------
with col_result:
    st.subheader("🏷️ Verdict & AI Reasoning")

    if not news_text.strip():
        st.markdown(
            "<div style='background-color:#e0e0e0; padding:20px; text-align:center; font-size:20px; border-radius:10px; border:1px solid #ccc;'>"
            "ℹ️ Awaiting Input</div>",
            unsafe_allow_html=True
        )
    else:
        result = predict_fake_news(news_text)
        verdict_bg = "#d4edda" if result == 1 else "#f8d7da"
        ai_bg = "#e6f9e6" if result == 1 else "#ffe6e6"
        emoji = "✅" if result == 1 else "🚨"
        verdict_text = "REAL" if result == 1 else "FAKE"
        verdict_color = "green" if result == 1 else "red"

        # Verdict Box
        st.markdown(f"""
            <div style='background-color:{verdict_bg}; padding:12px; border-radius:10px; 
                        text-align:center; border:2px solid #ccc; box-shadow: 3px 3px 8px rgba(0,0,0,0.1);'>
                <span style='font-size:32px; animation:bounce 1s infinite;'>{emoji}</span><br>
                <strong style='font-size:24px; color:{verdict_color};'>{verdict_text}</strong>
            </div>
            <style>
                @keyframes bounce {{
                    0%, 20%, 50%, 80%, 100% {{transform:translateY(0);}}
                    40% {{transform:translateY(-8px);}}
                    60% {{transform:translateY(-4px);}}
                }}
            </style>
        """, unsafe_allow_html=True)

        # AI Reasoning
        st.subheader("🤖 AI Reasoning")
        explanation, is_dynamic = explain_with_huggingface(news_text, result)
        lines = explanation.strip().split('\n')
        formatted_html = "".join(f"<p style='margin:0; font-weight:bold; padding:2px 0;'>{line}</p>" for line in lines)

        reasoning_label = "Dynamic AI reasoning (Hugging Face)" if is_dynamic else "Offline fallback reasoning"

        st.markdown(
            f"""
            <div style='background-color:{ai_bg}; padding:10px; border-radius:10px; border:2px solid #ccc; 
                        box-shadow: 2px 2px 6px rgba(0,0,0,0.08);'>
                <p style='margin:0; font-style:italic; font-size:13px; color:#555;'>{reasoning_label}</p>
                {formatted_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Critical Reading Tip
        st.markdown(
            """
            <div style="background-color:#fef3c7; color:#854d0e; padding:6px; border-radius:10px; border:1px solid #fde68a; margin-top:6px;">
            <strong>💡 Tip:</strong> AI explanations are helpful, but always check sources and think critically.
            </div>
            """,
            unsafe_allow_html=True
        )

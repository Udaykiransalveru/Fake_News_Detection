import streamlit as st
import requests
import sqlite3

# ==========================
# DATABASE
# ==========================

conn = sqlite3.connect(
    "users.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    claim TEXT,
    verdict TEXT
)
""")
conn.commit()

# ==========================
# SESSION STATE
# ==========================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "page" not in st.session_state:
    st.session_state.page = "Login"
# ==========================
# FUNCTIONS
# ==========================

def register_user(username, password):

    cursor.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    )

    if cursor.fetchone():
        return False

    cursor.execute(
        "INSERT INTO users VALUES (?,?)",
        (username, password)
    )

    conn.commit()

    return True


def login_user(username, password):

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    return cursor.fetchone()


st.set_page_config(
page_title="AI News Verification System",
page_icon="📰",
layout="wide"
)
st.markdown("""
<style>

.block-container{
    max-width:1200px;
    padding-top:1rem;
}

.stButton>button{
    width:100%;
    height:50px;
    border-radius:10px;
    font-size:16px;
    font-weight:bold;
}

h1{
    text-align:center;
}

</style>
""", unsafe_allow_html=True)
if not st.session_state.logged_in:

    st.title("🔐 User Authentication")

    option = st.sidebar.selectbox(
    "Select",
    ["Login", "Register"],
    index=0 if st.session_state.page == "Login" else 1
)
    

    if option == "Register":

        st.subheader("Create Account")

        username = st.text_input("Username")
        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Register"):

            if register_user(
                username.strip(),
                password.strip()
            ):

                st.session_state.page = "Login"

                st.info(
                    "✅ Registration Successful! Please select Login."
                )

            else:

                st.error(
                    "User already exists"
                )

    else:

        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            if login_user(
                username.strip(),
                password.strip()
            ):

                st.session_state.logged_in = True
                st.session_state.username = username

                st.rerun()

            else:

                st.error(
                    "Invalid Credentials"
                )

    st.stop()

# ----------------------------------

# HEADER

# ----------------------------------



st.markdown("""
<h1>📰 AI News Verification System</h1>
<p style='text-align:center;font-size:18px'>
Real-Time Fact Checking using AI + Google Fact Check API
</p>
""", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.markdown("## 👤 User")

    st.success(
        st.session_state.username
    )

    st.divider()

    st.metric(
        "Dataset",
        "44,898"
    )

    st.metric(
        "Accuracy",
        "98.79%"
    )

    st.divider()

    with st.expander(
        "⚙️ Technologies Used"
    ):

        st.markdown("""
        - Python
        - Streamlit
        - FastAPI
        - TF-IDF
        - Logistic Regression
        - NewsAPI
        - Google Fact Check API
        - Scikit-Learn
        """)

    st.divider()

    if st.button(
        "🚪 Logout"
    ):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

st.markdown(
    f"## Welcome, {st.session_state.username}"
)
# ----------------------------------

# INPUT

# ----------------------------------

news_text = st.text_area(
    "📝 Enter News Article or Claim",
    height=250,
    placeholder="Paste a news article or claim here..."
)
# ----------------------------------

# VERIFY BUTTON

# ----------------------------------

if st.button("🔍 Verify News"):


    if not news_text.strip():

        st.warning(
            "Please enter a news article."
        )

    else:

        try:

            response = requests.post(
                "http://127.0.0.1:8000/verify",
                json={
                    "news_text": news_text
                },
                timeout=30
            )

            result = response.json()

            prediction = result.get(
                "prediction",
                "Unknown"
            )

            confidence = result.get(
                "confidence",
                0
            )

            verdict = result.get(
                "verdict",
                "NO EVIDENCE FOUND"
            )

            cursor.execute(
                """
                INSERT INTO history(
                    username,
                    claim,
                    verdict
                )
                VALUES (?, ?, ?)
                """,
                (
                    st.session_state.username,
                    news_text,
                    verdict
                )
            )

            conn.commit()

            matched_sources = result.get(
                "matched_sources",
                0
            )

            # ----------------------------------
            # FINAL VERDICT
            # ----------------------------------

            st.subheader(
                "🎯 Verification Result"
            )
            if verdict == "FALSE CLAIM":

                st.error(
                    "❌ FALSE CLAIM"
                )

            elif verdict == "TRUE CLAIM":

                st.success(
                    "✅ TRUE CLAIM"
                )

            elif verdict == "FACT CHECK FOUND":

                st.info(
                    "🔍 FACT CHECK FOUND"
                )

            elif verdict == "HIGHLY VERIFIED":

                st.success(
                    "✅ HIGHLY VERIFIED"
                )

            elif verdict == "PARTIALLY VERIFIED":

                st.info(
                    "📰 PARTIALLY VERIFIED"
                )

            else:

                st.warning(
                    "⚠️ NO EVIDENCE FOUND"
                )

            st.write(
                f"Trusted Sources Matched: {matched_sources}"
            )

            # ----------------------------------
            # FACT CHECK RESULTS
            # ----------------------------------

            st.subheader(
                "🔍 Fact Check Results"
            )

            fact_checks = result.get(
                "fact_checks",
                []
            )

            if fact_checks:

                for fact in fact_checks:

                    claim = fact.get(
                        "text",
                        "Unknown Claim"
                    )

                    match_score = fact.get(
                        "match_score",
                        0
                    )

                    st.markdown(
                        f"### {claim}"
                    )

                    st.write(
                        f"Match Score: {match_score}%"
                    )

                    reviews = fact.get(
                        "claimReview",
                        []
                    )

                    if reviews:

                        review = reviews[0]

                        publisher = review.get(
                            "publisher",
                            {}
                        ).get(
                            "name",
                            "Unknown"
                        )

                        rating = review.get(
                            "textualRating",
                            "No Rating"
                        )

                        url = review.get(
                            "url",
                            ""
                        )

                        st.write(
                            f"Publisher: {publisher}"
                        )

                        st.write(
                            f"Rating: {rating}"
                        )

                        if url:

                            st.markdown(
                                f"[View Fact Check]({url})"
                            )

                    st.divider()

            else:

                st.info(
                    "No fact-check records found."
                )

            # ----------------------------------
            # TRUSTED NEWS SOURCES
            # ----------------------------------

            st.subheader(
                "📰 Trusted News Sources"
            )

            articles = result.get(
                "articles",
                []
            )

            if articles:

                for article in articles:

                    st.markdown(
                        f"### {article['title']}"
                    )

                    st.write(
                        f"Source: {article['source']}"
                    )

                    st.write(
                        f"Similarity Score: {article['similarity']}%"
                    )

                    if article["url"]:

                        st.markdown(
                            f"[Read Article]({article['url']})"
                        )

                    st.divider()

            else:

                st.info(
                    "No trusted news sources found."
                )

            # ----------------------------------
            # AI ANALYSIS
            # ----------------------------------

     

            if verdict in ["FALSE CLAIM", "TRUE CLAIM"]:

                st.success(
                    "✅ Verified using Google Fact Check API"
                )

            else:

                if verdict == "NO EVIDENCE FOUND":

                    st.info(
                        "No trusted evidence found online. AI prediction shown below is only a machine learning estimate."
                    )

                st.subheader("🤖 AI Analysis")

                col1, col2 = st.columns(2)

                with col1:

                    if prediction == "UNCERTAIN":

                        st.warning(
                            "AI Model is not confident enough."
                        )

                    else:

                        st.metric(
                            "Prediction",
                            prediction
                        )

                with col2:

                    st.metric(
                        "Confidence",
                        f"{confidence}%"
                    )

                st.progress(
                    int(confidence)
                )

        except Exception as e:

            st.error(
                f"API Connection Error: {e}"
                )

            st.info(
                "Make sure FastAPI is running:\n\nuvicorn api:app --reload"
                )
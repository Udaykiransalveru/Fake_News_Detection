import streamlit as st
import sqlite3

# =========================
# DATABASE
# =========================

conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

conn.commit()

# =========================
# SESSION STATE
# =========================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# =========================
# FUNCTIONS
# =========================

def register_user(username, password):

    cursor.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    )

    user = cursor.fetchone()

    if user:
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

    user = cursor.fetchone()

    return user


# =========================
# LOGIN / REGISTER PAGE
# =========================

if not st.session_state.logged_in:

    st.title("🔐 User Authentication")

    default_index = 0

    if st.session_state.get(
        "register_success",
        False
    ):
        default_index = 0

    option = st.sidebar.selectbox(
        "Select",
        ["Login", "Register"],
        index=default_index
    )

    if st.session_state.get(
        "register_success",
        False
    ):

        st.success(
            "✅ Account created successfully. Please login."
        )

        st.session_state.register_success = False

    # =====================
    # REGISTER
    # =====================

    if option == "Register":

        st.subheader("Create Account")

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Register"):

            if register_user(
                username.strip(),
                password.strip()
            ):

                st.success(
                    "✅ Registration Successful! Please Login."
                )

                st.session_state.register_success = True

                st.rerun()

            else:

                st.error(
                    "❌ User already exists"
                )

    # =====================
    # LOGIN
    # =====================

    else:

        st.subheader("Login")

        username = st.text_input(
            "Username"
        )

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

# =========================
# DASHBOARD
# =========================

else:

    st.title("📰 AI News Verification System")

    st.success(
        f"Welcome {st.session_state.username}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Dataset Size",
            "44,898"
        )

    with col2:
        st.metric(
            "Accuracy",
            "98.79%"
        )

    st.divider()

    news_text = st.text_area(
        "Enter News Article / Claim"
    )

    if st.button("Verify News"):

        if news_text:

            st.success(
                "News Verification Module Here"
            )

        else:

            st.warning(
                "Please enter news text"
            )

    st.divider()

    if st.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""

        st.rerun()
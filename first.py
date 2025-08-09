import streamlit as st

st.set_page_config(page_title="Welcome | Tea Optimizer", page_icon="🍵", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap');

    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 80vh;
        color: #3b2e2e;
        background: linear-gradient(135deg, #fceabb 0%, #f8b500 100%);
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(248, 181, 0, 0.5);
        padding: 40px;
        font-family: 'Great Vibes', cursive;
        text-align: center;
    }
    .welcome-title {
        font-size: 4rem;
        margin-bottom: 0.2rem;
    }
    .welcome-subtitle {
        font-size: 1.8rem;
        margin-bottom: 2rem;
        font-weight: 500;
        color: #5a4634;
    }
    .welcome-emoji {
        font-size: 5rem;
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="welcome-container">
        <div class="welcome-emoji">🍵</div>
        <h1 class="welcome-title">Welcome to Tea Optimizer</h1>
        <p class="welcome-subtitle">Discover your perfect cup with smart routing and mood-based coffee recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Use a key in session_state to remember page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

if st.button("☕ Go to Tea Optimizer"):
    st.session_state.page = "app"
    # Set the URL query parameter using st.experimental_set_query_params replacement
    st.experimental_set_query_params = None  # Just to silence the warning below if any
    st.experimental_rerun()

# Use st.query_params to read the current URL query params and update session_state accordingly
query_params = st.query_params
if query_params.get("page") == ["app"]:
    st.session_state.page = "app"

if st.session_state.page == "app":
    # Redirect to app.py or rerun your app code accordingly
    # For example, if you are running multiple files, you may use:
    st.markdown("Redirecting to app... (implement navigation here)")
    # Or if single script with multiple pages, use st.experimental_rerun or your routing logic here

# main.py  (Streamlit Cloud + Neon friendly)
import os
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import SYSTEM_PROMPT


# ---------------------------
# Helpers
# ---------------------------
def ensure_sslmode_require(db_url: str) -> str:
    """Append sslmode=require if not already present."""
    if not db_url:
        return db_url
    parsed = urlparse(db_url)
    qs = parse_qs(parsed.query)
    if "sslmode" not in qs:
        qs["sslmode"] = ["require"]
        new_q = urlencode(qs, doseq=True)
        db_url = urlunparse(parsed._replace(query=new_q))
    return db_url


def force_psycopg_v3(url: str) -> str:
    """Normalize driver to psycopg v3 for SQLAlchemy."""
    if url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


@st.cache_resource(show_spinner=False)
def get_db(db_url: str) -> SQLDatabase:
    # Small, resilient pool fits Neon serverless + Streamlit Cloud
    return SQLDatabase.from_uri(
        db_url,
        include_tables=["trips", "riders"],
        engine_args=dict(
            pool_pre_ping=True,
            pool_size=1,
            max_overflow=2,
            pool_recycle=1800,
        ),
    )


@st.cache_resource(show_spinner=False)
def get_llm(model_id: str, api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_id,
        temperature=0,
        max_output_tokens=512,
        google_api_key=api_key,
    )


def make_agent(llm: ChatGoogleGenerativeAI, db: SQLDatabase):
    # Not caching the agent to avoid pickling/version issues
    return create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        system_message=SYSTEM_PROMPT,
    )


def health_check(db: SQLDatabase) -> bool:
    try:
        with db._engine.connect() as conn:  # direct SQLAlchemy check
            conn.exec_driver_sql("SELECT 1;")
        return True
    except Exception:
        return False


# ---------------------------
# App Config / Secrets
# ---------------------------
st.set_page_config(page_title="Fetii Data Chat", layout="wide")
st.title("🗄️ Fetii Data Chat — A LangChain SQL Agent")
st.markdown("Chat with your Fetii rideshare data 🚐✨")

# Prefer Streamlit secrets on Cloud, fallback to env for local dev
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_ID = st.secrets.get("GEMINI_MODEL_ID") or os.getenv(
    "GEMINI_MODEL_ID", "gemini-2.5-flash-lite-preview-06-17"
)
DATABASE_URL = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")

# Guardrails
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY. Add it in Streamlit Secrets.")
    st.stop()
if not DATABASE_URL:
    st.error("Missing DATABASE_URL. Add it in Streamlit Secrets.")
    st.stop()

# Normalize DB URL for Neon + SSL
DATABASE_URL = ensure_sslmode_require(force_psycopg_v3(DATABASE_URL))

# Init resources
llm = get_llm(MODEL_ID, GOOGLE_API_KEY)
db = get_db(DATABASE_URL)
agent = make_agent(llm, db)

# Sidebar health + tips
with st.sidebar:
    st.subheader("Status")
    ok = health_check(db)
    st.write("Database:", "✅ Connected" if ok else "❌ Not reachable")
    st.caption("💬 Ready when you are — type a question and get instant insights.")

    # Optional toggles
    show_sql = st.toggle("Show SQL / traces", value=False)
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.success("Conversation cleared.")
        st.rerun()

# ---------------------------
# Conversation history
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"q": str, "a": str, "trace": any}

st.subheader("Conversation")
for i, item in enumerate(st.session_state.history, 1):
    st.markdown(f"**Q{i}:** {item['q']}")
    st.markdown(item["a"])
    if show_sql and item.get("trace"):
        with st.expander("View SQL / trace"):
            st.write(item["trace"])
    st.divider()

# ---------------------------
# Quick examples (optional)
# ---------------------------
with st.expander("✨ Try an example"):
    c1, c2, c3 = st.columns(3)
    examples = [
        "How many trips were completed last week?",
        "Top 3 dropoff_address for riders aged 21–25 on Saturdays",
        "When do large groups (6+ riders) usually travel downtown?",
    ]
    if c1.button("Example 1"):
        st.session_state.example_q = examples[0]
        st.experimental_rerun()
    if c2.button("Example 2"):
        st.session_state.example_q = examples[1]
        st.experimental_rerun()
    if c3.button("Example 3"):
        st.session_state.example_q = examples[2]
        st.experimental_rerun()

prefill = st.session_state.get("example_q", "")

# ---------------------------
# New question form
# ---------------------------
with st.form("ask_form", clear_on_submit=True):
    user_q = st.text_input(
        "Ask a data question (e.g., 'Top 10 dropoff_address for riders aged 18–24 on Saturdays?')",
        key=f"q_{len(st.session_state.history)}",
        value=prefill,
    )
    submitted = st.form_submit_button("🔎 Ask", type="primary")

# Clear the prefill after rendering the form once
if "example_q" in st.session_state:
    del st.session_state["example_q"]

if submitted and user_q:
    try:
        with st.spinner("Answering… please wait ⏳"):
            # IMPORTANT: non-streaming for broad model compatibility
            res = agent.invoke({"input": user_q}, config={"stream": False})
            answer = res.get("output", str(res))

            # Collect any useful trace info for optional display
            trace = res.get("intermediate_steps") or res

        st.session_state.history.append({"q": user_q, "a": answer, "trace": trace})
        st.rerun()
    except Exception as e:
        print(e)
        with st.spinner("Fallback: querying model directly…"):
            try:
                raw = llm.invoke(user_q)
                answer = getattr(raw, "content", str(raw))
                st.session_state.history.append({"q": user_q, "a": answer, "trace": None})
                st.rerun()
            except Exception as inner_e:
                st.error(f"Query failed: {e}; Fallback failed: {inner_e}")

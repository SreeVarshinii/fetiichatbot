# main.py — Fetii Data Chat (Refreshed UI)
import os
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from typing import List, Tuple

import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import text, inspect

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
        return_intermediate_steps=True,
        handle_parsing_errors=True,  # capture SQL/tool traces
    )


def health_check(db: SQLDatabase) -> bool:
    try:
        with db._engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1;")
        return True
    except Exception:
        return False


def read_schema(db: SQLDatabase) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Return [(table, [(col, type), ...]), ...] for quick display."""
    insp = inspect(db._engine)
    tables = []
    for t in ["trips", "riders"]:
        if insp.has_table(t):
            cols = [(c["name"], str(c["type"])) for c in insp.get_columns(t)]
            tables.append((t, cols))
    return tables


def safe_select(sql: str) -> bool:
    """Only allow SELECT (and safe WITH CTEs)."""
    s = sql.strip().lower()
    return s.startswith("select") or s.startswith("with")


# ---------------------------
# App Config / Secrets
# ---------------------------
st.set_page_config(page_title="Fetii RideShare Data Chat", layout="wide")
st.markdown(
    """
    <style>
    .stChatMessage { max-width: 980px; margin-left: auto; margin-right: auto; }
    .answer-card { border: 1px solid rgba(250,250,250,0.1); border-radius: 12px; padding: 12px; }
    .sql-chip { font-size: 0.8rem; opacity: 0.8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Prefer Streamlit secrets on Cloud, fallback to env for local dev
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_ID = st.secrets.get("GEMINI_MODEL_ID") or os.getenv(
    "GEMINI_MODEL_ID", "gemini-2.5-flash-lite-preview-06-17"
)
DATABASE_URL = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")

left_col, right_col = st.columns([0.72, 0.28], gap="large")

with left_col:
    st.title("🚐 Fetii Data Chat")
    st.caption("Ask questions about trips & riders. The agent writes SQL for you.")

# with right_col:
#     st.subheader("Status")
#     if not GOOGLE_API_KEY:
#         st.error("Missing GOOGLE_API_KEY. Add it in Streamlit Secrets.")
#         st.stop()
#     if not DATABASE_URL:
#         st.error("Missing DATABASE_URL. Add it in Streamlit Secrets.")
#         st.stop()

# Normalize DB URL for Neon + SSL
DATABASE_URL = ensure_sslmode_require(force_psycopg_v3(DATABASE_URL))

# Init resources
llm = get_llm(MODEL_ID, GOOGLE_API_KEY)
db = get_db(DATABASE_URL)
agent = make_agent(llm, db)

ok = health_check(db)

# with right_col:
#     st.write("Database:", "✅ Connected" if ok else "❌ Not reachable")
#     show_traces = st.toggle("Show SQL & traces", value=False)
#     if st.button("🗑️ Clear chat"):
#         st.session_state.pop("chat_history", None)
#         st.session_state.pop("last_input", None)
#         st.success("Cleared.")
#         st.rerun()

# ---------------------------
# Right rail: schema + tips
# ---------------------------
with right_col:
    st.divider()
    st.subheader("Schema (preview)")
    schema = read_schema(db)
    if schema:
        for t, cols in schema:
            with st.expander(f"📄 {t}"):
                for c, ty in cols:
                    st.markdown(f"- **{c}** — `{ty}`")
    else:
        st.info("No tables found yet. Expecting `trips` and `riders`.")

    st.divider()
    st.subheader("Try asking…")
    st.markdown(
        "- How many trips were completed last week?\n"
        "- Top 3 `dropoff_address` for riders aged 21–25 on Saturdays\n"
        "- When do large groups (6+ riders) usually travel downtown?"
    )

# ---------------------------
# Tabs: Chat / Explore / SQL
# ---------------------------
body_left, = left_col,

tab_chat, tab_explore, tab_sql = body_left.tabs(["💬 Chat", "🧭 Explore", "🧪 SQL Console (read-only)"])

# Session state (chat)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[dict]: {"role":"user/assistant","content":str,"trace":any}

# --- CHAT TAB ---
with tab_chat:
    # Render existing messages
    for msg in st.session_state.chat_history:
        role = msg["role"]
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            if show_traces and msg.get("trace") and role == "assistant":
                with st.expander("View SQL / trace"):
                    st.write(msg["trace"])

    # Input
    prompt = st.chat_input("Ask a data question…")
    if prompt:
        # Echo user
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Agent answer
        try:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking…"):
                    res = agent.invoke({"input": prompt}, config={"stream": False})
                answer = res.get("output", str(res))
                trace = res.get("intermediate_steps") or res
                st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
                if show_traces and trace:
                    with st.expander("View SQL / trace"):
                        st.write(trace)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "trace": trace}
            )
        except Exception as e:
            # Fallback to direct model
            try:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Agent failed — asking the model directly…"):
                        raw = llm.invoke(prompt)
                        answer = getattr(raw, "content", str(raw))
                        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "trace": None}
                )
            except Exception as inner_e:
                with st.chat_message("assistant", avatar="🤖"):
                    st.error(f"Query failed: {e}; Fallback failed: {inner_e}")

# --- EXPLORE TAB (guided builders) ---
with tab_explore:
    st.markdown("Use these mini-builders to generate common questions quickly.")
    with st.form("builder_form"):
        left, right = st.columns(2)
        with left:
            st.markdown("**Top Drop-offs by Age on Day**")
            age_min = st.number_input("Age min", 0, 120, 18)
            age_max = st.number_input("Age max", 0, 120, 24)
            weekday = st.selectbox(
                "Day of week",
                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
                index=5
            )
            top_k = st.slider("Top K", 1, 20, 5)
        with right:
            st.markdown("**Trip Counts in Date Window**")
            start_date = st.date_input("Start date")
            end_date = st.date_input("End date")
            group_by = st.selectbox("Group by", ["day", "hour", "dow"], index=0)
        build = st.form_submit_button("Generate questions")

    if build:
        q1 = (
            f"Top {top_k} dropoff_address for riders aged {age_min}-{age_max} on {weekday}."
        )
        q2 = (
            f"How many trips from {start_date} to {end_date}? "
            f"Group by {group_by}."
        )
        st.success("Generated! Click to ask:")
        st.code(q1)
        st.code(q2)

# --- SQL CONSOLE TAB (read-only) ---
with tab_sql:
    st.caption("Read-only console. Only `SELECT` / `WITH` queries are allowed.")
    default_sql = "SELECT * FROM trips LIMIT 10;"
    sql = st.text_area("SQL", value=default_sql, height=160, label_visibility="collapsed")
    col_run, col_clear = st.columns([0.2, 0.8])
    if col_run.button("▶️ Run", use_container_width=True):
        if not safe_select(sql):
            st.error("Only SELECT/WITH queries are allowed here.")
        else:
            try:
                with db._engine.connect() as conn:
                    res = conn.execute(text(sql))
                    rows = res.fetchall()
                    cols = res.keys()
                if rows:
                    st.dataframe(
                        [{c: r[i] for i, c in enumerate(cols)} for r in rows],
                        use_container_width=True,
                    )
                    st.caption(f"Returned {len(rows)} rows.")
                else:
                    st.info("No rows returned.")
            except Exception as e:
                st.error(f"SQL error: {e}")
    if col_clear.button("Clear"):
        st.rerun()

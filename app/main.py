# main.py  (Streamlit Cloud + PostgreSQL friendly)
import os
import io
import json
import base64
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import streamlit as st
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# If you keep a separate prompts.py, we'll extend it below
try:
    from prompts import SYSTEM_PROMPT as BASE_SYSTEM_PROMPT
except Exception:
    BASE_SYSTEM_PROMPT = "You are a helpful SQL analyst."

# ---------------------------
# Helpers (PostgreSQL)
# ---------------------------
def force_psycopg_v3(url: str) -> str:
    """Normalize driver to postgresql+psycopg (psycopg3) for SQLAlchemy."""
    if url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

def ensure_sslmode_require(db_url: str) -> str:
    """Append sslmode=require if not already present (needed by Neon/most managed PG)."""
    if not db_url:
        return db_url
    parsed = urlparse(db_url)
    qs = parse_qs(parsed.query)
    if "sslmode" not in qs:
        qs["sslmode"] = ["require"]
        new_q = urlencode(qs, doseq=True)
        db_url = urlunparse(parsed._replace(query=new_q))
    return db_url

@st.cache_resource(show_spinner=False)
def get_db(db_url: str) -> SQLDatabase:
    # Small, resilient pool fits serverless PG + Streamlit Cloud
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
    # Ask the model to return structured JSON the app can render
    RENDER_GUIDE = """
Return ONE of these JSON shapes when presenting results:

- {"type":"table","dataframe":{"columns":[...],"rows":[...]}}
- {"type":"chart","library":"altair","spec":{...}}           # valid Vega-Lite
- {"type":"chart","library":"plotly","fig_json":{...}}       # plotly figure JSON
- {"type":"markdown","markdown":"..."}                       # for prose
- {"type":"image","format":"png","base64":"...","caption":"optional"}

Do NOT include backticks. Ensure valid JSON if using a structured type.
If unsure, reply with {"type":"markdown","markdown":"..."}.
"""
    system_msg = f"{BASE_SYSTEM_PROMPT}\n\n{RENDER_GUIDE}"
    return create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        system_message=system_msg,
        return_intermediate_steps=True,
        handle_parsing_errors=True,  # avoid hard failures on tool output parsing
    )

def health_check_verbose(db: SQLDatabase) -> tuple[bool, str | None]:
    """Return (ok, err_text). Shows the real reason for connection failure."""
    try:
        with db._engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1;")
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# ---------- Rendering ----------
def render_llm_payload(payload: dict):
    """Render a JSON payload from the LLM into Streamlit output."""
    typ = payload.get("type")

    if typ == "table":
        df_spec = payload.get("dataframe", {})
        cols = df_spec.get("columns", [])
        rows = df_spec.get("rows", [])
        df = pd.DataFrame(rows, columns=cols)
        st.dataframe(df, use_container_width=True)

    elif typ == "chart":
        lib = payload.get("library")
        if lib == "altair":
            import altair as alt
            spec = payload.get("spec", {})
            chart = alt.Chart.from_dict(spec)
            st.altair_chart(chart, use_container_width=True)
        elif lib == "plotly":
            import plotly.io as pio
            fig_json = payload.get("fig_json", {})
            fig = pio.from_json(json.dumps(fig_json))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Unsupported chart library: {lib}")

    elif typ == "markdown":
        st.markdown(payload.get("markdown", ""), unsafe_allow_html=False)

    elif typ == "image":
        fmt = payload.get("format", "png")
        b64 = payload.get("base64", "")
        try:
            img_bytes = base64.b64decode(b64)
            st.image(img_bytes, caption=payload.get("caption"), output_format=fmt)
        except Exception as e:
            st.error(f"Could not decode image: {e}")

    else:
        st.info("Unknown or empty payload; showing raw text below.")

def try_render(answer: str, prefer_structured: bool):
    """Try to parse & render structured JSON; fall back to markdown."""
    if not prefer_structured:
        st.markdown(answer)
        return
    try:
        payload = json.loads(answer)
        if isinstance(payload, dict) and "type" in payload:
            render_llm_payload(payload)
        else:
            st.markdown(answer)
    except json.JSONDecodeError:
        st.markdown(answer)

# ---------------------------
# App Config / Secrets
# ---------------------------
st.set_page_config(page_title="Fetii Data Chat (PostgreSQL)", layout="wide")
st.title("🗄️ Fetii Data Chat — PostgreSQL + LangChain SQL Agent")
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

# Normalize DB URL for PostgreSQL + SSL (e.g., Neon)
# Examples:
#   postgresql+psycopg://user:pass@host:5432/dbname?sslmode=require
DATABASE_URL = force_psycopg_v3(DATABASE_URL)
DATABASE_URL = ensure_sslmode_require(DATABASE_URL)

# Init resources
llm = get_llm(MODEL_ID, GOOGLE_API_KEY)
db = get_db(DATABASE_URL)
agent = make_agent(llm, db)

# ---------------------------
# Session state (init early)
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"q": str, "a_raw": str, "trace": any}

# ---------------------------
# Sidebar health + options
# ---------------------------
with st.sidebar:
    st.subheader("Status")
    ok, err = health_check_verbose(db)
    st.write("Database:", "✅ Connected" if ok else "❌ Not reachable")
    if not ok and err:
        st.error(err)  # shows the real error to help fix URL/creds/SSL quickly
    st.caption("💬 Type a question and get instant insights.")

    show_sql = st.toggle("Show SQL / traces", value=False)
    prefer_structured = st.toggle("Smart render (table/chart JSON)", value=True)

    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.success("Conversation cleared.")
        st.rerun()

# ---------------------------
# Conversation history
# ---------------------------
st.subheader("Conversation")
for i, item in enumerate(st.session_state.history, 1):
    st.markdown(f"**Q{i}:** {item['q']}")
    try_render(item["a_raw"], prefer_structured)
    if show_sql and item.get("trace"):
        with st.expander("View SQL / trace"):
            st.write(item["trace"])
    st.divider()

# ---------------------------
# Example questions (printed)
# ---------------------------
st.markdown("✨ **Try asking questions like:**")
st.markdown(
    "- How many trips were completed last week?\n"
    "- Top 3 dropoff_address for riders aged 21–25 on Saturdays\n"
    "- When do large groups (6+ riders) usually travel downtown?"
)

# ---------------------------
# New question form
# ---------------------------
with st.form("ask_form", clear_on_submit=True):
    user_q = st.text_input(
        "Ask a data question (e.g., 'Top 10 dropoff_address for riders aged 18–24 on Saturdays?')",
        key=f"q_{len(st.session_state.history)}",
    )
    submitted = st.form_submit_button("🔎 Ask", type="primary")

if submitted and user_q:
    try:
        with st.spinner("Answering… ⏳"):
            # IMPORTANT: non-streaming for broad model compatibility
            res = agent.invoke({"input": user_q}, config={"stream": False})
            answer_raw = res.get("output", str(res))
            trace = res.get("intermediate_steps") or res  # store any trace/info

        st.session_state.history.append({"q": user_q, "a_raw": answer_raw, "trace": trace})
        st.rerun()
    except Exception as e:
        print(e)
        with st.spinner("Fallback: querying model directly…"):
            try:
                raw = llm.invoke(user_q)
                answer_raw = getattr(raw, "content", str(raw))
                st.session_state.history.append({"q": user_q, "a_raw": answer_raw, "trace": None})
                st.rerun()
            except Exception as inner_e:
                st.error(f"Query failed: {e}; Fallback failed: {inner_e}")

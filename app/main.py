import os
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from prompts import SYSTEM_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI

gemini_key = os.getenv("GOOGLE_API_KEY")
if not gemini_key:
    st.error("Set GOOGLE_API_KEY in .env")
    st.stop()

MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash-lite-preview-06-17")  
# Other options: "gemini-1.5-pro", "gemini-2.5-flash-lite-preview-06-17"

llm = ChatGoogleGenerativeAI(
    model=MODEL_ID,
    temperature=0,
    max_output_tokens=512,
    google_api_key=gemini_key,  # explicitly pass API key
)

st.set_page_config(page_title="Fetii SQL Chat", layout="wide")
st.title("🗄️ Fetii SQL Chat — LangChain SQL Agent (Docker)")

db_url = os.getenv("DATABASE_URL", "postgresql+psycopg://fetii:fetii@db:5432/fetii")
db = SQLDatabase.from_uri(db_url, include_tables=["trips","riders","ride_demo"])

agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    system_message=SYSTEM_PROMPT
)

if "history" not in st.session_state:
    st.session_state.history = []

q = st.text_input("Ask a data question (e.g., 'Top 10 dropoff_address for riders aged 18–24 on Saturdays?')")

if st.button("Ask") or q:
    try:
        # IMPORTANT: pass config with stream=False
        res = agent.invoke({"input": q}, config={"stream": False})
        answer = res.get("output", res)
        print(answer)
        st.write(answer)
    except Exception as e:
        st.error(f"Query failed: {e}")

for i, (qq, aa) in enumerate(reversed(st.session_state.history[-20:]), start=1):
    st.markdown(f"**Q{i}:** {qq}")
    st.markdown(aa)
    st.divider()

# app/ui.py

import os
import sys
import tempfile
import streamlit as st

# ── Path setup ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="AURA-RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GPT-STYLE DARK THEME ───────────────────────────────
st.markdown("""
<style>
:root {
    --bg: #0b0b0f;
    --panel: #111117;
    --border: #2a2a2a;
    --text: #e6e6e6;
    --muted: #9a9a9a;
}

/* Base */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f0f14 !important;
    border-right: 1px solid var(--border);
}

/* Inputs */
input, textarea {
    background-color: #111117 !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

/* Buttons */
button {
    background-color: #1a1a1f !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
button:hover {
    background-color: #2a2a2a !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #2a2a2a;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# ── Imports ────────────────────────────────────────────
from app.components import (
    render_header,
    render_user_bubble,
    render_assistant_bubble,
    render_confidence_bar,
    render_sources,
    render_feedback_row,
    render_ingestion_status,
    render_empty_state,
)
from app.api_client import process_query, submit_feedback, run_ingestion


# ── Session State Init ─────────────────────────────────
def _init():
    defaults = {
        "history": [],
        "last_answer": None,
        "last_query": "",
        "feedback_sent": False,
        "ingestion_result": None,
        "session_id": "ui_session",
        "index_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.title("AURA-RAG")

    api_key = st.text_input(
        "GROQ API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
    )

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF/TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    ingest_btn = st.button("Ingest Documents", disabled=not uploaded_files)

    if ingest_btn and uploaded_files:
        with st.spinner("Processing documents..."):
            tmp_dir = tempfile.mkdtemp()
            paths = []

            for f in uploaded_files:
                path = os.path.join(tmp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())
                paths.append(path)

            result = run_ingestion(paths)

            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        st.session_state.ingestion_result = result
        if result["status"] == "completed":
            st.session_state.index_ready = True

    if st.session_state.ingestion_result:
        render_ingestion_status(st.session_state.ingestion_result)

    st.divider()

    if st.button("Clear History"):
        st.session_state.history = []
        st.session_state.last_answer = None
        st.session_state.feedback_sent = False
        st.rerun()


# ═══════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════
render_header()

if not st.session_state.history:
    render_empty_state()
else:
    for turn in st.session_state.history:
        if turn["role"] == "user":
            render_user_bubble(turn["content"])
        else:
            meta = turn.get("meta", {})
            render_assistant_bubble(
                answer=turn["content"],
                confidence=meta.get("confidence", 0.0),
                status=meta.get("status", "success"),
            )
            render_confidence_bar(meta.get("confidence", 0.0))
            render_sources(meta.get("sources", []))


# ═══════════════════════════════════════════════════════
# FEEDBACK
# ═══════════════════════════════════════════════════════
if (
    st.session_state.last_answer
    and not st.session_state.feedback_sent
    and st.session_state.history
    and st.session_state.history[-1]["role"] == "assistant"
):
    fb_type, fb_text, submitted = render_feedback_row()

    if submitted and fb_type:
        with st.spinner("Saving feedback..."):
            submit_feedback(
                query=st.session_state.last_query,
                answer=st.session_state.last_answer["answer"],
                feedback_type=fb_type,
                feedback_text=fb_text,
                session_id=st.session_state.session_id,
            )
        st.session_state.feedback_sent = True
        st.success("Feedback recorded")


# ═══════════════════════════════════════════════════════
# INPUT
# ═══════════════════════════════════════════════════════
with st.form("query_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])

    query = col1.text_input(
        "Ask something...",
        label_visibility="collapsed",
    )

    send = col2.form_submit_button("→")

if send and query.strip():
    if not api_key:
        st.error("Set API key first")
    elif not st.session_state.index_ready:
        st.warning("Upload and ingest documents first")
    else:
        st.session_state.history.append({
            "role": "user",
            "content": query
        })
        st.session_state.last_query = query
        st.session_state.feedback_sent = False

        with st.spinner("Thinking..."):
            response = process_query(query)

        st.session_state.history.append({
            "role": "assistant",
            "content": response["answer"],
            "meta": {
                "confidence": response["confidence"],
                "status": response["status"],
                "sources": response["sources"],
            },
        })

        st.session_state.last_answer = response
        st.rerun()
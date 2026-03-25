"""
app/components.py
-----------------
Reusable HTML/CSS rendering helpers for the AURA-RAG Streamlit UI.
All functions return or render styled markdown — no backend calls here.
"""

import streamlit as st


# ── Design tokens ────────────────────────────────────────────────────────────

THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;700&display=swap');

:root {
    --bg:       #080810;
    --surface:  #0f0f1a;
    --border:   #1c1c2e;
    --accent:   #4fffb0;
    --accent2:  #ff6ef7;
    --warn:     #ffb347;
    --text:     #dde0f0;
    --muted:    #4a4a6a;
    --mono:     'IBM Plex Mono', monospace;
    --display:  'Syne', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--display) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    padding: 1.5rem 1rem !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Inputs */
textarea, input[type="text"], input[type="password"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.88rem !important;
}
textarea:focus, input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79,255,176,0.1) !important;
    outline: none !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    border-radius: 3px !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

/* Progress bar */
.stProgress > div > div {
    background: var(--accent) !important;
}
.stProgress > div {
    background: var(--border) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.06em !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 3px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
"""


# ── Component renderers ──────────────────────────────────────────────────────

def render_theme():
    """Inject global CSS. Call once at top of ui.py."""
    st.markdown(THEME, unsafe_allow_html=True)


def render_header():
    """Top-of-page title bar."""
    st.markdown("""
    <div style="
        padding: 1.4rem 2rem 0.8rem 2rem;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: baseline;
        gap: 1rem;
        margin-bottom: 1rem;
    ">
        <span style="
            font-family: var(--mono);
            font-size: 1rem;
            font-weight: 600;
            color: var(--accent);
            letter-spacing: 0.12em;
        ">◈ AURA-RAG</span>
        <span style="
            font-family: var(--mono);
            font-size: 0.65rem;
            color: var(--muted);
            letter-spacing: 0.08em;
        ">Adaptive Understanding & Retrieval Architecture</span>
    </div>
    """, unsafe_allow_html=True)


def render_user_bubble(text: str):
    """Render a user message in the chat history."""
    st.markdown(f"""
    <div style="
        display: flex;
        justify-content: flex-end;
        margin: 0.5rem 2rem 0.5rem 4rem;
    ">
        <div style="
            background: var(--border);
            border-radius: 3px 3px 0 3px;
            padding: 0.65rem 1rem;
            font-size: 0.88rem;
            line-height: 1.6;
            max-width: 80%;
            color: var(--text);
        ">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_assistant_bubble(answer: str, confidence: float, status: str):
    """Render an assistant answer with confidence badge."""
    conf_pct  = int(confidence * 100)
    badge_col = "#4fffb0" if status == "success" else "#ffb347"
    status_label = "GROUNDED" if status == "success" else "FALLBACK"

    st.markdown(f"""
    <div style="
        margin: 0.5rem 4rem 0.5rem 2rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid {badge_col};
        border-radius: 0 3px 3px 3px;
        padding: 1rem 1.2rem;
    ">
        <div style="
            font-family: var(--mono);
            font-size: 0.62rem;
            letter-spacing: 0.1em;
            color: {badge_col};
            margin-bottom: 0.6rem;
        ">{status_label} · conf {conf_pct}%</div>
        <div style="
            font-size: 0.9rem;
            line-height: 1.75;
            color: var(--text);
        ">{answer}</div>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_bar(confidence: float):
    """Render a labeled confidence progress bar."""
    conf_pct = int(confidence * 100)
    col_label = "#4fffb0" if confidence >= 0.75 else ("#ffb347" if confidence >= 0.45 else "#ff6ef7")
    st.markdown(f"""
    <div style="margin: 0.3rem 2rem 0 2rem;">
        <div style="
            font-family: var(--mono);
            font-size: 0.62rem;
            color: var(--muted);
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        ">CONFIDENCE
            <span style="color:{col_label}; margin-left:0.5rem">{conf_pct}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(confidence)


def render_sources(sources: list):
    """Collapsible section showing retrieved chunks."""
    if not sources:
        return
    with st.expander(f"  Retrieved context  ({len(sources)} chunks)", expanded=False):
        for i, chunk in enumerate(sources, 1):
            st.markdown(f"""
            <div style="
                background: var(--bg);
                border: 1px solid var(--border);
                border-radius: 3px;
                padding: 0.7rem 0.9rem;
                margin-bottom: 0.5rem;
                font-family: var(--mono);
                font-size: 0.75rem;
                line-height: 1.65;
                color: var(--muted);
            ">
                <span style="color:var(--accent2);margin-right:0.5rem">chunk_{i:02d}</span>
                {chunk[:400]}{'…' if len(chunk) > 400 else ''}
            </div>
            """, unsafe_allow_html=True)


def render_feedback_row():
    """
    Inline feedback row — returns (feedback_type, feedback_text, submitted).
    Call after displaying an answer.
    """
    st.markdown("""
    <div style="
        font-family: var(--mono);
        font-size: 0.62rem;
        letter-spacing: 0.1em;
        color: var(--muted);
        margin: 0.3rem 2rem 0.2rem 2rem;
    ">RATE THIS ANSWER</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 4])
    helpful     = col1.button("👍  Helpful",     key="fb_helpful")
    not_helpful = col2.button("👎  Not Helpful", key="fb_not_helpful")
    fb_text     = col3.text_input(
        "comment",
        placeholder="Optional comment…",
        label_visibility="collapsed",
        key="fb_text",
    )

    if helpful:
        return "helpful", fb_text, True
    if not_helpful:
        return "not_helpful", fb_text, True
    return None, fb_text, False


def render_ingestion_status(result: dict):
    """Show ingestion result card."""
    if result["status"] == "completed":
        st.markdown(f"""
        <div style="
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 3px;
            padding: 0.8rem 1rem;
            font-family: var(--mono);
            font-size: 0.75rem;
            line-height: 1.8;
        ">
            <div style="color:var(--accent);letter-spacing:0.1em">✓ INGESTION COMPLETE</div>
            <div style="color:var(--muted)">{result['documents_processed']} file(s) indexed into FAISS + Chroma</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent2);
            border-radius: 3px;
            padding: 0.8rem 1rem;
            font-family: var(--mono);
            font-size: 0.75rem;
        ">
            <div style="color:var(--accent2);letter-spacing:0.1em">✗ INGESTION FAILED</div>
            <div style="color:var(--muted)">{result['message']}</div>
        </div>
        """, unsafe_allow_html=True)


def render_empty_state():
    """Shown when chat history is empty."""
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        color: var(--muted);
    ">
        <div style="
            font-family: var(--mono);
            font-size: 2rem;
            margin-bottom: 0.8rem;
            opacity: 0.3;
        ">◈</div>
        <div style="
            font-family: var(--mono);
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        ">Ask anything. AURA will retrieve, rank, and reason.</div>
    </div>
    """, unsafe_allow_html=True)

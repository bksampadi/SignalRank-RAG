import html
import os
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import logfire
import requests
import streamlit as st

SERVICE_TOKEN = os.getenv("SIGNALRANK_SERVICE_TOKEN")

API_URL = os.getenv(
    "SIGNALRANK_API_URL",
    "http://127.0.0.1:8000",
)


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="SignalRank-RAG",
    page_icon="◌",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

st.markdown(
    """
<style>
:root {
    --sr-bg: #FAFAF7;
    --sr-surface: #F4F5F1;
    --sr-ink: #29322D;
    --sr-muted: #728078;
    --sr-signal: #61766A;
    --sr-border: rgba(97, 118, 106, 0.16);
    --sr-border-strong: rgba(97, 118, 106, 0.26);
    --sr-radius-sm: 10px;
    --sr-radius: 14px;
    --sr-radius-lg: 20px;
}

html,
body,
[data-testid="stAppViewContainer"] {
    background: var(--sr-bg);
    color: var(--sr-ink);
}

.block-container {
    max-width: 900px;
    padding-top: 4rem;
    padding-bottom: 7.5rem;
}

/* Streamlit chrome */
#MainMenu,
footer,
[data-testid="stToolbar"] {
    visibility: hidden;
}

/* ------------------------------------------------------------------
   Product masthead
   ------------------------------------------------------------------ */

.st-key-brand_header {
    position: sticky;
    top: 2.75rem;
    z-index: 100;
    background: rgba(250, 250, 247, 0.94);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--sr-border);
    padding: 0.65rem 0 0.8rem 0;
    margin-bottom: 0.35rem;
}

.signalrank-brandline {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.signalrank-brand {
    font-size: 1.08rem;
    font-weight: 780;
    letter-spacing: -0.03em;
    line-height: 1;
}

.signalrank-brand span {
    opacity: 0.42;
    font-weight: 620;
}

.signalrank-mark {
    width: 9px;
    height: 9px;
    border: 1.5px solid var(--sr-signal);
    border-radius: 999px;
    box-shadow: inset 0 0 0 2px var(--sr-bg);
    background: var(--sr-signal);
}

.signalrank-tagline {
    margin-top: 0.28rem;
    font-size: 0.78rem;
    color: var(--sr-muted);
    letter-spacing: 0.01em;
}

/* ------------------------------------------------------------------
   Landing state
   ------------------------------------------------------------------ */

.signalrank-empty {
    text-align: center;
    padding-top: clamp(1.4rem, 4vh, 3rem);
    padding-bottom: 1.35rem;
}

.signalrank-empty h1 {
    max-width: 720px;
    margin: 0 auto;
    font-size: clamp(2.65rem, 6.2vw, 4.65rem);
    line-height: 0.98;
    letter-spacing: -0.058em;
    font-weight: 790;
}

.signalrank-empty h1 span {
    color: var(--sr-signal);
}

/* ------------------------------------------------------------------
   Branded chat turns
   ------------------------------------------------------------------ */

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    display: none;
}

[data-testid="stChatMessage"] {
    gap: 0;
    background: transparent;
    border: 0;
    padding-top: 1rem;
    padding-bottom: 1rem;
}

[data-testid="stChatMessageContent"] {
    font-size: 1rem;
    line-height: 1.72;
    width: 100%;
}

[data-testid="stChatMessageContent"] p {
    line-height: 1.72;
}

[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) {
    position: relative;
    padding-left: 1.15rem;
    border-left: 2px solid rgba(97, 118, 106, 0.28);
}

[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) {
    padding-left: 1.15rem;
    opacity: 0.92;
}

.signalrank-role {
    margin-bottom: 0.42rem;
    font-size: 0.67rem;
    font-weight: 760;
    letter-spacing: 0.115em;
    text-transform: uppercase;
    color: var(--sr-signal);
}

.signalrank-role-query {
    color: var(--sr-muted);
    opacity: 0.78;
}

/* ------------------------------------------------------------------
   Light alignment for existing controls
   ------------------------------------------------------------------ */

[data-testid="stChatInput"] {
    border-radius: var(--sr-radius-lg);
    border-color: var(--sr-border-strong);
    box-shadow: 0 10px 30px rgba(41, 50, 45, 0.06);
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(97, 118, 106, 0.48);
    box-shadow: 0 10px 34px rgba(41, 50, 45, 0.08);
}

div[data-testid="stButton"] > button,
div[data-testid="stPopover"] button {
    border-radius: var(--sr-radius-sm);
    border-color: var(--sr-border);
}

div[data-testid="stExpander"] {
    border-radius: var(--sr-radius);
    border-color: var(--sr-border);
}

@media (max-width: 640px) {
    .block-container {
        padding-top: 0.75rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .signalrank-empty {
        padding-top: 3.2rem;
    }

    .signalrank-empty h1 {
        font-size: clamp(2.5rem, 13vw, 3.6rem);
    }
}
</style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------


@st.cache_resource
def configure_logfire():
    try:
        logfire.configure(
            send_to_logfire="if-token-present",
        )
        logfire.instrument_requests()
        return True

    # Observability is optional; initialization failure must not block the UI.
    except Exception as exc:  # noqa: BLE001
        print(f"Logfire initialization failed: {exc}")
        return False


LOGFIRE_ENABLED = configure_logfire()


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "turns" not in st.session_state:
    st.session_state.turns = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = "Dense"

if "top_k" not in st.session_state:
    st.session_state.top_k = 2

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def clear_chat() -> None:
    st.session_state.turns = []
    st.session_state.feedback = {}
    st.session_state.pending_query = None


# ---------------------------------------------------------------------
# Product language
# ---------------------------------------------------------------------

WARMUP_MESSAGES = (
    "◌ Waking up the evidence engine…",
    "◌ Loading the retrieval stack…",
    "◌ Calibrating the rankers…",
    "◌ Searching the universe…",
    "◌ Almost there…",
)

GREETING_RESPONSES = {
    "hi": "Hi — what would you like to investigate?",
    "hello": "Hello — what would you like to investigate?",
    "hey": "Hey — what would you like to investigate?",
    "hi!": "Hi — what would you like to investigate?",
    "hello!": "Hello — what would you like to investigate?",
    "hey!": "Hey — what would you like to investigate?",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def get_http_error_detail(
    response: requests.Response | None,
) -> str | None:
    if response is None:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    detail = payload.get("detail")

    if isinstance(detail, str):
        return detail

    return None


def get_greeting_response(query: str) -> str | None:
    return GREETING_RESPONSES.get(query.strip().lower())


def format_source_count(count: int) -> str:
    noun = "source" if count == 1 else "sources"
    return f"{count} {noun}"


def wait_for_backend(
    warmup_notice,
    attempts: int = 30,
    delay_seconds: float = 1.0,
) -> bool:
    for attempt in range(attempts):
        try:
            response = requests.get(
                f"{API_URL}/health",
                timeout=1.0,
            )

            if response.ok:
                warmup_notice.empty()
                return True

        except requests.RequestException:
            pass

        message = WARMUP_MESSAGES[attempt % len(WARMUP_MESSAGES)]
        warmup_notice.caption(message)
        time.sleep(delay_seconds)

    warmup_notice.empty()
    return False


def format_score(score: float) -> str:
    if 0 < score < 0.001:
        return f"{score:.1e}"

    return f"{score:.3f}"


def render_role(label: str, *, query: bool = False) -> None:
    role_class = " signalrank-role-query" if query else ""
    safe_label = html.escape(label)

    st.markdown(
        f'<div class="signalrank-role{role_class}">{safe_label}</div>',
        unsafe_allow_html=True,
    )


def request_chat(
    query: str,
    mode: str,
    top_k: int,
) -> tuple[dict | None, str | None]:
    search_id = str(uuid.uuid4())

    span_context = (
        logfire.span(
            "chat request",
            session_id=st.session_state.session_id,
            search_id=search_id,
            retrieval_mode=mode.lower(),
            top_k=top_k,
            query_length=len(query),
        )
        if LOGFIRE_ENABLED
        else nullcontext()
    )

    started_at = time.perf_counter()

    with span_context:
        try:
            service_token = SERVICE_TOKEN

            if not service_token:
                raise RuntimeError("SIGNALRANK_SERVICE_TOKEN is not configured.")

            response = requests.post(
                f"{API_URL}/chat",
                headers={
                    "X-SignalRank-Service-Token": service_token,
                },
                json={
                    "query": query,
                    "mode": mode.lower(),
                    "top_k": top_k,
                },
                timeout=(5, 120),
            )

            response.raise_for_status()
            data = response.json()

        except requests.Timeout as exc:
            if LOGFIRE_ENABLED:
                logfire.error(
                    "Chat request timed out",
                    error=str(exc),
                    session_id=st.session_state.session_id,
                    search_id=search_id,
                )

            return (
                None,
                (
                    "The request timed out. "
                    "The first request can take longer while the service warms up."
                ),
            )

        except requests.ConnectionError as exc:
            if LOGFIRE_ENABLED:
                logfire.error(
                    "Chat service connection failed",
                    error=str(exc),
                    session_id=st.session_state.session_id,
                    search_id=search_id,
                )

            return (
                None,
                "SignalRank could not reach the service. Please try again shortly.",
            )

        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            detail = get_http_error_detail(exc.response)

            if LOGFIRE_ENABLED:
                logfire.error(
                    "Chat service returned HTTP error",
                    error=str(exc),
                    status_code=status_code,
                    detail=detail,
                    session_id=st.session_state.session_id,
                    search_id=search_id,
                )

            if status_code == 401:
                return (
                    None,
                    "SignalRank reached the service, but authentication failed.",
                )

            if status_code == 503:
                return (
                    None,
                    (
                        "LLM generation is temporarily unavailable. "
                        "SignalRank's retrieval service remains available."
                    ),
                )

            if status_code and status_code >= 500:
                return (
                    None,
                    (
                        "SignalRank reached the service, but the backend "
                        "encountered an error."
                    ),
                )

            return (
                None,
                f"The service returned HTTP {status_code or 'error'}.",
            )

        except requests.RequestException as exc:
            if LOGFIRE_ENABLED:
                logfire.error(
                    "Chat request failed",
                    error=str(exc),
                    session_id=st.session_state.session_id,
                    search_id=search_id,
                )

            return (
                None,
                "The request could not be completed.",
            )

        except RuntimeError as exc:
            return None, str(exc)

    elapsed_seconds = time.perf_counter() - started_at

    turn = {
        "search_id": search_id,
        "query": query,
        "mode": mode.lower(),
        "top_k": top_k,
        "route": data.get("route", "unknown"),
        "answer": data.get("answer"),
        "results": data.get("results", []),
        "elapsed_seconds": elapsed_seconds,
        "degraded": data.get("degraded", False),
        "effective_response_mode": data.get("effective_response_mode"),
    }

    return turn, None


# ---------------------------------------------------------------------
# Evidence + feedback
# ---------------------------------------------------------------------


def render_feedback(
    turn: dict,
    result: dict,
) -> None:
    chunk_id = result.get(
        "chunk_id",
        f"rank-{result.get('rank', 0)}",
    )

    feedback_key = f"{turn['search_id']}:{chunk_id}"
    recorded_feedback = st.session_state.feedback.get(feedback_key)

    relevant_col, not_relevant_col, _ = st.columns([1, 1.2, 4])

    relevant = relevant_col.button(
        "Relevant",
        key=f"relevant-{feedback_key}",
        disabled=recorded_feedback is not None,
    )

    not_relevant = not_relevant_col.button(
        "Not relevant",
        key=f"not-relevant-{feedback_key}",
        disabled=recorded_feedback is not None,
    )

    if relevant or not_relevant:
        relevance = "relevant" if relevant else "not_relevant"
        st.session_state.feedback[feedback_key] = relevance

        if LOGFIRE_ENABLED:
            logfire.info(
                "retrieval result feedback",
                session_id=st.session_state.session_id,
                search_id=turn["search_id"],
                query=turn["query"],
                query_length=len(turn["query"]),
                retrieval_mode=turn["mode"],
                chunk_id=chunk_id,
                doc_id=result.get("doc_id"),
                source_path=result.get("source_path"),
                rank=result.get("rank"),
                relevance=relevance,
            )

        st.rerun()

    if recorded_feedback:
        label = "Relevant" if recorded_feedback == "relevant" else "Not relevant"
        st.caption(f"Feedback recorded · {label}")


def render_evidence(turn: dict) -> None:
    results = turn["results"]

    for index, result in enumerate(results, start=1):
        rank = result.get("rank", index)
        score = float(result.get("score", 0.0))
        source_path = result.get("source_path", "")
        source_name = Path(source_path).name if source_path else "Unknown source"

        source_col, score_col = st.columns([5, 1])

        with source_col:
            st.markdown(f"**{rank}. {source_name}**")

        with score_col:
            st.caption(format_score(score))

        st.markdown(result.get("text", ""))

        if source_path:
            st.caption(source_path)

        st.write("")
        render_feedback(turn, result)

        if index < len(results):
            st.divider()


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        render_role("Query", query=True)
        st.markdown(turn["query"])

    with st.chat_message("assistant"):
        render_role("SignalRank")

        answer = turn.get("answer")

        if answer:
            st.markdown(answer)
        else:
            st.markdown("SignalRank couldn't produce an answer for that request.")

        route = turn.get("route", "unknown")
        results = turn.get("results", [])

        if route == "conversation":
            return

        if results:
            mode_label = turn["mode"].title()
            st.caption(f"{mode_label} · {format_source_count(len(results))}")

            with st.expander("Inspect evidence"):
                render_evidence(turn)
        else:
            st.caption("No supporting evidence retrieved")


# ---------------------------------------------------------------------
# Retrieval controls
# ---------------------------------------------------------------------


def render_retrieval_controls() -> None:
    with st.popover("Tune"):
        st.segmented_control(
            "Mode",
            options=[
                "Dense",
                "BM25",
                "Hybrid",
            ],
            key="retrieval_mode",
        )

        st.select_slider(
            "Sources",
            options=[1, 3, 5, 7, 10],
            key="top_k",
        )


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

with st.container(key="brand_header"):
    header_col, action_col = st.columns(
        [4.7, 1.6],
        vertical_alignment="center",
    )

    with header_col:
        st.markdown(
            """
<div class="signalrank-brandline">
    <span class="signalrank-mark"></span>
    <div class="signalrank-brand">SignalRank<span>-RAG</span></div>
</div>
<div class="signalrank-tagline">Retrieval, reranking & grounded responses.</div>
            """,
            unsafe_allow_html=True,
        )

    with action_col:
        if st.session_state.turns:
            tune_col, new_chat_col = st.columns(2)

            with tune_col:
                render_retrieval_controls()

            with new_chat_col:
                st.button(
                    "New",
                    on_click=clear_chat,
                    width="stretch",
                    type="tertiary",
                    help="Start a new conversation.",
                )

        else:
            render_retrieval_controls()


# ---------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------

example_query = None
typed_query = None

pending_query = st.session_state.pending_query

if pending_query:
    query = str(pending_query["query"])
    selected_mode = str(pending_query["mode"])
    selected_top_k = int(pending_query["top_k"])

    st.write("")

    with st.chat_message("user"):
        render_role("Query", query=True)
        st.markdown(query)

    with st.chat_message("assistant"):
        render_role("SignalRank")

        warmup_notice = st.empty()
        warmup_notice.caption("◌ Connecting to SignalRank…")

        backend_ready = wait_for_backend(warmup_notice)
        warmup_notice.empty()

        if not backend_ready:
            st.session_state.pending_query = None
            st.info(
                "SignalRank is taking unusually long to wake up. "
                "Please try again in a moment."
            )
            st.stop()

        with st.spinner("Searching and ranking evidence…"):
            turn, error = request_chat(
                query=query,
                mode=selected_mode,
                top_k=selected_top_k,
            )

        st.session_state.pending_query = None

        if error:
            st.info(error)
            st.stop()

        if turn:
            st.session_state.turns.append(turn)
            st.rerun()

elif not st.session_state.turns:
    st.markdown(
        """
<div class="signalrank-empty">
    <h1>Find the signal.</h1>
</div>
        """,
        unsafe_allow_html=True,
    )

    # st.chat_input becomes inline when placed inside a layout container.
    with st.container():
        typed_query = st.chat_input(
            "Ask SignalRank…",
            key="empty_chat_input",
        )

    st.write("")

    example_left, example_right = st.columns(2)

    if example_left.button(
        "🦕  Dinosaur extinction",
        help="Supported by the demo corpus.",
        width="stretch",
    ):
        example_query = "What caused the extinction of the dinosaurs?"

    if example_right.button(
        "☕  Making coffee",
        help="Intentionally absent from the demo corpus.",
        width="stretch",
    ):
        example_query = "How do I make coffee?"



# ---------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------

else:
    st.write("")

    for turn in st.session_state.turns:
        render_turn(turn)

    st.write("")

    # Main-body chat input stays pinned to the bottom.
    typed_query = st.chat_input(
        "Ask SignalRank…",
        key="conversation_chat_input",
    )


# ---------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------

submitted_query = typed_query or example_query

if submitted_query:
    query = submitted_query.strip()

    if query:
        selected_mode = st.session_state.retrieval_mode or "Dense"
        selected_top_k = st.session_state.top_k

        greeting_response = get_greeting_response(query)

        if greeting_response:
            turn = {
                "search_id": str(uuid.uuid4()),
                "query": query,
                "mode": selected_mode.lower(),
                "top_k": selected_top_k,
                "route": "conversation",
                "answer": greeting_response,
                "results": [],
                "elapsed_seconds": 0.0,
                "degraded": False,
                "effective_response_mode": None,
            }

            st.session_state.turns.append(turn)
            st.rerun()

        st.session_state.pending_query = {
            "query": query,
            "mode": selected_mode,
            "top_k": selected_top_k,
        }

        st.rerun()

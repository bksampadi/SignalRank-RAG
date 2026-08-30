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
.block-container {
    max-width: 860px;
    padding-top: 3.5rem;
    padding-bottom: 7rem;
}

/* Streamlit chrome */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

[data-testid="stToolbar"] {
    visibility: hidden;
}

/* Brand */
.signalrank-brand {
    font-size: 1.15rem;
    font-weight: 760;
    letter-spacing: -0.025em;
    margin: 0;
}

.signalrank-brand span {
    opacity: 0.45;
    font-weight: 600;
}

.signalrank-tagline {
    margin-top: 0.15rem;
    font-size: 0.82rem;
    opacity: 0.5;
}

/* Empty-state landing */
.signalrank-empty {
    text-align: center;
    padding-top: clamp(3rem, 10vh, 7rem);
    padding-bottom: 1.5rem;
}

.signalrank-empty h1 {
    font-size: clamp(2.4rem, 6vw, 4rem);
    line-height: 1.03;
    letter-spacing: -0.05em;
    margin: 0;
    font-weight: 760;
}

.signalrank-empty p {
    max-width: 560px;
    margin: 1rem auto 0;
    font-size: 1.05rem;
    line-height: 1.6;
    opacity: 0.58;
}

/* Quiet technical metadata */
.signalrank-meta {
    font-size: 0.78rem;
    opacity: 0.5;
    margin-top: 0.45rem;
}

/* Chat */
[data-testid="stChatMessage"] {
    background: transparent;
    border: 0;
    padding-top: 0.75rem;
    padding-bottom: 0.75rem;
}

[data-testid="stChatMessageContent"] {
    font-size: 1rem;
    line-height: 1.7;
}

[data-testid="stChatMessageContent"] p {
    line-height: 1.7;
}

/* Input */
[data-testid="stChatInput"] {
    border-radius: 18px;
}

/* Buttons */
div[data-testid="stButton"] > button {
    border-radius: 10px;
}

/* Expanders */
div[data-testid="stExpander"] {
    border-radius: 12px;
}

/* Retrieval controls */
.retrieval-status {
    padding-top: 0.4rem;
    font-size: 0.78rem;
    opacity: 0.5;
}

/* Evidence */
.evidence-source {
    font-size: 0.95rem;
    font-weight: 650;
}

.evidence-path {
    font-size: 0.76rem;
    opacity: 0.45;
}

.evidence-score {
    font-size: 0.76rem;
    opacity: 0.55;
}

hr {
    opacity: 0.1;
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
    st.session_state.top_k = 5


def clear_chat() -> None:
    st.session_state.turns = []
    st.session_state.feedback = {}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

WARMUP_MESSAGES = (
    "◌ Waking up the evidence engine…",
    "◌ Loading the retrieval stack…",
    "◌ Calibrating the rankers…",
    "◌ Searching the universe…",
    "◌ Almost there…",
)

GREETING_RESPONSES = {
    "hi": "Hi! How can I help you today?",
    "hello": "Hello! How can I help you today?",
    "hey": "Hey! What would you like to explore?",
    "hi!": "Hi! How can I help you today?",
    "hello!": "Hello! How can I help you today?",
    "hey!": "Hey! What would you like to explore?",
}


def get_greeting_response(
    query: str,
) -> str | None:
    return GREETING_RESPONSES.get(
        query.strip().lower()
    )

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

        message = WARMUP_MESSAGES[
            attempt % len(WARMUP_MESSAGES)
        ]

        warmup_notice.caption(message)
        time.sleep(delay_seconds)

    warmup_notice.empty()
    return False

def format_score(score: float) -> str:
    if 0 < score < 0.001:
        return f"{score:.1e}"

    return f"{score:.3f}"

def post_chat(
    query: str,
    mode: str,
    top_k: int,
    service_token: str,
) -> requests.Response:
    return requests.post(
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
                raise RuntimeError(
                    "SIGNALRANK_SERVICE_TOKEN is not configured."
                )

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
                "SignalRank could not connect to the local retrieval service.",
            )

        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None

            if LOGFIRE_ENABLED:
                logfire.error(
                    "Chat service returned HTTP error",
                    error=str(exc),
                    status_code=status_code,
                    session_id=st.session_state.session_id,
                    search_id=search_id,
                )

            if status_code == 401:
                return (
                    None,
                    "SignalRank reached the service, but authentication failed.",
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
    }

    return turn, None


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

    relevant_col, not_relevant_col, _ = st.columns([1.1, 1.35, 3])

    relevant = relevant_col.button(
        "👍 Relevant",
        key=f"relevant-{feedback_key}",
        disabled=recorded_feedback is not None,
        width="stretch",
    )

    not_relevant = not_relevant_col.button(
        "👎 Not relevant",
        key=f"not-relevant-{feedback_key}",
        disabled=recorded_feedback is not None,
        width="stretch",
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
            st.markdown(
                f'<div class="evidence-source">{rank}. {source_name}</div>',
                unsafe_allow_html=True,
            )

        with score_col:
            st.markdown(
                f'<div class="evidence-score" '
                f'style="text-align:right;">'
                f"{format_score(score)}"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown(result.get("text", ""))

        if source_path:
            st.markdown(
                f'<div class="evidence-path">{source_path}</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        render_feedback(turn, result)

        if index < len(results):
            st.divider()


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.markdown(turn["query"])

    with st.chat_message("assistant"):
        answer = turn.get("answer")

        if answer:
            st.markdown(answer)
        else:
            st.markdown("I couldn't produce an answer for that request.")

        route = turn.get("route", "unknown")
        results = turn.get("results", [])

        # Conversation should feel like conversation.
        # Don't expose retrieval machinery when none was used.
        if route == "conversation":
            return

        if results:
            mode_label = turn["mode"].title()
            elapsed = turn["elapsed_seconds"]

            st.markdown(
                f'<div class="signalrank-meta">'
                f"{mode_label} · "
                f"{len(results)} sources · "
                f"{elapsed:.2f}s"
                f"</div>",
                unsafe_allow_html=True,
            )

            with st.expander(f"Inspect evidence · {len(results)} sources"):
                render_evidence(turn)

        else:
            st.markdown(
                '<div class="signalrank-meta">No supporting evidence retrieved</div>',
                unsafe_allow_html=True,
            )


def render_retrieval_controls() -> None:
    control_col, status_col = st.columns(
        [1.4, 4.6],
        vertical_alignment="center",
    )

    with control_col, st.popover("⚙ Retrieval"):
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

    with status_col:
        mode = st.session_state.retrieval_mode or "Dense"

        st.markdown(
            f'<div class="retrieval-status">'
            f"{mode} · "
            f"{st.session_state.top_k} sources"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

header_col, action_col = st.columns(
    [5, 1],
    vertical_alignment="center",
)

with header_col:
    st.markdown(
        """
<div class="signalrank-brand">
SignalRank<span>-RAG</span>
</div>
<div class="signalrank-tagline">
Evidence you can inspect.
</div>
        """,
        unsafe_allow_html=True,
    )

with action_col:
    if st.session_state.turns:
        st.button(
            "New chat",
            on_click=clear_chat,
            width="stretch",
        )


# ---------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------

example_query = None
typed_query = None

if not st.session_state.turns:
    st.markdown(
        """
<div class="signalrank-empty">
<h1>Find the evidence<br>that matters.</h1>
<p>
Ask naturally. SignalRank decides whether to answer conversationally
or retrieve, rerank, and ground the response in inspectable evidence.
</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Inline input for the landing page.
    with st.container():
        typed_query = st.chat_input(
            "Ask SignalRank...",
            key="empty_chat_input",
        )

    render_retrieval_controls()

    st.write("")

    example_left, example_right = st.columns(2)

    if example_left.button(
        "🦕  What caused dinosaur extinction?",
        width="stretch",
    ):
        example_query = "What caused the extinction of the dinosaurs?"

    if example_right.button(
        "☕  How do I make coffee?",
        width="stretch",
    ):
        example_query = "How do I make coffee?"

    st.caption(
        "The dinosaur question is supported by the demo corpus. "
        "Coffee is intentionally absent."
    )

    with st.expander("About the demo corpus"):
        st.markdown(
            """
**39 benchmark documents**

**Science & space**  
Aliens · climate observations · dinosaurs · lunar rovers ·
Mars · Mars orbiters · oceans · radio astronomy · solar energy ·
unexplained signals · volcanoes · weather · weather forecasting

**Biology & medicine**  
Antibiotics · antiviral resistance · immune escape · vaccines

**Energy**  
Batteries · battery management · grid storage

**Computing & machine learning**  
Machine-learning training · neural networks · optimization ·
Python · retrieval · static typing · type systems

**History & mythology**  
Pharaohs · Greek city-states · cults · festivals · gods ·
heroes · monsters · oracles · poetry · sacrifices · temples ·
underworld
            """
        )

        st.caption(
            "Demo searches and relevance feedback may be logged "
            "for evaluation. Please do not enter sensitive information."
        )


# ---------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------

else:
    st.write("")

    for turn in st.session_state.turns:
        render_turn(turn)

    st.write("")
    render_retrieval_controls()

    # Main-body chat input stays pinned to the bottom.
    typed_query = st.chat_input(
        "Ask SignalRank...",
        key="conversation_chat_input",
    )


# ---------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------

submitted_query = typed_query or example_query

if submitted_query:
    query = submitted_query.strip()

    if query:
        selected_mode = (
            st.session_state.retrieval_mode
            or "Dense"
        )
        selected_top_k = st.session_state.top_k

        # Show the user message immediately.
        with st.chat_message("user"):
            st.markdown(query)

        # -------------------------------------------------------------
        # Fast path: obvious greetings need no backend or LLM.
        # -------------------------------------------------------------

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
            }

            st.session_state.turns.append(turn)
            st.rerun()

        # -------------------------------------------------------------
        # Everything else uses SignalRank.
        # -------------------------------------------------------------

        with st.chat_message("assistant"):
            warmup_notice = st.empty()

            backend_ready = wait_for_backend(
                warmup_notice,
            )

            warmup_notice.empty()

            if not backend_ready:
                st.info(
                    "SignalRank is taking unusually long to wake up. "
                    "Please try again in a moment."
                )

            else:
                with st.spinner("Thinking..."):
                    turn, error = request_chat(
                        query=query,
                        mode=selected_mode,
                        top_k=selected_top_k,
                    )

                if error:
                    st.info(error)

                elif turn:
                    st.session_state.turns.append(turn)
                    st.rerun()
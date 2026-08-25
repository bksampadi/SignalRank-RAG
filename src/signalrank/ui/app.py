import os
import uuid

import logfire
import requests
import streamlit as st

API_URL = os.getenv(
    "SIGNALRANK_API_URL",
    "http://127.0.0.1:8000",
)


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

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "last_search" not in st.session_state:
    st.session_state.last_search = None

if "feedback" not in st.session_state:
    st.session_state.feedback = {}


st.set_page_config(
    page_title="SignalRank-RAG",
    page_icon="◌",
    layout="centered",
)


st.title("SignalRank-RAG")

st.caption("Find the evidence that matters.")
st.caption("Demo corpus: 39 chunks across science, history, computing, and mythology.")
st.caption(
    "Searches and feedback may be logged for evaluation. "
    "Please don't enter sensitive information."
)

st.write("")


query = st.text_area(
    "What would you like to find?",
    placeholder=("Ask a question or describe the information you are looking for..."),
    height=130,
)

mode = st.segmented_control(
    "Retrieval mode",
    options=["Dense", "BM25", "Hybrid"],
    default="Dense",
)

top_k = st.slider(
    "Number of results",
    min_value=1,
    max_value=10,
    value=5,
)


search = st.button(
    "Search",
    type="primary",
    width="stretch",
)


if search:
    if not query.strip():
        st.warning("Enter a query first.")

    else:
        selected_mode = mode or "Dense"
        search_id = str(uuid.uuid4())

        with (
            logfire.span(
                "retrieval search",
                session_id=st.session_state.session_id,
                search_id=search_id,
                retrieval_mode=selected_mode.lower(),
                top_k=top_k,
                query_length=len(query),
            ),
            st.spinner("Searching the corpus..."),
        ):
            try:
                response = requests.post(
                    f"{API_URL}/retrieve",
                    json={
                        "query": query,
                        "mode": selected_mode.lower(),
                        "top_k": top_k,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

            except requests.RequestException as exc:
                logfire.error(
                    "Retrieval service request failed",
                    error=str(exc),
                    session_id=st.session_state.session_id,
                    search_id=search_id,
                )

                st.error("SignalRank-RAG could not reach the retrieval service.")

            else:
                st.session_state.last_search = {
                    "search_id": search_id,
                    "query": query,
                    "mode": selected_mode.lower(),
                    "top_k": top_k,
                    "results": data["results"],
                }


last_search = st.session_state.last_search

if last_search:
    results = last_search["results"]

    st.write("")
    st.subheader(f"{len(results)} results")

    for result in results:
        with st.container(border=True):
            header, score = st.columns([4, 1])

            with header:
                st.markdown(f"**Result {result['rank']}**")

            with score:
                st.markdown(f"{result['score']:.3f}")

            st.write(result["text"])
            st.caption(result["source_path"])

            feedback_key = f"{last_search['search_id']}:{result['chunk_id']}"
            recorded_feedback = st.session_state.feedback.get(feedback_key)

            relevant_col, not_relevant_col = st.columns(2)

            relevant = relevant_col.button(
                "👍 Relevant",
                key=f"relevant-{feedback_key}",
                width="stretch",
                disabled=recorded_feedback is not None,
            )

            not_relevant = not_relevant_col.button(
                "👎 Not relevant",
                key=f"not-relevant-{feedback_key}",
                width="stretch",
                disabled=recorded_feedback is not None,
            )

            if relevant or not_relevant:
                relevance = "relevant" if relevant else "not_relevant"

                st.session_state.feedback[feedback_key] = relevance

                logfire.info(
                    "retrieval result feedback",
                    session_id=st.session_state.session_id,
                    search_id=last_search["search_id"],
                    query=last_search["query"],
                    query_length=len(last_search["query"]),
                    retrieval_mode=last_search["mode"],
                    chunk_id=result["chunk_id"],
                    doc_id=result["doc_id"],
                    source_path=result["source_path"],
                    rank=result["rank"],
                    relevance=relevance,
                )

                st.rerun()

            if recorded_feedback:
                label = (
                    "Relevant" if recorded_feedback == "relevant" else "Not relevant"
                )
                st.caption(f"Feedback recorded: {label}")

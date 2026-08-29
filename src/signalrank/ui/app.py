import os
import uuid

import logfire
import requests
import streamlit as st

SERVICE_TOKEN = os.getenv("SIGNALRANK_SERVICE_TOKEN")

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

if "query_input" not in st.session_state:
    st.session_state.query_input = ""


st.set_page_config(
    page_title="SignalRank-RAG",
    page_icon="◌",
    layout="centered",
)


st.title("SignalRank-RAG")
st.caption("Find the evidence that matters.")
st.caption(
    "Demo corpus: 39 benchmark documents across science, space, computing, "
    "medicine, energy, history, and mythology."
)
st.caption(
    "Searches and feedback may be logged for evaluation. "
    "Please don't enter sensitive information."
)

with st.expander("What's in the demo corpus?"):
    st.markdown(
        """
        **Science & space:** aliens, climate observations, dinosaurs, lunar rovers,
        Mars, Mars orbiters, oceans, radio astronomy, solar energy, unexplained
        signals, volcanoes, weather, weather forecasting

        **Biology & medicine:** antibiotics, antiviral resistance, immune escape,
        vaccines

        **Energy:** batteries, battery management, grid storage

        **Computing & ML:** machine-learning training, neural networks,
        optimization, Python, retrieval, static typing, type systems

        **History & mythology:** pharaohs, Greek city-states, cults, festivals,
        gods, heroes, monsters, oracles, poetry, sacrifices, temples, underworld
        """
    )

st.write("")

st.markdown("**Try an example**")
in_corpus_col, out_of_domain_col = st.columns(2)

if in_corpus_col.button(
    "🦕 In corpus · Dinosaur extinction",
    width="stretch",
):
    st.session_state.query_input = "What caused the extinction of dinosaurs?"

if out_of_domain_col.button(
    "☕ Out of domain · Making coffee",
    width="stretch",
):
    st.session_state.query_input = "How do I make coffee?"

st.caption(
    "Dinosaurs are covered by the demo corpus; coffee is intentionally absent. "
    "Compare what the retrievers return when relevant evidence exists — "
    "and when it doesn't."
)

st.write("")

query = st.text_area(
    "What would you like to find?",
    placeholder="Ask a question or describe the information you are looking for...",
    height=130,
    key="query_input",
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
    st.session_state.last_search = None

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
                if not SERVICE_TOKEN:
                    raise RuntimeError("SIGNALRANK_SERVICE_TOKEN is not configured.")

                response = requests.post(
                    f"{API_URL}/chat",
                    headers={
                        "X-SignalRank-Service-Token": SERVICE_TOKEN,
                    },
                    json={
                        "query": query,
                        "mode": selected_mode.lower(),
                        "top_k": top_k,
                    },
                    timeout=(5, 120),
                )
                response.raise_for_status()
                data = response.json()

            except requests.Timeout as exc:
                logfire.error(
                    "Retrieval service timed out",
                    error=str(exc),
                    session_id=st.session_state.session_id,
                    search_id=search_id,
                )

                st.error(
                    "The retrieval service timed out. "
                    "The first request may take longer while the model starts."
                )

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
                    "route": data["route"],
                    "answer": data["answer"],
                    "results": data["results"],
                }


last_search = st.session_state.last_search

if last_search:
    results = last_search["results"]

    st.subheader("Answer")
    st.write(last_search["Answer"])

    if results:
        with st.expander(f"View retrieved evidence ({len(results)})"):
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
                            "Relevant"
                            if recorded_feedback == "relevant"
                            else "Not relevant"
                        )
                        st.caption(f"Feedback recorded: {label}")

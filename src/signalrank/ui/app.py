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


st.set_page_config(
    page_title="SignalRank-RAG",
    page_icon="◌",
    layout="centered",
)


st.title("SignalRank-RAG")

st.caption("Find the evidence that matters.")

st.write("")


query = st.text_area(
    "What would you like to find?",
    placeholder=("Ask a question or describe the information you are looking for..."),
    height=130,
)

mode = st.segmented_control(
    "Retrieval mode",
    options=["Dense", "BM25"],
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

        with (
            logfire.span(
                "retrieval search",
                session_id=st.session_state.session_id,
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
                )

                st.error("SignalRank-RAG could not reach the retrieval service.")

            else:
                results = data["results"]

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

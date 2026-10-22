"""
Streamlit UI for the query phase: ask questions of the ingested papers, of uploaded PDFs, or both.

    streamlit run src/app.py

Sidebar: iteration, chunking, retrieval, embedding model, top-k, reranker, generator, and
search scope; defaults come from data/best_config_{iteration}.json when it exists.
Uploaded PDFs are indexed in memory for the browser session (session_index.py); the
corpus indexes on disk are never modified.
"""

import html
import time

import streamlit as st

from config import (
    CHUNK_METHODS, CHUNK_SIZE, EMBEDDING_MODELS, GENERATION_CANDIDATES, GENERATION_MODEL, HYBRID_ALPHA, ITERATION,
    OVERLAP, RERANKER, RETRIEVAL_METHODS, SEMANTIC_CHUNK_EMBEDDING_MODEL, TOP_K, method_key, model_key,
)
from models import AnswerEntry, RetrievalConfig, RetrieverType
from query_phase import QuerySettings, answer_entry
from saving import best_config_path
from session_index import Scope, UploadedFile, build_session_index, corpus_retriever, scoped_retriever

RERANKERS = ["none", "cross-encoder", "cohere"]
TOP_K_MAX = 10


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading the corpus index…")
def cached_corpus_retriever(iteration: int, chunk_method: str, retrieval: str, model: str | None):
    s = QuerySettings(iteration=iteration)
    config = RetrievalConfig(chunk_method=chunk_method, embedding_model=model, retrieval_method=retrieval, top_k=TOP_K)
    return corpus_retriever(s, config)


@st.cache_resource
def cached_generator(model: str):
    from generation import LLMGenerator
    return LLMGenerator(model)


def best_defaults(iteration: int) -> dict:
    """Sidebar defaults: the iteration's best configuration, else .env."""
    import json

    path = best_config_path(iteration)
    if path.exists():
        best = json.loads(path.read_text())["config"]
        return {"chunk": best["chunk_method"].upper(), "retrieval": best["retrieval_method"],
                "model": best["embedding_model"] or EMBEDDING_MODELS[0], "reranker": best["reranker"]}
    return {"chunk": CHUNK_METHODS[0], "retrieval": RETRIEVAL_METHODS[0], "model": EMBEDDING_MODELS[0],
            "reranker": RERANKER}


def _index(options: list, value) -> int:
    return options.index(value) if value in options else 0


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar(has_uploads: bool) -> tuple[QuerySettings, RetrievalConfig, str, Scope]:
    st.sidebar.header("Configuration")
    iteration = int(st.sidebar.number_input("Iteration (corpus indexes)", min_value=1, value=ITERATION, step=1))
    d = best_defaults(iteration)
    chunk = st.sidebar.selectbox("Chunking", CHUNK_METHODS, index=_index(CHUNK_METHODS, d["chunk"]))
    retrieval = st.sidebar.selectbox("Retrieval", RETRIEVAL_METHODS, index=_index(RETRIEVAL_METHODS, d["retrieval"]))
    uses_embeddings = RetrieverType(retrieval) is not RetrieverType.BM25
    model = st.sidebar.selectbox("Embedding model", EMBEDDING_MODELS, index=_index(EMBEDDING_MODELS, d["model"]),
                                 disabled=not uses_embeddings, help="Not used by BM25")
    top_k = st.sidebar.slider("Top-k chunks", 1, TOP_K_MAX, TOP_K)
    reranker = st.sidebar.selectbox("Reranker", RERANKERS, index=_index(RERANKERS, d["reranker"]))
    generators = list(dict.fromkeys([GENERATION_MODEL, *GENERATION_CANDIDATES]))
    generator = st.sidebar.selectbox("Generator", generators)
    scopes = list(Scope)
    scope = Scope(st.sidebar.radio("Search", [s.value for s in scopes],
                                   index=scopes.index(Scope.BOTH if has_uploads else Scope.CORPUS)))
    if best_config_path(iteration).exists():
        st.sidebar.caption(f"Defaults: iteration {iteration}'s best configuration")
    config = RetrievalConfig(chunk_method=method_key(chunk), embedding_model=model if uses_embeddings else None,
                             retrieval_method=retrieval, top_k=top_k, reranker=reranker,
                             hybrid_alpha=HYBRID_ALPHA if retrieval == "hybrid" else None)
    return QuerySettings(iteration=iteration, reranker=reranker, top_k=top_k), config, generator, scope


# ── Uploads ───────────────────────────────────────────────────────────────────

def session_index_for(files, chunk: str):
    """The session's index of `files` for this chunking method, rebuilt only when either changes."""
    if not files:
        return None
    uploads = [UploadedFile(f.name, f.getvalue()) for f in files]
    key = (tuple(u.digest for u in uploads), chunk)
    if st.session_state.get("upload_key") != key:
        with st.spinner(f"Indexing {len(uploads)} PDF(s) with {chunk.lower()} chunking…"):
            st.session_state["upload_index"] = build_session_index(uploads, chunk.upper(), CHUNK_SIZE, OVERLAP,
                                                                    SEMANTIC_CHUNK_EMBEDDING_MODEL)
        st.session_state["upload_key"] = key
    return st.session_state["upload_index"]


# ── Display ───────────────────────────────────────────────────────────────────

def highlighted(content: str, snippet: str | None) -> str:
    """Chunk text as HTML with the cited snippet marked; "$" escaped so Streamlit doesn't render math."""
    text = html.escape(content)
    if snippet:
        quoted = html.escape(snippet)
        text = text.replace(quoted, f"<mark>{quoted}</mark>", 1)
    return f'<div style="white-space: pre-wrap; font-size: 0.9em">{text.replace("$", "&#36;")}</div>'


def show_answer(entry: AnswerEntry, retrieval_seconds: float) -> None:
    if entry.error:
        st.error(f"Generation failed: {entry.error}")
    else:
        r = entry.response
        st.subheader("Answer")
        st.markdown(r.answer)
        st.caption(f"Confidence {r.confidence:.2f} · retrieval {retrieval_seconds * 1000:.0f} ms · "
                   f"generation {entry.generation_seconds:.1f} s")
        if r.citations:
            st.subheader("Citations")
            rank_of = {ref.chunk_id: ref.rank for ref in entry.retrieved}
            for c in r.citations:
                st.markdown(f"- **[chunk {rank_of.get(c.chunk_id, '?')}] {c.source}, page {c.page_number}** "
                            f"(score {c.relevance_score:.3f}) — “{html.escape(c.text_snippet).replace('$', '&#36;')}”")

    st.subheader("Source chunks")
    chunks = {c.id: c for c in (entry.response.chunks_used if entry.response else [])}
    snippets = {c.chunk_id: c.text_snippet for c in entry.response.citations} if entry.response else {}
    for ref in entry.retrieved:
        cited = ref.chunk_id in snippets
        section = f" · {ref.section_title}" if ref.section_title else ""
        label = f"{'✅ ' if cited else ''}#{ref.rank} {ref.source} · page {ref.page_number}{section} · score {ref.score:.3f}"
        with st.expander(label, expanded=cited):
            chunk = chunks.get(ref.chunk_id)
            if chunk:
                st.markdown(highlighted(chunk.content, snippets.get(ref.chunk_id)), unsafe_allow_html=True)


# ── Page ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Research Paper QA", page_icon="📄", layout="wide")
    st.title("Research Paper QA")
    files = st.file_uploader("Upload PDFs (searched alongside or instead of the ingested papers)", type=["pdf"],
                             accept_multiple_files=True)
    s, config, generator, scope = sidebar(bool(files))

    session = session_index_for(files, config.chunk_method.upper())
    if session:
        st.caption(f"Uploaded: {len(session.documents)} PDF(s), {len(session.chunks)} {session.chunk_method} chunks")
        for name, error in session.errors.items():
            st.warning(f"{name}: {error}")

    with st.form("ask"):
        question = st.text_input("Question", placeholder="e.g. What is the concept of unambiguous efficiency?")
        asked = st.form_submit_button("Ask", type="primary")
    if not (asked and question.strip()):
        return

    try:
        corpus = None
        if scope in (Scope.CORPUS, Scope.BOTH):
            corpus = cached_corpus_retriever(s.iteration, config.chunk_method, config.retrieval_method.value,
                                             config.embedding_model)
        retriever = scoped_retriever(scope, config, s, session, corpus)
        with st.spinner("Retrieving…"):
            start = time.monotonic()
            results = retriever.retrieve(question, config.top_k)
            retrieval_seconds = time.monotonic() - start
        with st.spinner(f"Generating with {model_key(generator)}…"):
            entry = answer_entry(question, results, retrieval_seconds, cached_generator(generator))
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        return
    st.caption(f"Configuration: **{config.name}** · search: {scope.value} · generator: {generator}")
    show_answer(entry, retrieval_seconds)


main()

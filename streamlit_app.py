from pathlib import Path

import streamlit as st
from werkzeug.utils import secure_filename

from app import (
    HRKnowledgeBase,
    UPLOAD_DIR,
    allowed_pdf,
    answer_locally,
    normalize_text,
)


st.set_page_config(
    page_title="HR Policy Q&A Bot",
    page_icon=":briefcase:",
    layout="wide",
)


def get_knowledge_base() -> HRKnowledgeBase:
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = HRKnowledgeBase()
    return st.session_state.knowledge_base


def get_messages() -> list[dict]:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Upload HR policy PDFs, then ask questions about leave, benefits, conduct, reimbursements, or working hours.",
                "sources": [],
            }
        ]
    return st.session_state.messages


def save_uploaded_pdf(uploaded_file) -> Path:
    filename = secure_filename(uploaded_file.name)
    pdf_path = UPLOAD_DIR / filename
    pdf_path.write_bytes(uploaded_file.getbuffer())
    return pdf_path


knowledge_base = get_knowledge_base()
messages = get_messages()

st.title("HR Policy / Document Q&A Bot")

with st.sidebar:
    st.header("Upload policy PDFs")
    uploaded_files = st.file_uploader(
        "Choose one or more HR PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Index PDFs", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please choose at least one PDF.")
        else:
            indexed = []
            for uploaded_file in uploaded_files:
                if not allowed_pdf(uploaded_file.name):
                    st.error(f"{uploaded_file.name} is not a PDF.")
                    st.stop()

                pdf_path = save_uploaded_pdf(uploaded_file)
                chunks_added = knowledge_base.add_pdf(pdf_path)
                indexed.append(f"{pdf_path.name} ({chunks_added} chunks)")

            messages.append(
                {
                    "role": "assistant",
                    "content": "Indexed " + ", ".join(indexed) + ".",
                    "sources": [],
                }
            )
            st.success("PDFs indexed successfully.")

    if st.button("Clear in-memory index", use_container_width=True):
        knowledge_base.clear()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "The index has been cleared. Upload PDFs to start again.",
                "sources": [],
            }
        ]
        st.rerun()

    st.divider()
    st.metric("Chunks indexed", len(knowledge_base.chunks))

    documents = sorted({chunk.filename for chunk in knowledge_base.chunks})
    if documents:
        st.caption("Indexed documents")
        for document in documents:
            st.write(f"- {document}")
    else:
        st.caption("No documents indexed yet.")

st.subheader("Ask a question")

for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for index, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**{index}. {source['filename']}, page {source['page']}** "
                        f"(score: {source['score']})"
                    )
                    st.write(source["text"])

question = st.chat_input("Ask: what is the leave policy?")
if question:
    cleaned_question = normalize_text(question)
    messages.append({"role": "user", "content": cleaned_question, "sources": []})

    if not knowledge_base.ready:
        answer = "Upload and index at least one HR policy PDF first."
        sources = []
    else:
        sources = knowledge_base.search(cleaned_question)
        answer = answer_locally(cleaned_question, sources)

    messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()

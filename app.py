import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from streamlit.components.v1 import html
import os
import markdown2

# Page config
st.set_page_config(page_title="RAG Playground", layout="centered")

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Title and description
st.markdown("""
<h1 style='color: #1f77b4; text-align: center;'>📚 DSAI RAG PLAYGROUND</h1>
<p style='text-align: center; font-size: 14px; color: #1f77b4; margin-top: -10px; font-style: italic;'>Chat with Your Documents</p>
""", unsafe_allow_html=True)

# Sidebar: Upload PDFs
with st.sidebar:
    st.markdown("<h3 style='text-align: left;'>📄 Upload PDFs</h3>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        all_text = ""
        for uploaded_file in uploaded_files:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                all_text += page.extract_text() or ""

        st.success("✅ PDFs processed!")

        # Text splitting
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(all_text)

        # Embeddings + VectorStore
        with st.spinner("🔍 Indexing..."):
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(texts, embeddings)
            retriever = vectorstore.as_retriever()

            # RAG QA Chain
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            st.session_state.qa_chain = qa_chain

        st.success("📚 Playground is ready! Start chatting in the main area.")

# Main Chat Interface
if st.session_state.qa_chain:
    for speaker, message in st.session_state.chat_history:
        if speaker == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(
                    f"<div style='background-color:#2f2f2f; color:white; padding:6px 10px; border-radius:8px; white-space:pre-wrap; margin-bottom:2px;'>{message}</div>",
                    unsafe_allow_html=True
                )
        else:
            with st.chat_message("assistant", avatar="💬"):
                html_message = markdown2.markdown(message)
                st.markdown(
                    f"<div style='background-color:#2f2f2f; color:white; padding:6px 10px; border-radius:8px; margin-bottom:2px; line-height:1.4;'>{html_message}</div>",
                    unsafe_allow_html=True
                )

    # Chat input
    prompt = st.chat_input("Ask a question about your documents...")

    if prompt:
        with st.chat_message("user", avatar="👤"):
            st.markdown(
                f"<div style='background-color:#2f2f2f; color:white; padding:6px 10px; border-radius:8px; white-space:pre-wrap; margin-bottom:2px;'>{prompt}</div>",
                unsafe_allow_html=True
            )

        with st.spinner("Generating response..."):
            answer = st.session_state.qa_chain.run(prompt)

        with st.chat_message("assistant", avatar="💬"):
            html_answer = markdown2.markdown(answer)
            st.markdown(
                f"<div style='background-color:#2f2f2f; color:white; padding:6px 10px; border-radius:8px; margin-bottom:2px; line-height:1.4;'>{html_answer}</div>",
                unsafe_allow_html=True
            )

        st.session_state.chat_history.append(("user", prompt))
        st.session_state.chat_history.append(("assistant", answer))

    # ✅ Show download button if there is chat history
    if st.session_state.chat_history:
        chat_text = ""
        for speaker, message in st.session_state.chat_history:
            chat_text += f"{speaker.capitalize()}: {message}\n\n"

        st.download_button(
            label="📥 Download Chat History",
            data=chat_text,
            file_name="rag_chat_history.txt",
            mime="text/plain",
            help="Download the full chat history as a text file"
        )

else:
    # Upload message when no PDF has been uploaded
    st.markdown("""
    <h4 style='text-align:center; color:gray;'>👈 Upload PDFs in the sidebar to enable chat</h4>
    """, unsafe_allow_html=True)


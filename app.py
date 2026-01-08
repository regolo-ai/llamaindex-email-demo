import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.imap import ImapReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
import traceback
import json


load_dotenv()

REGOLO_AI_API_KEY = os.getenv("REGOLO_AI_API_KEY")
OPENAI_HOST = os.getenv("OPENAI_HOST")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "gpt-3.5-turbo")
USER_EMAIL = os.getenv("USER_EMAIL")
USER_PASSWORD = os.getenv("USER_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
INDEX_STORAGE_DIR = "index_storage"

st.title("LlamaIndex Email RAG with Configurable OpenAI Model")

if not REGOLO_AI_API_KEY or not USER_EMAIL or not USER_PASSWORD:
    st.error("Missing environment variables. Please update your `.env` file with all required fields.")

st.sidebar.write(f"Using Model: {OPENAI_MODEL}")
st.sidebar.write(f"Using Embedding Model: {OPENAI_EMBEDDING_MODEL}")
st.sidebar.write(f"Using OpenAI Provider: {OPENAI_HOST}")
st.sidebar.write(f"Connecting to: {USER_EMAIL}")

index_file_path = os.path.join(INDEX_STORAGE_DIR, "docstore.json")

llm = OpenAILike(
    model=OPENAI_MODEL,
    api_key=REGOLO_AI_API_KEY,
    api_base=OPENAI_HOST,
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=8192,
)
embeddings = OpenAILikeEmbedding(
    model_name=OPENAI_EMBEDDING_MODEL,
    api_key=REGOLO_AI_API_KEY,
    api_base=OPENAI_HOST,
)
Settings.llm = llm
Settings.embed_model = embeddings

def fetch_emails_with_imap():
    reader = ImapReader(
        username=USER_EMAIL,
        password=USER_PASSWORD,
        host=IMAP_SERVER
    )
    return list(reader.load_data(search_criteria="ALL"))

if st.button("Index Emails"):
    try:
        emails = fetch_emails_with_imap()
        st.success(f"Fetched {len(emails)} emails")

        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        nodes = node_parser.get_nodes_from_documents(emails)
        for node in nodes:
            try:
                embedding = embeddings._get_text_embedding(node.get_text())
                setattr(node, "embedding", embedding)
            except Exception as e:
                st.error(f"Error on embedding: {e}")

        index = VectorStoreIndex.from_documents(
            nodes,
            node_parser=node_parser
        )
        index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
        st.sidebar.write(f"Emails avalaible: {len(index.docstore.docs)}")
    except Exception as e:
        st.error(f"Error occurred while indexing emails: {e}")
        st.error(traceback.format_exc())

if os.path.exists(index_file_path):
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
    docstore = storage_context.docstore

    try:
        all_document_ids = docstore.get_document_ids()
        nodes = [docstore.get_document(doc_id) for doc_id in all_document_ids]
    except AttributeError:
        nodes = []

    index = VectorStoreIndex.from_documents(
        nodes,
        storage_context=storage_context,
    )
else:
    index = VectorStoreIndex.from_documents(
            [],
        )
    index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)

st.sidebar.write(f"Emails avalaible: {len(index.docstore.docs)}")

with st.form("query_form"):
    query = st.text_input("Search your emails or ask a question")
    submit_query = st.form_submit_button("Run Query")

    if submit_query:
        if not index_file_path:
            st.warning("No index found. Please index your emails first.")
        else:
            try:
                response = index.as_query_engine(similarity_top_k=3).query(query)
                st.write("### Query Response")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred while querying: {e}")

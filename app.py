import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.readers.imap import ImapReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.core.vector_stores import VectorStoreQuery
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
INDEX_FILE = os.path.join(INDEX_STORAGE_DIR, "vector_store.json")

st.title("LlamaIndex Email RAG with Configurable OpenAI Model")

if not REGOLO_AI_API_KEY or not USER_EMAIL or not USER_PASSWORD:
    st.error("Missing environment variables. Please update your `.env` file with all required fields.")

st.sidebar.markdown(f"""
**Model:** {OPENAI_MODEL}

**Embedding Model:** {OPENAI_EMBEDDING_MODEL}

**OpenAI Provider:** {OPENAI_HOST}

**User Email:** {USER_EMAIL}
""")

llm = OpenAILike(
    model=OPENAI_MODEL,
    api_key=REGOLO_AI_API_KEY,
    api_base=OPENAI_HOST,
    is_chat_model=False,
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


def save_vector_store(file_path, nodes):
    with open(file_path, "w") as f:
        json.dump(
            [{"text": node['text'], "embedding": node['embedding'], "id": node['id']} for node in nodes],
            f,
        )
    st.success(f"Vector store saved to {file_path}")


def load_vector_store(file_path):
    vector_store = SimpleVectorStore()
    id_to_text = {}

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        nodes = []
        for item in data:
            node = TextNode(
                text=item["text"],
                id_=item["id"]
            )
            node.embedding = item["embedding"]
            nodes.append(node)
            id_to_text[item["id"]] = item["text"]

        vector_store.add(nodes)

        st.sidebar.write(f"Email indexed: {len(data)}")

    return vector_store, id_to_text


if st.button("Index Emails"):
    try:
        emails = fetch_emails_with_imap()
        st.success(f"Fetched {len(emails)} emails")

        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        nodes = node_parser.get_nodes_from_documents(emails)
        indexed_nodes = []
        to_save = []
        for node in nodes:
            text = node.get_text()
            embedding = embeddings._get_text_embedding(text)
            text_node = TextNode(text=text)
            to_save.append({"text": text, "embedding": embedding, "id": node.node_id})
            text_node.embedding = embedding
            indexed_nodes.append(text_node)
        index = SimpleVectorStore()
        index.add(nodes=indexed_nodes)

        os.makedirs(INDEX_STORAGE_DIR, exist_ok=True)
        save_vector_store(INDEX_FILE, to_save)
    except Exception as e:
        st.error(f"Error occurred while indexing emails: {e}")
        st.error(traceback.format_exc())

index, id_to_text = load_vector_store(INDEX_FILE)

with st.form("query_form"):
    query = st.text_input("Search your emails or ask a question")
    submit_query = st.form_submit_button("Run Query")

    if submit_query:
        if not os.path.exists(INDEX_FILE):
            st.warning("No index found. Please index your emails first (or restart).")
        else:
            try:
                query_embedding = embeddings._get_text_embedding(query)
                vs_query = VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=3,
                )
                response = index.query(vs_query, similarity_top_k=3)
                texts = [id_to_text[_id] for _id in response.ids]
                context = "\n\n".join(
                    f"[Email {i + 1}]\n{text}"
                    for i, text in enumerate(texts)
                )

                st.write(f"### Email found: {len(texts)}")

                st.write("### Query Response (takes a while, no streaming)")

                prompt = f"""
                You are an assistant who responds using EXCLUSIVELY the provided emails.

                EMAIL:
                {context}

                QUESTION:
                {query}

                INSTRUCTIONS:

                Use only the information contained in the emails.
                If the emails do not contain the answer, state it clearly.
                Respond in a detailed and structured manner.
                """
                print(prompt)
                response = llm.complete(prompt)
                st.write(response.text)
            except Exception as e:
                st.error(f"An error occurred while querying: {e}")
                st.error(traceback.format_exc())

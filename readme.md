# LlamaIndex Email RAG Demo

This is a working demo using [Streamlit](https://streamlit.io/) and [LlamaIndex](https://www.llamaindex.ai/) to build a simple email-based Retrieval-Augmented Generation (RAG) system with an Open-AI compatible provider like Regolo.AI.

## Features
- Index and query data directly from your email inbox.
- Supports configurable OpenAI-compatible models for indexing and querying.
- Uses persistent storage so you don't need to re-index your email data every time.

## Prerequisites
- Python 3.8 or above
- An OpenAI-compatible API key, like Regolo.AI
- An IMAP-compatible email account

## Setup
1. Clone this repository and navigate to the project directory.
2. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - On **Linux/MacOS**:
     ```bash
     source .venv/bin/activate
     ```
   - On **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Create a `.env` file in the root directory

## How to Run
1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Use the app:
   - Click the "Index Emails" button to fetch and index your emails.
   - Enter a query to search your indexed emails or ask a question.

3. The app will use your configured OpenAI-compatible model to perform RAG.

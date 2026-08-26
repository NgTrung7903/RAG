# ⚖️ LegalGPT - AI Legal Consultant

LegalGPT is an intelligent Retrieval-Augmented Generation (RAG) web application designed to act as an AI legal consultant. It ingests massive legal PDF documents, processes them into a searchable vector database, and allows users to ask complex legal questions using Google's Gemini AI.

## 🚀 Features

- **Document Ingestion:** Processes large PDF files and splits them into context-aware chunks.
- **Smart Retrieval:** Uses FAISS (Facebook AI Similarity Search) for fast, local vector searches.
- **Google Gemini Powered:** Utilizes `gemini-3.6-flash` for high-quality, legally-grounded answers.
- **Hallucination Guardrails:** Strict system prompts prevent the AI from fabricating laws or penalties not present in the uploaded document.
- **Rate-Limit Resilient:** Built-in sleep and retry mechanisms to gracefully handle API limits when processing massive documents.
- **Beautiful Web UI:** Clean and interactive chat interface built with Streamlit.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM & Embeddings:** Google Generative AI (Gemini)
- **Framework:** LangChain
- **Vector Database:** FAISS
- **PDF Processing:** PyPDF

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NgTrung7903/RAG.git
   cd RAG
   ```

2. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your document**
   Place the legal PDF you want to query inside the root directory and name it `document.pdf`.

## 🎮 How to Use

1. **Run the application:**
   ```bash
   python -m streamlit run app.py
   ```
2. **Open your browser** to `http://localhost:8501`.
3. **Configure your API Key:** In the sidebar, paste your Google API Key (you can get one from [Google AI Studio](https://aistudio.google.com/)).
4. **Initialize Database:** Click the button to build the vector index. (This may take a few minutes for large PDFs).
5. **Start Chatting!** Ask legal questions and receive answers cited directly from your document.

## ⚠️ Disclaimer
This tool is for educational and experimental purposes. It relies entirely on the provided documents and AI interpretations. Always consult a certified human lawyer for real legal advice.

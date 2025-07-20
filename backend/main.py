from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import faiss, pickle, numpy as np
from docx import Document
import os, json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from notion_client import Client
from googleapiclient.discovery import build
from google.oauth2 import service_account
from atlassian import Confluence
import uvicorn

# ----------------------------
# Environment Setup
# ----------------------------
load_dotenv()

# Models will be loaded lazily to save memory
embedding_model = None
text_generator = None

HF_MODEL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-large"

# Notion Setup
NOTION_TOKEN = os.getenv("NOTION_API_KEY")
notion = Client(auth=NOTION_TOKEN) if NOTION_TOKEN else None

# Google Docs Setup (Render Safe)
GOOGLE_SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
GOOGLE_JSON_CONTENT = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
if GOOGLE_JSON_CONTENT:
    creds_info = json.loads(GOOGLE_JSON_CONTENT)
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=GOOGLE_SCOPES)
    docs_service = build('docs', 'v1', credentials=creds)
else:
    docs_service = None

# Confluence Setup
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL", "")
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL", "")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", "")
confluence = None
if CONFLUENCE_BASE_URL and CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN:
    confluence = Confluence(
        url=CONFLUENCE_BASE_URL,
        username=CONFLUENCE_EMAIL,
        password=CONFLUENCE_API_TOKEN
    )

# ----------------------------
# FAISS Index Setup
# ----------------------------
embedding_dim = 384
INDEX_PATH = "embeddings_store/index.faiss"
DOCS_PATH = "embeddings_store/docs.pkl"

def load_index():
    global index, documents
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
        print(f"Loaded FAISS index with {len(documents)} documents.")
    else:
        index = faiss.IndexFlatL2(embedding_dim)
        documents = []
        print("Initialized empty FAISS index.")

def save_index():
    os.makedirs("embeddings_store", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    print(f"Saved FAISS index with {len(documents)} documents.")

load_index()

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Internal Docs AI Agent")

class Question(BaseModel):
    query: str

# ----------------------------
# Lazy Loading
# ----------------------------
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(HF_MODEL_EMBED)
    return embedding_model

def get_text_generator():
    global text_generator
    if text_generator is None:
        from transformers import pipeline
        text_generator = pipeline("text2text-generation", model=GEN_MODEL)
    return text_generator

# ----------------------------
# Utility Functions
# ----------------------------
def search_docs(query, top_k=2):
    if not documents or index.ntotal == 0:
        return []
    query_emb = get_embedding_model().encode([query])[0]
    D, I = index.search(np.array([query_emb], dtype="float32"), top_k)
    valid_indices = [i for i in I[0] if i < len(documents)]
    return [documents[i] for i in valid_indices]

def generate_agent_answer(query, docs):
    if not docs:
        return "I couldn’t find any relevant information in the company docs yet. Upload some files first."
    context = "\n".join([f"{title}: {content}" for title, content in docs])
    prompt = f"""
You are a helpful AI assistant.
Use the following context to answer the question.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}
Answer:
"""
    generator = get_text_generator()
    result = generator(prompt, max_length=250, do_sample=False)[0]['generated_text']
    return result.strip()

def fetch_notion_page_text(page_id: str):
    if not notion:
        return ""
    blocks = notion.blocks.children.list(page_id)["results"]
    content = []
    for block in blocks:
        if "paragraph" in block:
            text = "".join([t["text"]["content"] for t in block["paragraph"]["rich_text"]])
            content.append(text)
        elif "heading_1" in block:
            content.append(block["heading_1"]["rich_text"][0]["text"]["content"])
        elif "heading_2" in block:
            content.append(block["heading_2"]["rich_text"][0]["text"]["content"])
    return "\n".join(content)

def fetch_google_doc_text(doc_id: str):
    if not docs_service:
        return ""
    doc = docs_service.documents().get(documentId=doc_id).execute()
    content = []
    for element in doc.get('body', {}).get('content', []):
        if 'paragraph' in element:
            for text_run in element['paragraph'].get('elements', []):
                text = text_run.get('textRun', {}).get('content', '').strip()
                if text:
                    content.append(text)
    return "\n".join(content)

def fetch_confluence_page_text(page_id: str):
    if not confluence:
        return ""
    try:
        page = confluence.get_page_by_id(page_id, expand="body.storage")
        html_content = page["body"]["storage"]["value"]
        from bs4 import BeautifulSoup
        return BeautifulSoup(html_content, "html.parser").get_text(separator="\n")
    except Exception:
        return ""

# ----------------------------
# API Endpoints
# ----------------------------
@app.post("/ask-question")
def ask_question(question: Question):
    query = question.query.strip().lower()
    casual_responses = {
        "hi": "Hey there! 👋",
        "hello": "Hello! 😊",
        "how are you": "I’m good! Got questions?",
        "who are you": "I’m UR BRO 😎, your AI docs assistant!",
    }
    for key, response in casual_responses.items():
        if key in query:
            return {"answer": response}

    docs = search_docs(question.query)
    if not docs:
        return {"answer": "I couldn’t find anything about that in the docs. Upload some files or connect Notion/Google Docs/Confluence."}
    answer = generate_agent_answer(question.query, docs)
    sources = [title for title, _ in docs]
    return {"answer": f"{answer}\n\n(Sources: {', '.join(sources)})", "sources": sources}

@app.get("/list-docs")
def list_docs():
    titles = [title for title, _ in documents]
    return {"total": len(titles), "documents": titles}

# ----------------------------
# Run Server (Render)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT dynamically
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)

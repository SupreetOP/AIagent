from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import faiss, pickle, numpy as np
from docx import Document
import os, json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader
from notion_client import Client
from googleapiclient.discovery import build
from google.oauth2 import service_account
from atlassian import Confluence
from bs4 import BeautifulSoup
import requests

# ----------------------------
# Load Environment + Models
# ----------------------------
load_dotenv()

HF_MODEL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(HF_MODEL_EMBED)

GEN_MODEL = "google/flan-t5-large"
text_generator = pipeline("text2text-generation", model=GEN_MODEL)

# Gemini API Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-pro"  # default model

def call_gemini(prompt):
    if not GEMINI_API_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

# Notion Setup
NOTION_TOKEN = os.getenv("NOTION_API_KEY")
notion = Client(auth=NOTION_TOKEN)

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
    model: str = "gemini"  # or "flan"

# ----------------------------
# Utility Functions
# ----------------------------
def search_docs(query, top_k=2):
    if not documents or index.ntotal == 0:
        return []
    query_emb = embedding_model.encode([query])[0]
    D, I = index.search(np.array([query_emb], dtype="float32"), top_k)
    valid_indices = [i for i in I[0] if i < len(documents)]
    return [documents[i] for i in valid_indices]

def generate_agent_answer(query, docs, model="gemini"):
    if not docs:
        return "I couldn’t find any relevant information in the company docs yet. Upload some files first."

    context = "\n".join([f"{title}: {content}" for title, content in docs])
    prompt = f"""
Use the following context to answer the question.
If the answer isn't in the context, say you don't know politely.

Context:
{context}

Question: {query}
Answer in a refined, conversational tone:
"""
    if model == "gemini" and GEMINI_API_KEY:
        gemini_resp = call_gemini(prompt)
        if gemini_resp:
            return gemini_resp
    # Fallback to Flan-T5
    result = text_generator(prompt, max_length=300, do_sample=False)[0]['generated_text']
    return result.strip()

def fetch_notion_page_text(page_id: str):
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
        return BeautifulSoup(html_content, "html.parser").get_text(separator="\n")
    except Exception as e:
        print(f"Failed to fetch Confluence page: {e}")
        return ""

# ----------------------------
# API Endpoints
# ----------------------------
@app.post("/ask-question")
def ask_question(question: Question):
    query = question.query.strip().lower()

    casual_responses = {
        "hi": "Hey there! 👋 How can I help you?",
        "hello": "Hello! 😊 What’s on your mind today?",
        "how are you": "I’m doing great! Got any questions for me?",
        "who are you": "I’m UR BRO 😎 — your AI assistant! I can read PDFs, DOCX, Notion, Google Docs, and Confluence for you.",
        "tell me a joke": "Sure! Why did the computer get cold? Because it forgot to close its Windows. 😄"
    }
    for key, response in casual_responses.items():
        if key in query:
            return {"answer": response}

    docs = search_docs(question.query)
    if not docs:
        return {"answer": "I couldn’t find anything about that. 🤔 Upload files or fetch from Notion, Google Docs, or Confluence."}

    answer = generate_agent_answer(question.query, docs, model=question.model)
    sources = [title for title, _ in docs]
    return {"answer": f"{answer}\n\n(Sourced from: {', '.join(sources)})", "sources": sources}

@app.post("/fetch-notion-doc")
async def fetch_notion_doc(page_id: str = Form(...), title: str = Form("Notion Document")):
    global documents, index
    try:
        content = fetch_notion_page_text(page_id)
    except Exception as e:
        return {"error": f"Failed to fetch Notion page: {str(e)}"}
    if not content.strip():
        return {"error": "Page is empty or cannot be read."}
    emb = embedding_model.encode([content])[0]
    index.add(np.array([emb], dtype="float32"))
    documents.append((title, content))
    save_index()
    return {"message": f"Notion page '{title}' indexed successfully.", "total_docs": len(documents)}

@app.post("/fetch-google-doc")
async def fetch_google_doc(document_id: str = Form(...), title: str = Form("Google Document")):
    global documents, index
    try:
        content = fetch_google_doc_text(document_id)
    except Exception as e:
        return {"error": f"Failed to fetch Google Doc: {str(e)}"}
    if not content.strip():
        return {"error": "Google Doc is empty or cannot be read."}
    emb = embedding_model.encode([content])[0]
    index.add(np.array([emb], dtype="float32"))
    documents.append((title, content))
    save_index()
    return {"message": f"Google Doc '{title}' indexed successfully.", "total_docs": len(documents)}

@app.post("/fetch-confluence-page")
async def fetch_confluence_page(page_id: str = Form(...), title: str = Form("Confluence Page")):
    global documents, index
    try:
        content = fetch_confluence_page_text(page_id)
    except Exception as e:
        return {"error": f"Failed to fetch Confluence page: {str(e)}"}
    if not content.strip():
        return {"error": "Confluence page is empty or cannot be read."}
    emb = embedding_model.encode([content])[0]
    index.add(np.array([emb], dtype="float32"))
    documents.append((title, content))
    save_index()
    return {"message": f"Confluence page '{title}' indexed successfully.", "total_docs": len(documents)}

@app.get("/list-docs")
def list_docs():
    if not documents:
        return {"documents": [], "message": "No documents indexed yet."}
    titles = [title for title, _ in documents]
    return {"total": len(titles), "documents": titles}

@app.delete("/delete-doc")
def delete_doc(title: str):
    global documents, index
    to_delete = [i for i, (t, _) in enumerate(documents) if t.lower() == title.lower()]
    if not to_delete:
        return {"error": f"No document found with title '{title}'."}
    for i in sorted(to_delete, reverse=True):
        documents.pop(i)
    index = faiss.IndexFlatL2(embedding_dim)
    if documents:
        embeddings = [embedding_model.encode([content])[0] for _, content in documents]
        index.add(np.array(embeddings, dtype="float32"))
    save_index()
    return {"message": f"Deleted '{title}'.", "remaining_docs": len(documents)}

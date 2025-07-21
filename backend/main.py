from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss, pickle, numpy as np
from docx import Document
import os, json, shutil, requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from notion_client import Client
from googleapiclient.discovery import build
from google.oauth2 import service_account
from atlassian import Confluence
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load Environment + Models
# ----------------------------
load_dotenv()

HF_MODEL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(HF_MODEL_EMBED)

NOTION_TOKEN = os.getenv("NOTION_API_KEY")
notion = Client(auth=NOTION_TOKEN)

GOOGLE_SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
GOOGLE_JSON_CONTENT = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
docs_service = None
if GOOGLE_JSON_CONTENT:
    creds_info = json.loads(GOOGLE_JSON_CONTENT)
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=GOOGLE_SCOPES)
    docs_service = build('docs', 'v1', credentials=creds)

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

UPLOADS_DIR = "uploaded_files"
os.makedirs(UPLOADS_DIR, exist_ok=True)

def load_index():
    global index, documents
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
        print(f"Loaded FAISS index with {len(documents)} chunks.")
    else:
        index = faiss.IndexFlatL2(embedding_dim)
        documents = []
        print("Initialized empty FAISS index.")

def save_index():
    os.makedirs("embeddings_store", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    print(f"Saved FAISS index with {len(documents)} chunks.")

load_index()

# ----------------------------
# Text Chunking
# ----------------------------
CHUNK_SIZE = 500

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def add_to_index(name, text):
    chunks = chunk_text(text)
    for idx, chunk in enumerate(chunks):
        emb = embedding_model.encode([chunk])[0]
        index.add(np.array([emb], dtype="float32"))
        documents.append((f"{name} (part {idx+1})", chunk))
    save_index()

# ----------------------------
# Search Logic
# ----------------------------
def search_docs(query, top_k=3, min_similarity=0.35):
    if not documents or index.ntotal == 0:
        return []

    query_emb = embedding_model.encode([query])[0].reshape(1, -1)
    all_embs = np.vstack([embedding_model.encode([content])[0] for _, content in documents])
    sims = cosine_similarity(query_emb, all_embs)[0]

    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked[:top_k]:
        if score >= min_similarity:
            results.append((documents[idx][0], documents[idx][1], score))
    return results

# ----------------------------
# FastAPI App & Endpoints
# ----------------------------
app = FastAPI(title="Docs Q&A Agent")

origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str

class LinkInput(BaseModel):
    link: str

@app.post("/ask-question")
def ask_question(question: Question):
    query = question.query.strip()

    # Quick greetings
    casual = {
        "hi": "Hey there! 👋",
        "hello": "Hello! 😊",
        "how are you?": "I’m doing great! Ask me something from your files?",
        "who are you?": "I’m UR BRO 😎 — your AI assistant!"
    }
    if query.lower() in casual:
        return {"answer": casual[query.lower()]}

    # Search local indexed documents
    matches = search_docs(query, top_k=3)
    if not matches:
        return {"answer": "No relevant content found in your uploaded documents. Try uploading more files."}

    # Build answer manually (no external AI)
    response_parts = [f"**From {title} (relevance {score:.2f}):**\n{content[:500]}..." for title, content, score in matches]
    combined_answer = "\n\n".join(response_parts)

    return {"answer": f"Here’s what I found:\n\n{combined_answer}", "sources": [title for title, _, _ in matches]}

@app.post("/refresh-docs")
async def refresh_docs(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text_content = ""
    try:
        if file.filename.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text_content += page.extract_text() or ""
        elif file.filename.lower().endswith(".docx"):
            doc = Document(file_path)
            text_content = "\n".join([p.text for p in doc.paragraphs])
        else:
            os.remove(file_path)
            return {"error": "Unsupported file type. Only PDF or DOCX allowed."}
    except Exception as e:
        os.remove(file_path)
        return {"error": f"Failed to read file: {str(e)}"}

    if not text_content.strip():
        os.remove(file_path)
        return {"error": f"'{file.filename}' has no readable text. Upload another document."}

    add_to_index(file.filename, text_content)
    return {"message": f"File '{file.filename}' processed and indexed."}

@app.post("/process-link")
def process_link(link_data: LinkInput):
    link = link_data.link.strip()
    if not link:
        return {"error": "No link provided."}

    text_content, title = "", ""
    try:
        if "notion.so" in link:
            resp = requests.get(link)
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string if soup.title else "Notion Page"
            text_content = soup.get_text()
        elif "docs.google.com" in link and docs_service:
            doc_id = link.split("/d/")[1].split("/")[0]
            doc = docs_service.documents().get(documentId=doc_id).execute()
            title = doc.get("title", "Google Doc")
            text_content = " ".join([c.get("paragraph", {}).get("elements", [{}])[0].get("textRun", {}).get("content", "")
                                     for c in doc.get("body", {}).get("content", [])])
        elif "atlassian.net/wiki" in link and confluence:
            page_id = link.split("/")[-1]
            page = confluence.get_page_by_id(page_id, expand="body.storage")
            title = page.get("title", "Confluence Page")
            html = page["body"]["storage"]["value"]
            soup = BeautifulSoup(html, "html.parser")
            text_content = soup.get_text()
        else:
            resp = requests.get(link)
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string if soup.title else "Web Page"
            text_content = soup.get_text()
    except Exception as e:
        return {"error": f"Failed to fetch/process link: {str(e)}"}

    if not text_content.strip():
        return {"error": "No readable text found on the page."}

    name = f"{title[:40]}.link"
    add_to_index(name, text_content)
    return {"message": f"Link '{title}' processed and indexed.", "filename": name}

@app.get("/list-docs")
def list_docs():
    return {"documents": [title for title, _ in documents]}

@app.delete("/delete-doc")
def delete_doc(filename: str):
    global documents, index
    to_delete = [i for i, (t, _) in enumerate(documents) if t.lower() == filename.lower()]
    if not to_delete:
        return {"error": f"No document found with filename '{filename}'."}

    for i in sorted(to_delete, reverse=True):
        documents.pop(i)

    index = faiss.IndexFlatL2(embedding_dim)
    if documents:
        embeddings = [embedding_model.encode([content])[0] for _, content in documents]
        index.add(np.array(embeddings, dtype="float32"))
    save_index()

    file_to_remove = os.path.join(UPLOADS_DIR, filename)
    if os.path.exists(file_to_remove):
        os.remove(file_to_remove)

    return {"message": f"Deleted '{filename}'.", "remaining_docs": len(documents)}

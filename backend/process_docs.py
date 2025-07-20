import faiss
import pickle
import numpy as np
import os
import requests
from dotenv import load_dotenv

# -----------------------
# Load environment
# -----------------------
load_dotenv()
PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")  # "huggingface", "openai", "local"
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Local embedding model if needed

# -----------------------
# Globals: FAISS Index + Docs
# -----------------------
embedding_dim = 384  # Dimension (depends on model)
index = faiss.IndexFlatL2(embedding_dim)
doc_names = []  # Stores (title, content) for all docs

# -----------------------
# Embedding Helper
# -----------------------
#def get_embedding(text):
#    if PROVIDER == "huggingface":
#        return get_hf_embedding(text)
#    elif PROVIDER == "openai":
#        from openai import OpenAI
 #       client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 #       response = client.embeddings.create(input=text, model="text-embedding-3-small")
 #       return response.data[0].embedding
 #   elif PROVIDER == "local":
 #       from sentence_transformers import SentenceTransformer
  #      model = SentenceTransformer(HF_MODEL)
 #       return model.encode([text])[0].tolist()
 #   else:
 #       raise ValueError(f"Unsupported provider: {PROVIDER}")

def get_hf_embedding(text):
    api_url = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
    headers = {"Authorization": f"Bearer " + HF_API_KEY}
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Hugging Face API Error {response.status_code}: {response.text}")

    result = response.json()
    # Mean-pool token embeddings into a single vector
    if isinstance(result, list) and isinstance(result[0], list):
        arr = np.array(result)
        return np.mean(arr, axis=0).tolist()
    return result

# -----------------------
# Save / Load Index
# -----------------------
def save_index():
    os.makedirs("embeddings_store", exist_ok=True)
    faiss.write_index(index, "embeddings_store/index.faiss")
    with open("embeddings_store/docs.pkl", "wb") as f:
        pickle.dump(doc_names, f)
    print(f"Index & {len(doc_names)} documents saved!")

def load_existing_index():
    """Load previously saved FAISS index + documents."""
    global index, doc_names
    if os.path.exists("embeddings_store/index.faiss") and os.path.exists("embeddings_store/docs.pkl"):
        index = faiss.read_index("embeddings_store/index.faiss")
        with open("embeddings_store/docs.pkl", "rb") as f:
            doc_names = pickle.load(f)
        print(f"Loaded existing index with {len(doc_names)} documents.")
    else:
        print("No existing index found. Starting fresh.")

# -----------------------
# Add New Docs (Incremental)
# -----------------------
def process_new_docs(new_docs: dict):
    """
    Takes a dictionary like:
    {
        "Doc Title": "Content here...",
        "Another Title": "Some text..."
    }
    Embeds, adds to FAISS, saves index.
    """
    load_existing_index()  # Ensure we have any old docs loaded

    vectors = []
    for title, content in new_docs.items():
        emb = get_hf_embedding(content)
        vectors.append(emb)
        doc_names.append((title, content))
        print(f"Indexed new doc: {title} (embedding size {len(emb)})")

    if vectors:
        index.add(np.array(vectors, dtype="float32"))
        save_index()
    else:
        print("No vectors to add (check your documents).")

# -----------------------
# Script Run (Testing)
# -----------------------
if __name__ == "__main__":
    # Example initial run (mock docs)
    test_docs = {
        "Refund Policy": """
        Customers can request a refund within 14 days of purchase for all digital products.
        Physical goods must be returned within 30 days in their original packaging.
        Refunds will be processed within 5-7 business days.
        For any issues, contact support@company.com.
        """,

        "Employee Handbook": """
        All employees are entitled to 20 vacation days per year.
        Sick leave requires a doctor's certificate for absences longer than 2 days.
        Remote work must be approved by the direct manager.
        Company laptops must be returned when an employee leaves.
        """,

        "IT Security Guidelines": """
        Employees must reset passwords every 90 days.
        Multi-factor authentication (MFA) is mandatory for all company accounts.
        USB drives are prohibited for transferring sensitive company data.
        Report phishing emails immediately to it-support@company.com.
        """,

        "Travel Reimbursement": """
        Employees can claim reimbursement for travel expenses related to business trips.
        All claims must be submitted with receipts within 15 days of travel.
        The company will not cover personal entertainment, alcohol, or luxury items.
        Domestic flights must be booked in economy class unless pre-approved.
        """,

        "Work From Home Policy": """
        Employees may work remotely up to 3 days per week.
        A stable internet connection is required.
        The company will not reimburse home office equipment unless approved.
        Work hours remain the same as office hours (9 AM - 6 PM).
        """
    }
    process_new_docs(test_docs)
    print("Documents processed and saved.")

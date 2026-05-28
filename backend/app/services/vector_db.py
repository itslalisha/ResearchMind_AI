import os
import json
import time
import urllib.request
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_FAISS_PATH = os.path.join(os.path.dirname(__file__), "../../../local_faiss_index")

# --- ULTIMATE BYPASS: Direct HTTP REST API (No SDK Bugs) ---
class DirectRESTEmbeddings:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Using the standard, highly accurate text-embedding-004 model
        self.model_name = "text-embedding-004"
        self.batch_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:batchEmbedContents?key={self.api_key}"
        self.single_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:embedContent?key={self.api_key}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds multiple chunks directly via Google's raw REST API."""
        embeddings = []
        batch_size = 15  # Optimal batch size for fast cloud processing
        headers = {'Content-Type': 'application/json'}

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.model_name}",
                        "content": {"parts": [{"text": text}]}
                    } for text in batch
                ]
            }
            
            req = urllib.request.Request(
                self.batch_url, 
                data=json.dumps(payload).encode('utf-8'), 
                headers=headers
            )
            
            try:
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read().decode())
                    for item in data.get('embeddings', []):
                        embeddings.append(item['values'])
            except Exception as e:
                # Catch detailed exact error from Google's server
                error_details = e.read().decode() if hasattr(e, 'read') else str(e)
                raise RuntimeError(f"Raw REST API Batch Error: {error_details}")
                
            time.sleep(0.3)  # Small breather for the API rate limits
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single search query directly via Google's raw REST API."""
        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": text}]}
        }
        
        req = urllib.request.Request(
            self.single_url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers=headers
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                return data['embedding']['values']
        except Exception as e:
            error_details = e.read().decode() if hasattr(e, 'read') else str(e)
            raise RuntimeError(f"Raw REST API Query Error: {error_details}")


class VectorDBService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = None

    def configure_embeddings(self, api_key: str):
        self.embeddings = DirectRESTEmbeddings(api_key=api_key)

    def create_and_save_index(self, extracted_pages: list) -> str:
        if not self.embeddings:
            raise ValueError("Embeddings not configured. Please pass API key first.")
            
        documents = []
        for page in extracted_pages:
            text = ""
            if isinstance(page, dict):
                text = page.get("text") or page.get("content") or page.get("page_text") or ""
                if not text:
                    for val in page.values():
                        if isinstance(val, str) and len(val.strip()) > len(text):
                            text = val
            elif isinstance(page, str):
                text = page
                
            text = str(text).strip()
            if not text or len(text) < 5:
                continue
            
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": page.get("source", "Uploaded_Paper") if isinstance(page, dict) else "Uploaded_Paper",
                        "page": page.get("page_number", 0) if isinstance(page, dict) else 0
                    }
                )
                documents.append(doc)
                
        if not documents:
            return "No text contents found to index."
            
        db = FAISS.from_documents(documents, self.embeddings)
        db.save_local(DB_FAISS_PATH)
        return f"Successfully indexed {len(documents)} chunks from the paper."

    def similarity_search(self, query: str, k: int = 4):
        if not self.embeddings:
            raise ValueError("Embeddings not configured. Please pass API key first.")
            
        if not os.path.exists(DB_FAISS_PATH):
            raise FileNotFoundError("No vector store found. Please process a research paper first.")
            
        db = FAISS.load_local(DB_FAISS_PATH, self.embeddings, allow_dangerous_deserialization=True)
        return db.similarity_search(query, k=k)
import os
import time
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_FAISS_PATH = os.path.join(os.path.dirname(__file__), "../../../local_faiss_index")

class DirectGoogleEmbeddings:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model_name = "models/text-embedding-004"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds texts in small, lightning-fast batches to prevent Render timeouts."""
        embeddings = []
        # Google API accepts up to 100 texts per batch, we'll use 20 for extreme speed & safety
        batch_size = 20
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    # Removing task_type forces general fast embedding layout
                )
                embeddings.extend(response['embedding'])
                time.sleep(0.1)  # Micro-sleep to avoid rate limits
            except Exception as e:
                raise RuntimeError(f"Google Embedding Batch Failed: {str(e)}")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string seamlessly."""
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=text
            )
            return response['embedding']
        except Exception as e:
            raise RuntimeError(f"Google Query Embedding Failed: {str(e)}")


class VectorDBService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,       # Made slightly smaller for faster parallel processing
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = None

    def configure_embeddings(self, api_key: str):
        self.embeddings = DirectGoogleEmbeddings(api_key=api_key)

    def create_and_save_index(self, extracted_pages: list) -> str:
        if not self.embeddings:
            raise ValueError("Embeddings not configured. Please pass API key first.")
            
        documents = []
        for page in extracted_pages:
            text = page.get("text", "").strip()
            if not text:
                continue
            
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": page.get("source", "Unknown"),
                        "page": page.get("page_number", 0)
                    }
                )
                documents.append(doc)
                
        if not documents:
            return "No text contents found to index."
            
        # Build FAISS database and save locally
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
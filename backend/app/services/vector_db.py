import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_FAISS_PATH = os.path.join(os.path.dirname(__file__), "../../../local_faiss_index")

# --- Custom Wrapper to force Official Google SDK ---
class DirectGoogleEmbeddings:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model_name = "models/text-embedding-004"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of texts using the official SDK."""
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=texts,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            raise RuntimeError(f"Official Google SDK Embedding Failed: {str(e)}")

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string using the official SDK."""
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return response['embedding'][0] if isinstance(response['embedding'][0], list) else response['embedding']
        except Exception as e:
            raise RuntimeError(f"Official Google SDK Query Embedding Failed: {str(e)}")


class VectorDBService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = None

    def configure_embeddings(self, api_key: str):
        """Initializes the custom official SDK wrapper instead of LangChain's wrapper."""
        self.embeddings = DirectGoogleEmbeddings(api_key=api_key)

    def create_and_save_index(self, extracted_pages: list) -> str:
        if not self.embeddings:
            raise ValueError("Embeddings not configured. Please pass API key first.")
            
        documents = []
        for page in extracted_pages:
            text = page.get("text", "").strip()
            if not text:
                continue
            
            # Split page text into manageable chunks
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
            
        # Build FAISS database and save locally on the server instance
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
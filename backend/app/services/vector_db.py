import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_FAISS_PATH = os.path.join(os.path.dirname(__file__), "../../../local_faiss_index")

class VectorDBService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Using Local HuggingFace Embeddings (No API Key required for embeddings)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def create_and_save_index(self, extracted_pages: list) -> str:
        documents = []
        
        # Smart Page Number Extractor using enumerate (Index 0 becomes Page 1)
        for idx, page in enumerate(extracted_pages):
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
            
            actual_page_num = idx + 1
            
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": page.get("source", "Uploaded_Paper") if isinstance(page, dict) else "Uploaded_Paper",
                        "page": actual_page_num
                    }
                )
                documents.append(doc)
                
        if not documents:
            return "No text contents found to index."
            
        if os.path.exists(DB_FAISS_PATH):
            shutil.rmtree(DB_FAISS_PATH)
            
        db = FAISS.from_documents(documents, self.embeddings)
        db.save_local(DB_FAISS_PATH)
        return f"Successfully indexed {len(documents)} chunks locally."

    def similarity_search(self, query: str, k: int = 4):
        if not os.path.exists(DB_FAISS_PATH):
            raise FileNotFoundError("No vector store found. Please process a research paper first.")
            
        db = FAISS.load_local(DB_FAISS_PATH, self.embeddings, allow_dangerous_deserialization=True)
        return db.similarity_search(query, k=k)
import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Define where to store the vector index locally on disk
DB_FAISS_PATH = os.path.join(os.path.dirname(__file__), "../../../local_faiss_store")

class VectorDBService:
    def __init__(self):
        # 1. Text splitter ko __init__ ke andar hi rakhein (4 spaces indent)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Shuruat mein embeddings None rahegi, jab tak configure_embeddings call na ho
        self.embeddings = None

    # 2. Yeh ek alag method hona chahiye (Class ke parallel, 4 spaces indent)
    def configure_embeddings(self, api_key: str):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
       )

    def create_and_save_index(self, extracted_pages: List[Dict[str, Any]]) -> str:
        """
        Converts extracted page dictionaries into LangChain Documents,
        chunks them, generates vector embeddings, and stores them in FAISS.
        """
        raw_documents = []
        for page in extracted_pages:
            doc = Document(
                page_content=page["page_content"],
                metadata=page["metadata"]
            )
            raw_documents.append(doc)
        
        # Split documents into smaller semantic chunks
        split_docs = self.text_splitter.split_documents(raw_documents)
        
        # Build the FAISS vector database index
        vector_store = FAISS.from_documents(split_docs, self.embeddings)
        
        # Persist the vector index locally to disk so we don't re-embed every time
        vector_store.save_local(DB_FAISS_PATH)
        return f"Successfully indexed {len(split_docs)} text chunks."

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Queries the local FAISS index to find the top 'k' most relevant context chunks.
        """
        if not os.path.exists(DB_FAISS_PATH):
            raise FileNotFoundError("FAISS index has not been created yet. Please upload a PDF first.")
        
        # Load the index from disk
        vector_store = FAISS.load_local(
            DB_FAISS_PATH, 
            self.embeddings, 
            allow_dangerous_deserialization=True # Required by LangChain for loading local pickles securely
        )
        
        # Perform similarity lookup
        results = vector_store.similarity_search(query, k=k)
        return results
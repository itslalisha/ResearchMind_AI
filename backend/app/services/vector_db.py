import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Define where to store the vector index locally on disk
DB_FAISS_PATH = os.path.join(os.path.dirname(__file__), "../../../local_faiss_store")

class VectorDBService:
    def __init__(self):
        # Using a highly-rated, lightweight open-source embedding model
        # 'all-MiniLM-L6-v2' runs fast locally on CPU and yields great semantic search scores
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        # 1000 character chunks with a 200 character overlap keeps structural text intact
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
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
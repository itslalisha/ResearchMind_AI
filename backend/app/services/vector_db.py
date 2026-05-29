import os
import shutil

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# =========================================================
# FAISS PATH
# =========================================================
DB_FAISS_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../local_faiss_index"
)

# =========================================================
# VECTOR DATABASE SERVICE
# =========================================================
class VectorDBService:

    def __init__(self):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        self.embeddings = None

    # =====================================================
    # CONFIGURE EMBEDDINGS
    # =====================================================
    def configure_embeddings(self, api_key: str):

        if not api_key:
            raise ValueError("GOOGLE_API_KEY missing")

        # OFFICIAL WORKING GEMINI EMBEDDINGS
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )

    # =====================================================
    # CREATE + SAVE INDEX
    # =====================================================
    def create_and_save_index(
        self,
        extracted_pages: list
    ) -> str:

        if not self.embeddings:
            raise ValueError(
                "Embeddings not configured."
            )

        documents = []

        for page in extracted_pages:

            text = ""

            # Handle dict
            if isinstance(page, dict):

                text = (
                    page.get("text")
                    or page.get("content")
                    or page.get("page_text")
                    or ""
                )

                # fallback
                if not text:

                    for val in page.values():

                        if (
                            isinstance(val, str)
                            and len(val.strip()) > len(text)
                        ):
                            text = val

            # Handle string
            elif isinstance(page, str):
                text = page

            text = str(text).strip()

            # Skip empty text
            if not text or len(text) < 5:
                continue

            chunks = self.text_splitter.split_text(text)

            for chunk in chunks:

                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": (
                            page.get(
                                "source",
                                "Uploaded_Paper"
                            )
                            if isinstance(page, dict)
                            else "Uploaded_Paper"
                        ),
                        "page": (
                            page.get(
                                "page_number",
                                0
                            )
                            if isinstance(page, dict)
                            else 0
                        )
                    }
                )

                documents.append(doc)

        if not documents:
            return "No text contents found."

        # DELETE OLD INDEX
        if os.path.exists(DB_FAISS_PATH):
            shutil.rmtree(DB_FAISS_PATH)

        # CREATE FAISS
        db = FAISS.from_documents(
            documents,
            self.embeddings
        )

        # SAVE
        db.save_local(DB_FAISS_PATH)

        return (
            f"Successfully indexed "
            f"{len(documents)} chunks."
        )

    # =====================================================
    # SIMILARITY SEARCH
    # =====================================================
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ):

        if not self.embeddings:
            raise ValueError(
                "Embeddings not configured."
            )

        if not os.path.exists(DB_FAISS_PATH):
            raise FileNotFoundError(
                "No vector DB found."
            )

        db = FAISS.load_local(
            DB_FAISS_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        return db.similarity_search(
            query,
            k=k
        )
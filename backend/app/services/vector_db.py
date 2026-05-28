import os
import json
import time
import shutil
import urllib.request
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DB_FAISS_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../local_faiss_index"
)


class DirectRESTEmbeddings:
    def __init__(self, api_key: str):

        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing.")

        self.api_key = api_key

        # Latest working embedding model
        self.model_name = "text-embedding-004"

        # IMPORTANT: use v1 (NOT v1beta)
        self.batch_url = (
            f"https://generativelanguage.googleapis.com/v1/"
            f"models/{self.model_name}:batchEmbedContents"
            f"?key={self.api_key}"
        )

        self.single_url = (
            f"https://generativelanguage.googleapis.com/v1/"
            f"models/{self.model_name}:embedContent"
            f"?key={self.api_key}"
        )


    def embed_documents(self, texts: list[str]) -> list[list[float]]:

        embeddings = []

        batch_size = 15

        headers = {
            "Content-Type": "application/json"
        }

        for i in range(0, len(texts), batch_size):

            batch = texts[i:i + batch_size]

            payload = {
                "requests": [
                    {
                        "content": {
                            "parts": [
                                {"text": text}
                            ]
                        }
                    }
                    for text in batch
                ]
            }

            req = urllib.request.Request(
                self.batch_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )

            try:
                with urllib.request.urlopen(req) as response:

                    data = json.loads(
                        response.read().decode()
                    )

                    if "embeddings" not in data:
                        raise RuntimeError(
                            f"No embeddings returned: {data}"
                        )

                    for item in data["embeddings"]:
                        embeddings.append(item["values"])

            except Exception as e:

                error_details = (
                    e.read().decode()
                    if hasattr(e, "read")
                    else str(e)
                )

                raise RuntimeError(
                    f"Google Embedding Batch Failed: "
                    f"{error_details}"
                )

            # Avoid rate limits
            time.sleep(1)

        return embeddings


    def embed_query(self, text: str) -> list[float]:

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "content": {
                "parts": [
                    {"text": text}
                ]
            }
        }

        req = urllib.request.Request(
            self.single_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req) as response:

                data = json.loads(
                    response.read().decode()
                )

                if "embedding" not in data:
                    raise RuntimeError(
                        f"No embedding returned: {data}"
                    )

                return data["embedding"]["values"]

        except Exception as e:

            error_details = (
                e.read().decode()
                if hasattr(e, "read")
                else str(e)
            )

            raise RuntimeError(
                f"Google Embedding Query Failed: "
                f"{error_details}"
            )


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

        self.embeddings = DirectRESTEmbeddings(
            api_key=api_key
        )


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

            # Handle dict pages
            if isinstance(page, dict):

                text = (
                    page.get("text")
                    or page.get("content")
                    or page.get("page_text")
                    or ""
                )

                # fallback: longest string
                if not text:

                    for val in page.values():

                        if (
                            isinstance(val, str)
                            and len(val.strip()) > len(text)
                        ):
                            text = val

            # Handle string pages
            elif isinstance(page, str):

                text = page

            text = str(text).strip()

            # Skip empty chunks
            if not text or len(text) < 5:
                continue

            # Split text
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
            return "No text contents found to index."

        # Delete old FAISS index
        if os.path.exists(DB_FAISS_PATH):
            shutil.rmtree(DB_FAISS_PATH)

        # Create FAISS DB
        db = FAISS.from_documents(
            documents,
            self.embeddings
        )

        # Save DB
        db.save_local(DB_FAISS_PATH)

        return (
            f"Successfully indexed "
            f"{len(documents)} chunks."
        )


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
                "No vector store found. "
                "Please upload a paper first."
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
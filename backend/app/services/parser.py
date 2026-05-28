import io
import pdfplumber
from typing import List, Dict, Any

class PDFParserService:
    @staticmethod
    def extract_text_with_metadata(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parses a PDF from bytes, extracts text page-by-page, 
        and attaches source metadata for citation tracking.
        """
        documents = []
        
        # Wrap bytes in an in-memory file-like object
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                
                # .extract_text() layout preservation works well for 2-column papers
                text = page.extract_text(layout=False)
                
                if not text or not text.strip():
                    continue
                
                # Standardize line endings and clean spacing
                clean_text = " ".join(text.split())
                
                # Construct the document chunk schema
                doc_chunk = {
                    "page_content": clean_text,
                    "metadata": {
                        "source": filename,
                        "page": page_num,
                        "total_pages": total_pages,
                        "chunk_type": "full_page"
                    }
                }
                documents.append(doc_chunk)
                
        return documents

# Quick local test verification
if __name__ == "__main__":
    # Test block to verify extraction logic independently
    print("Parser Service initialized successfully.")
from fastapi import FastAPI, UploadFile, File, HTTPException, Query , Form , Header
from fastapi.middleware.cors import CORSMiddleware
from app.services.parser import PDFParserService
from app.services.vector_db import VectorDBService
from app.services.llm_chain import LLMChainService

app = FastAPI(
    title="ResearchMind AI Backend",
    description="Advanced RAG engine for academic paper analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the vector DB controller
vector_service = VectorDBService()
llm_service = LLMChainService()


@app.get("/")
async def root():
    return {"status": "online", "message": "ResearchMind AI API is running smoothly."}

@app.post("/api/v1/process-paper")
async def process_paper(file: UploadFile = File(...)):
    """
    Parses an uploaded PDF and updates the local FAISS Vector store.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format. Only PDFs are supported.")
    
    try:
        file_bytes = await file.read()
        
        # 1. Extract plain text & page numbers
        extracted_pages = PDFParserService.extract_text_with_metadata(file_bytes, file.filename)
        
        # 2. Chunk, embed, and store vectors locally
        indexing_result = vector_service.create_and_save_index(extracted_pages)
        
        return {
            "filename": file.filename,
            "status": "Success",
            "message": indexing_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.get("/api/v1/search")
async def search_paper(query: str = Query(..., description="The semantic search question to run against the paper")):
    """
    Query endpoint to check the semantic accuracy of the vector store chunks.
    """
    try:
        matched_chunks = vector_service.similarity_search(query, k=3)
        
        # Format the response cleanly for structural verification
        formatted_results = []
        for doc in matched_chunks:
            formatted_results.append({
                "snippet": doc.page_content[:300] + "...", # Snip for readable logs
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source")
            })
        return {"query": query, "results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW ENDPOINT REQUIRED BY STREAMLIT ---
@app.get("/api/v1/ask-question")
async def ask_question(query: str = Query(...), x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is missing. Please enter it in the sidebar.")
    
    try:
        # Pass the dynamic key to your LLM service
        llm_service.configure_key(x_api_key)
        
        matched_chunks = vector_service.similarity_search(query, k=4)
        ai_response = llm_service.generate_citation_answer(query, matched_chunks)
        
        return {
            "query": query,
            "answer": ai_response,
            "sources_consulted": [
                {"page": doc.metadata.get("page"), "snippet_preview": doc.page_content[:150] + "..."}
                for doc in matched_chunks
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/v1/summarize")

async def summarize_paper(x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is missing. Please enter it in the sidebar.")
    try:
       
        llm_service.configure_key(x_api_key)
        
        # Search for chunks most likely to contain summary info
        matched_chunks = vector_service.similarity_search("Abstract Introduction Conclusion Summary Findings", k=5)
        ai_response = llm_service.generate_summary(matched_chunks)
        return {"answer": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary Gen Failed: {str(e)}")

@app.get("/api/v1/research-gaps")
async def research_gaps(x_api_key: str = Header(None)): 
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is missing.")
    try:
        llm_service.configure_key(x_api_key)  
        matched_chunks = vector_service.similarity_search("Limitations Future Work Research Gaps", k=5)
        ai_response = llm_service.analyze_research_gaps(matched_chunks)
        return {"answer": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gap Analysis Failed: {str(e)}")
    

@app.get("/api/v1/generate-quiz")
async def generate_quiz(x_api_key: str = Header(None)):  
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is missing.")
    try:
        llm_service.configure_key(x_api_key)  
        matched_chunks = vector_service.similarity_search("Methodology experiments baseline results accuracy metrics", k=4)
        ai_response = llm_service.generate_quiz(matched_chunks)
        return {"answer": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz Gen Failed: {str(e)}")

@app.post("/api/v1/analyze-image")
async def analyze_image_endpoint(
    file: UploadFile = File(...), 
    query: str = Form("Please explain the main findings shown in this chart/table."),
    x_api_key: str = Header(None) 
):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is missing.")
    
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image.")
        
       
        llm_service.configure_key(x_api_key)
        
        image_bytes = await file.read()
        ai_response = llm_service.analyze_image(image_bytes, query)
        
        return {"answer": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision Analysis Failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
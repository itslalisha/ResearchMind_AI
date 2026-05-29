from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Header
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

# Initialize local controllers
vector_service = VectorDBService()
llm_service = LLMChainService()

@app.get("/")
async def root():
    return {"status": "online", "message": "ResearchMind AI API is running locally."}

@app.post("/api/v1/process-paper")
async def process_paper(file: UploadFile = File(...)):
    """Parses PDF and runs local HuggingFace Embeddings"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format.")
    try:
        file_bytes = await file.read()
        extracted_pages = PDFParserService.extract_text_with_metadata(file_bytes, file.filename)
        indexing_result = vector_service.create_and_save_index(extracted_pages)
        return {"filename": file.filename, "status": "Success", "message": indexing_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.get("/api/v1/search")
async def search_paper(query: str = Query(...)):
    try:
        matched_chunks = vector_service.similarity_search(query, k=3)
        formatted_results = [
            {"snippet": doc.page_content[:300] + "...", "page": doc.metadata.get("page"), "source": doc.metadata.get("source")}
            for doc in matched_chunks
        ]
        return {"query": query, "results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ask-question")
async def ask_question(query: str = Query(...), x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is missing.")
    try:
        llm_service.configure_key(x_api_key)
        matched_chunks = vector_service.similarity_search(query, k=4)
        ai_response = llm_service.generate_citation_answer(query, matched_chunks)
        
        # Frontend dropdown ke liye page numbers ke sath text snippet bhi bhej rahe hain
        source_data = [
            {
                "page": doc.metadata.get("page", "Unknown"),
                "snippet": doc.page_content[:300] + "..."  # Shuruwaati 300 characters ka preview
            } 
            for doc in matched_chunks
        ]
        
        return {"query": query, "answer": ai_response, "sources": source_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/summarize")
async def summarize_paper(x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is missing.")
    try:
        llm_service.configure_key(x_api_key)
        matched_chunks = vector_service.similarity_search("Abstract Introduction Conclusion Summary Findings", k=5)
        ai_response = llm_service.generate_summary(matched_chunks)
        
        # Adding sources for the frontend dropdown
        source_data = [
            {"page": doc.metadata.get("page", "Unknown"), "snippet": doc.page_content[:300] + "..."} 
            for doc in matched_chunks
        ]
        return {"answer": ai_response, "sources": source_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/research-gaps")
async def research_gaps(x_api_key: str = Header(None)): 
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is missing.")
    try:
        llm_service.configure_key(x_api_key)  
        matched_chunks = vector_service.similarity_search("Limitations Future Work Research Gaps", k=5)
        ai_response = llm_service.analyze_research_gaps(matched_chunks)
        
        # Adding sources for the frontend dropdown
        source_data = [
            {"page": doc.metadata.get("page", "Unknown"), "snippet": doc.page_content[:300] + "..."} 
            for doc in matched_chunks
        ]
        return {"answer": ai_response, "sources": source_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/generate-quiz")
async def generate_quiz(x_api_key: str = Header(None)):  
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is missing.")
    try:
        llm_service.configure_key(x_api_key)  
        matched_chunks = vector_service.similarity_search("Methodology experiments baseline results accuracy metrics", k=4)
        ai_response = llm_service.generate_quiz(matched_chunks)
        
        # Adding sources for the frontend dropdown
        source_data = [
            {"page": doc.metadata.get("page", "Unknown"), "snippet": doc.page_content[:300] + "..."} 
            for doc in matched_chunks
        ]
        return {"answer": ai_response, "sources": source_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

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
        
        # Configure key for vision model
        llm_service.configure_key(x_api_key)
        
        image_bytes = await file.read()
        ai_response = llm_service.analyze_image(image_bytes, query)
        
        return {"answer": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision Analysis Failed: {str(e)}")    
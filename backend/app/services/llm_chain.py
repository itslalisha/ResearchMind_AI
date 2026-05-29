import os
import google.generativeai as genai
from typing import List
from langchain_core.documents import Document
from PIL import Image
import io

class LLMChainService:
    def __init__(self):
        # Do NOT configure the API key here anymore!
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def configure_key(self, api_key: str):
        # This dynamically sets the key for the user's specific request
        genai.configure(api_key=api_key)

    def generate_citation_answer(self, query: str, matched_chunks: list) -> str:
        """Generates a detailed, conversational Q&A answer with natural citations."""
        
        # Structuring context clearly for the LLM
        context_parts = []
        for doc in matched_chunks:
            page_num = doc.metadata.get("page", "Unknown")
            context_parts.append(f"--- Excerpt from Page {page_num} ---\n{doc.page_content}")
            
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""
        You are a highly intelligent and helpful academic research assistant. Answer the user's question comprehensively based ONLY on the provided paper excerpts.
        
        Guidelines for your response:
        1. **Be Detailed & Elaborative:** Do not just give a one-word or one-line list. Explain what each point means based on the text.
        2. **Excellent Formatting:** Use bold text, bullet points, and proper paragraphs to make the answer easy to read.
        3. **Natural Citations:** Cite the source pages naturally at the end of a complete sentence or explanation using the format (Page X). 
        4. **DO NOT** repeat the page number after every single word or short phrase.
        
        User's Question: {query}
        
        Provided Context:
        {context_text}
        """
        response = self.model.generate_content(prompt)
        return response.text
    

    def generate_summary(self, matched_chunks: list) -> str:
        # Pushing page numbers into the context
        context_parts = []
        for doc in matched_chunks:
            page_num = doc.metadata.get("page", "Unknown")
            context_parts.append(f"[Page {page_num}]: {doc.page_content}")
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""
        You are an expert academic researcher. Please read the following excerpts from a research paper and provide a HIGHLY DETAILED and COMPREHENSIVE summary. 
        
        Your response MUST include:
        1. A detailed introduction of the core problem.
        2. The main methodology or approach used.
        3. Key findings and results (in detailed bullet points).
        
        CRITICAL RULE: You must naturally cite the page numbers at the end of relevant sentences using the format (Page X).
        
        Paper Excerpts:
        {context_text}
        """
        response = self.model.generate_content(prompt)
        return response.text

    def analyze_research_gaps(self, matched_chunks: list) -> str:
        context_parts = []
        for doc in matched_chunks:
            page_num = doc.metadata.get("page", "Unknown")
            context_parts.append(f"[Page {page_num}]: {doc.page_content}")
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""
        You are an expert academic reviewer. Analyze the following research paper excerpts to deeply identify research gaps, limitations, and future work.
        
        Please structure your highly detailed response as follows:
        - **Core Limitations:** (Detail what the paper missed).
        - **Future Research Directions:** (Provide actionable suggestions).
        
        CRITICAL RULE: Cite the source pages naturally for every point you make using the format (Page X).

        Paper Excerpts:
        {context_text}
        """
        response = self.model.generate_content(prompt)
        return response.text

    def generate_quiz(self, matched_chunks: list) -> str:
        context_parts = []
        for doc in matched_chunks:
            page_num = doc.metadata.get("page", "Unknown")
            context_parts.append(f"[Page {page_num}]: {doc.page_content}")
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""
        Generate a comprehensive quiz based on the following research paper excerpts.
        
        Create 5 Multiple Choice Questions (MCQs). For each question:
        1. Write the question clearly.
        2. Provide 4 distinct options (A, B, C, D).
        3. Clearly state the **Correct Answer**.
        4. Provide a **Detailed Explanation** and cite the specific page number where the answer was found, formatted as (Source: Page X).

        Paper Excerpts:
        {context_text}
        """
        response = self.model.generate_content(prompt)
        return response.text
    
    def analyze_image(self, image_bytes: bytes, user_query: str) -> str:
        # Convert raw bytes into a format Gemini can see
        image = Image.open(io.BytesIO(image_bytes))

        prompt = f"""
        You are an expert data scientist and academic researcher. 
        Analyze this image (which is a chart, table, or diagram from a research paper).
        User Question: {user_query}

        Provide a detailed, highly accurate explanation based ONLY on what is visible in the image.
        """

        # Pass BOTH the text prompt and the image object to Gemini
        response = self.model.generate_content([prompt, image])
        return response.text    

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

    def generate_citation_answer(self, query: str, context_chunks: List[Document]) -> str:
        """
        Takes raw vector store documents, packages them with citation markers,
        and prompts Gemini to answer strictly based on the provided text.
        """
        formatted_context = ""
        for idx, doc in enumerate(context_chunks):
            page_num = doc.metadata.get("page", "Unknown")
            source_file = doc.metadata.get("source", "Document")
            formatted_context += f"--- Context Block {idx+1} [Source: {source_file} | Page: {page_num}] ---\n"
            formatted_context += f"{doc.page_content}\n\n"

        prompt = f"""
You are the advanced core engine of ResearchMind AI, an elite assistant helping researchers analyze technical papers.

User Question: "{query}"

Retrieved Context from the Research Paper:
{formatted_context}

CRITICAL INSTRUCTIONS:
1. Rely ONLY on the clear facts stated directly within the Context Blocks above. Do not assume or extrapolate beyond this data.
2. You MUST cite the source page numbers directly inside your answers next to the facts you present (e.g., "The model achieves 94% accuracy [Page 4].").
3. If the retrieved context does not contain enough information to decisively answer the user's question, state clearly that the necessary details are not present in the current index. Do not hallucinate.

Provide a clear, highly structured academic breakdown response with clear citations:
"""
        response = self.model.generate_content(prompt)
        return response.text
    

    def generate_summary(self, context_chunks: List[Document]) -> str:
        formatted_context = "\n\n".join([doc.page_content for doc in context_chunks])
        prompt = f"""
You are an expert AI research assistant. Based strictly on the provided paper excerpts below, write a highly structured, 3-part summary of the paper.

Context:
{formatted_context}

Format your response exactly like this:
### 🎯 Objective
(What problem is the paper trying to solve?)
### ⚙️ Core Methodology
(How did they solve it?)
### 🏆 Key Results
(What were the final metrics or findings?)
"""
        return self.model.generate_content(prompt).text

    def analyze_research_gaps(self, context_chunks: List[Document]) -> str:
        formatted_context = "\n\n".join([f"[Page: {doc.metadata.get('page')}] {doc.page_content}" for doc in context_chunks])
        prompt = f"""
You are an elite academic reviewer. Analyze the provided excerpts from this paper and identify the limitations and future research gaps.

Context:
{formatted_context}

Provide a bulleted list of 3 to 5 clear research gaps or limitations mentioned or implied in the text. Cite the page numbers.
"""
        return self.model.generate_content(prompt).text

    def generate_quiz(self, context_chunks: List[Document]) -> str:
        formatted_context = "\n\n".join([doc.page_content for doc in context_chunks])
        prompt = f"""
You are an academic professor. Based on the provided paper excerpts, generate a short, challenging multiple-choice quiz (3 questions) to test the reader's understanding.

Context:
{formatted_context}

Format your response strictly like this:
**Question 1:** [Question text]
A) ...
B) ...
C) ...
D) ...
*Correct Answer:* [Answer] - [Brief explanation]

(Repeat for Question 2 and 3)
"""
        return self.model.generate_content(prompt).text
    
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

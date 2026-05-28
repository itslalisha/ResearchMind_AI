import streamlit as st  
import requests

# Set page configuration with a wide academic layout
st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INITIALIZE SESSION STATE FIRST ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

BACKEND_URL = "http://127.0.0.1:8000"

st.title("🤖 ResearchMind AI — Advanced Paper Assistant")
st.caption("Analyze dense research literature with automated source grounding and zero hallucinations.")

# --- SIDEBAR: Upload & Metadata Operations ---

with st.sidebar:
    st.header("📄 Document Center")
    
    # 1. New API Key Input Field
    user_api_key = st.text_input("🔑 Enter Google Gemini API Key", type="password")
    st.caption("Don't have one? [Get a free key here](https://aistudio.google.com/app/apikey)")
    st.divider()
    
    uploaded_file = st.file_uploader("Upload an Academic Paper (PDF)", type=["pdf"])
    
    # 2. Block the button if the key is missing
    if uploaded_file is not None:
        if st.button("🚀 Process & Index Paper", use_container_width=True, disabled=not user_api_key):
            with st.spinner("Parsing layout grids and initializing local vectors..."):
                try:
                    # 3. Send the API key securely in the Request Headers
                    headers = {"x-api-key": user_api_key}
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    
                    response = requests.post(f"{BACKEND_URL}/api/v1/process-paper", files=files, headers=headers)
                    # ... (rest of your response handling)
                    
                    if response.status_code == 200:
                        st.success(f"Indexed successfully!\n{response.json().get('message')}")
                    else:
                        st.error(f"Backend processing failure: {response.text}")
                except Exception as e:
                    st.error(f"Connection error to backend: {str(e)}")

# Add this inside the 'with st.sidebar:' block, at the very bottom
    st.divider()
    st.header("💾 Export Findings")
    
    if st.session_state.chat_history:
        # Convert chat history to a readable text format
        export_text = "# ResearchMind AI - Chat Notes\n\n"
        for msg in st.session_state.chat_history:
            role = "USER" if msg["role"] == "user" else "AI"
            export_text += f"### {role}:\n{msg['content']}\n\n---\n\n"
            
        st.download_button(
            label="📥 Download Chat as Markdown",
            data=export_text,
            file_name="Research_Notes.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.info("Start chatting to enable notes download.")

# --- MAIN SCREEN: Conversational RAG Engine UI ---
# Initialize session state tracking for persistent chat bubbles
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display existing chat message streams
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("🔍 Verified Page Citations"):
                for source in message["sources"]:
                    st.write(f"• **Page {source['page']}**: {source['snippet_preview']}")


# --- QUICK ACTION BUTTONS ---
st.markdown("### ⚡ Quick Actions")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("📝 Summarize Paper", use_container_width=True):
        if not user_api_key:
            st.warning("Please enter your Gemini API Key in the sidebar first!")
        else:
            st.session_state.chat_history.append({"role": "user", "content": "Can you summarize this paper for me?"})
            with st.spinner("Extracting Abstract and Conclusion..."):
                
                headers = {"x-api-key": user_api_key}
                res = requests.get(f"{BACKEND_URL}/api/v1/summarize", headers=headers)
                
                if res.status_code == 200:
                    st.session_state.chat_history.append({"role": "assistant", "content": res.json().get("answer")})
                    st.rerun()
                else:
                    st.error(f"Error: {res.text}")

with col2:
    if st.button("🔍 Find Research Gaps", use_container_width=True):
        if not user_api_key:
            st.warning("Please enter your Gemini API Key in the sidebar first!")
        else:
            st.session_state.chat_history.append({"role": "user", "content": "What are the limitations and future research gaps in this paper?"})
            with st.spinner("Analyzing limitations and future work..."):
               
                headers = {"x-api-key": user_api_key}
                res = requests.get(f"{BACKEND_URL}/api/v1/research-gaps", headers=headers)
                
                if res.status_code == 200:
                   st.session_state.chat_history.append({"role": "assistant", "content": res.json().get("answer")})
                   st.rerun()
                else:
                   st.error(f"Error: {res.text}")


with col3:
    if st.button("🧠 Generate Quiz", use_container_width=True):
        if not user_api_key:
            st.warning("Please enter your Gemini API Key in the sidebar first!")
        else:
            st.session_state.chat_history.append({"role": "user", "content": "Generate a quick quiz based on this paper."})
            with st.spinner("Crafting tricky questions..."):
               
                headers = {"x-api-key": user_api_key}
                res = requests.get(f"{BACKEND_URL}/api/v1/generate-quiz", headers=headers)
                
                if res.status_code == 200:
                   st.session_state.chat_history.append({"role": "assistant", "content": res.json().get("answer")})
                   st.rerun()
                else:
                   st.error(f"Error: {res.text}")


# --- MULTIMODAL VISION SECTION ---
with st.expander("👁️ Analyze a Chart, Table or Diagram (Multimodal Vision)"):
    st.markdown("Upload a screenshot of any complex graph or table from the paper, and ResearchMind AI will explain it.")
    
    # Image uploader
    vision_image = st.file_uploader("Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    vision_query = st.text_input("What do you want to know about this image?", value="Explain the key trends or data points in this image.")
    
    if vision_image and st.button("🚀 Analyze Image", use_container_width=True):
        # 1. Pehle check karein ki key hai ya nahi
        if not user_api_key:
            st.error("Please enter your Gemini API Key in the sidebar first!")
        else:
            st.image(vision_image, caption="Uploaded for Analysis", use_container_width=True)
            
            with st.spinner("ResearchMind AI is looking at your image..."):
                try:
                    files = {"file": (vision_image.name, vision_image.getvalue(), vision_image.type)}
                    data = {"query": vision_query}
                    
                   
                    headers = {"x-api-key": user_api_key}
                    
                   
                    res = requests.post(f"{BACKEND_URL}/api/v1/analyze-image", files=files, data=data, headers=headers)
                    
                    if res.status_code == 200:
                        st.success("Analysis Complete!")
                        st.markdown(res.json().get("answer"))
                        st.session_state.chat_history.append({"role": "user", "content": f"[Uploaded Image] {vision_query}"})
                        st.session_state.chat_history.append({"role": "assistant", "content": res.json().get("answer")})
                    else:
                        st.error(f"Error: {res.json().get('detail', res.text)}")
                except Exception as e:
                    st.error(f"Failed to connect: {str(e)}")

# Collect fresh user question input
if user_query := st.chat_input("Ask a question about the indexed methodologies or metrics..."):
    # 1. Check if key exists
    if not user_api_key:
        st.error("Please enter your Gemini API Key in the sidebar to ask questions.")
    else:
        # Render User Message bubble
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Query Backend
        with st.chat_message("assistant"):
            with st.spinner("Querying local vector space and synthesizing analytical context..."):
                try:
                   
                    headers = {"x-api-key": user_api_key}
                    
                    response = requests.get(f"{BACKEND_URL}/api/v1/ask-question", params={"query": user_query}, headers=headers)
                    
                    if response.status_code == 200:
                        payload = response.json()
                        answer_text = payload.get("answer")
                        sources_list = payload.get("sources_consulted", [])
                        
                        st.markdown(answer_text)
                        
                        with st.expander("🔍 Verified Page Citations"):
                            for source in sources_list:
                                st.write(f"• **Page {source['page']}**: {source['snippet_preview']}")
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer_text,
                            "sources": sources_list
                        })
                    else:
                        error_detail = response.json().get("detail", response.text)
                        st.error(f"Backend Error: {error_detail}")
                        
                except Exception as e:
                    st.error(f"Communication breakdown with AI Orchestrator: {str(e)}")
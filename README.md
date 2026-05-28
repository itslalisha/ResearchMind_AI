# 🤖 ResearchMind AI — Advanced AI Research Paper Assistant

ResearchMind AI is a full-stack **Retrieval-Augmented Generation (RAG)** application that helps researchers, students, and professionals interact intelligently with academic research papers.

The platform transforms static PDFs into interactive AI-powered assistants capable of:

* Answering context-aware questions
* Generating summaries
* Detecting research gaps
* Creating quizzes
* Explaining complex charts and diagrams using multimodal vision models

Built with **FastAPI, Streamlit, LangChain, FAISS, and Google Gemini**, the application provides a scalable and production-ready architecture for AI-powered document understanding.

---

# ✨ Features

## 📄 Smart PDF Processing

* Upload academic research papers in PDF format
* Automatic text extraction and chunking
* Semantic indexing using FAISS vector embeddings

## 💬 Citation-Based Question Answering

* Ask questions directly about the paper
* Responses include exact page references and citations
* Context-aware retrieval using RAG pipelines

## 📝 AI-Generated Summaries

Generate structured summaries including:

* Objective
* Methodology
* Key Findings
* Conclusion

## 🔍 Research Gap Analysis

Automatically identify:

* Limitations in the paper
* Future work opportunities
* Potential improvement areas

## 🧠 Quiz & MCQ Generation

* Generate dynamic multiple-choice questions
* Test understanding of core research concepts

## 👁️ Multimodal Vision Analysis

Upload:

* Charts
* Tables
* Architecture diagrams
* Graph screenshots

The Gemini Vision model explains trends, relationships, and visual insights.

## 💾 Export Notes

Download:

* Chat history
* Summaries
* Research insights
* Generated notes

as clean Markdown files.

## 🔐 Bring Your Own API Key (BYOK)

Users securely provide their own Google Gemini API Key, enabling:

* Unlimited scalability
* No centralized API cost
* Better privacy and control

---

# 🏗️ System Architecture

```text
                ┌────────────────────┐
                │   Streamlit UI     │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │    FastAPI API     │
                └─────────┬──────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
 ┌────────────┐   ┌──────────────┐   ┌─────────────┐
 │  LangChain │   │ Gemini API   │   │   FAISS     │
 │ Orchestrat.│   │ LLM + Vision │   │ Vector DB   │
 └────────────┘   └──────────────┘   └─────────────┘
```

---

# 🛠️ Tech Stack

| Category          | Technology              |
| ----------------- | ----------------------- |
| Frontend          | Streamlit               |
| Backend           | FastAPI                 |
| LLM Provider      | Google Gemini 2.5 Flash |
| Embeddings        | Sentence Transformers   |
| Vector Database   | FAISS                   |
| Frameworks        | LangChain               |
| PDF Parsing       | PyPDF                   |
| Vision Processing | Pillow (PIL)            |
| Language          | Python                  |

---

# 📂 Project Structure

```bash
ResearchMind-AI/
│
├── backend/
│   ├── app/
│   ├── requirements.txt
│   └── main.py
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
├── assets/
├── exports/
├── README.md
└── .gitignore
```

---

# 🚀 Installation & Local Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ResearchMind-AI.git
cd ResearchMind-AI
```

---

# ⚙️ Backend Setup (FastAPI)

## Navigate to Backend

```bash
cd backend
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Backend Server

```bash
uvicorn app.main:app --reload --port 8000
```

Backend will run at:

```bash
http://localhost:8000
```

---

# 🎨 Frontend Setup (Streamlit)

Open a new terminal.

## Navigate to Frontend

```bash
cd frontend
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run app.py
```

Frontend will run at:

```bash
http://localhost:8501
```

---

# 🔑 Google Gemini API Setup

1. Visit Google AI Studio:
   https://aistudio.google.com/

2. Generate your API key

3. Paste the key inside the application sidebar

---

# 📌 Usage Workflow

1. Upload a research paper PDF
2. System extracts and indexes content
3. Ask questions naturally
4. Generate:

   * Summaries
   * MCQs
   * Research gaps
   * Visual explanations
5. Export notes and findings

---

# 🌐 Deployment

## Backend Deployment

You can deploy the FastAPI backend on:

* Render
* Railway
* Heroku
* Docker
* AWS EC2

## Frontend Deployment

Deploy Streamlit frontend using:

* Streamlit Community Cloud

Before deployment, update:

```python
BACKEND_URL
```

inside:

```python
frontend/app.py
```

to your deployed backend URL.

---

# 📸 Future Improvements

* PDF highlighting with citations
* Multi-paper comparison
* Research recommendation engine
* ArXiv integration
* Voice-based paper interaction
* Collaborative annotation system

---

# 👨‍💻 Author

## Lalisha Lilhore

M.Tech in Artificial Intelligence
Indian Institute of Technology (IIT) Patna

### Interests

* Natural Language Processing (NLP)
* Large Language Models (LLMs)
* Multimodal AI
* Computer Vision
* Deep Learning
* Retrieval-Augmented Generation (RAG)

---

# 🤝 Contributing

Contributions are welcome!

## Steps

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project useful:

* Give the repository a star ⭐
* Share it with researchers and developers
* Contribute to improve the project

---

# 📧 Contact

For collaborations or queries:

* GitHub:https://github.com/itslalisha
* LinkedIn: https://www.linkedin.com/in/lalisha-lilhore-401657240/

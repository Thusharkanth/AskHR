# AskHR – A Local Domain-Specific HR Chatbot  

### 📌 Project Overview  
AskHR is a **domain-specific chatbot** built for **CompanyX** to handle **Human Resource (HR) queries** securely and efficiently. Unlike general-purpose chatbots, AskHR is designed to:  
- Answer only HR-related questions.  
- Run fully **offline (local deployment)** to ensure **data security**.  
- Use **Retrieval-Augmented Generation (RAG)** for accurate and up-to-date responses.  

### 🔑 Key Features  
- **HR-Specific**: Focused only on company HR policies, FAQs, and guidelines.  
- **Secure**: No data leaves the local environment.  
- **Efficient**: Uses RAG instead of fine-tuning to avoid heavy retraining.  
- **Scalable**: Backend via **Python + FastAPI**, with plans for UI integration.  

### ⚙️ Technical Stack  
- **Model**: LLaMA 2 (7B) via `llama.cpp`  
- **Frameworks**: Python, LangChain, HuggingFace, Sentence-Transformers  
- **Database**: ChromaDB (vector search for documents)  
- **Deployment**: Local setup → FastAPI → UI (planned)  

### 🚧 Challenges & Limitations  
- Hardware constraints (RTX 2050, 4 GB VRAM).  
- RAG performance depends on embedding quality & retrieval.  
- Steep learning curve with LLMs and AI frameworks.  

### 🚀 Future Enhancements  
- GPU-accelerated frameworks (vLLM, TensorRT-LLM).  
- Hybrid RAG (dense + keyword search).  
- Advanced embeddings (e.g., ColBERTv2, InstructorXL).  
- Hybrid approach (Fine-Tuning + RAG).  
- Web UI deployment inside MintHRM systems.  

---

📄 **Full Documentation** is included in the repo for detailed explanation.  

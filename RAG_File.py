import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp

PDF_PATH = r"C:\Users\User\Desktop\Mint HRM\company_chatbot\DATA\PRO_032844.pdf"
PERSIST_DIR = "./faiss_index"
LLAMA_MODEL_PATH = (
    r"C:\Users\User\Desktop\Mint HRM\company_chatbot\model\llama-2-7b-chat.Q3_K_M.gguf"
)

# 1) Load PDF llm
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # returns list of Document (each page/doc)

# 2) Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"[+] created {len(chunks)} chunks")


# 3) Embedding wrapper using SentenceTransformers
class SBEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embs = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embs]

    def embed_query(self, text):
        emb = self.model.encode([text], convert_to_numpy=True)
        return emb[0].tolist()


embeddings = SBEmbeddings()

# 4) Build / load FAISS index
if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
    print("[+] creating FAISS index (this may take a while)...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(PERSIST_DIR)
    print("[+] FAISS index saved to", PERSIST_DIR)
else:
    print("[+] loading FAISS index from", PERSIST_DIR)
    db = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)


# 5) LLM setup — choose one option
# Option A: llama-cpp-python (GGUF file)


llm = LlamaCpp(
    model_path="model/llama-2-7b-chat.Q3_K_M.gguf",
    verbose=False,  # disable logs
    n_ctx=2048,
    temperature=0.9,
)

# 6) Retrieval QA chain
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False
)

# 7) Interactive query loop
print("Ready — ask questions. Type 'exit' to quit.")
while True:
    q = input("\n> YOU:  ")
    if q.strip().lower() in ("exit", "quit"):
        break
    res = qa(q)
    answer = res["result"] if isinstance(res, dict) else res
    sources = res.get("source_documents", []) if isinstance(res, dict) else []
    print("\n--- ANSWER ---\n")
    print(answer.strip())

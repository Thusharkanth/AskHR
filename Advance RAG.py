# #Implemtentaion of RAG for a Campany Specific / domain specific chatbot

# %%
import re
import uuid
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm
import numpy as np

from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

import chromadb

from sentence_transformers import CrossEncoder

from langchain.llms import LlamaCpp

# %%
# Initializing ChomaDB  - store the path to your ChromaDB persist directory
CHROMA_PERSIST_DIR = "./chroma_db"


# %%
# Embedding model name  - will download automatically if not present
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# %%
# reranker (Cross-Encoder)
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# %%
# LOCAL LLaMA model path
LLAMA_MODEL_PATH = (
    r"C:\Users\User\Desktop\Mint HRM\company_chatbot\model\llama-2-7b-chat.Q3_K_M.gguf"
)

# %%
# Chunking parameters..
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# %%
# Retrieval / rerank parameters
TOP_K_RETRIEVE = 8
TOP_K_RERANK = 3

# %%
# LLaMA generation parameters  (pass to LlamaCpp)
LLAMA_MAX_TOKENS = 512
LLAMA_N_CTX = 2048
LLAMA_N_THREADS = 8
LLAMA_N_GPU_LAYERS = 20

# %%
# Initializing chromadb client  ( local persistent on disk)

client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
COLLECTION_NAME = "company_documents"

# Create or get Collection

try:
    chroma_collection = client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' already exists. Using existing collection.")
except Exception:
    chroma_collection = client.create_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' created.")

# %% [markdown]
# Data cleaning

# %%
# Identify duplicate chunks of text.


def compute_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# %%
# Creates a unique-ish ID for a file.


def make_file_id(file_path: str) -> str:
    st = Path(file_path).stat()
    base = f"{Path(file_path).name}-{st.st_size}-{st.st_mtime}"
    return hashlib.sha1(base.encode()).hexdigest()


# %%
# Cleans text before feeding it into embeddings.
def clean_text_basic(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u0000-\u001F]+", " ", text)  # remove control characters
    return text.strip()


# %% [markdown]
# Document Loader

# %%
# function to load documents from various file types


def load_file_to_documents(file_path: str):
    p = Path(file_path)
    suffix = p.suffix.lower()

    if suffix == ".txt":
        loader = TextLoader(str(p), encoding="utf-8")

    elif suffix == ".pdf":
        loader = PyPDFLoader(str(p))

    elif suffix in [".doc", ".docx"]:
        loader = UnstructuredWordDocumentLoader(str(p))

    else:
        loader = TextLoader(str(p), encoding="utf-8")

    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from {file_path}")
    return docs


# %% [markdown]
# Chunking Process

# %%
# chunking function

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


def chunk_docs_with_md5(docs: List[Any]) -> List[Dict[str, Any]]:
    chunks = []

    # Processing each document
    for d in docs:
        content = clean_text_basic(d.page_content)
        if not content:
            continue
        pieces = text_splitter.split_text(content)

        #  Creating chunk dictionaries
        for idx, part in enumerate(pieces):
            md5 = compute_md5(part)
            chunks.append(
                {
                    "text": part,
                    "md5": md5,
                    "source_meta": d.metadata if hasattr(d, "metadata") else {},
                }
            )

    return chunks


# %% [markdown]
# Loading a sentence embedding model and using it to convert text into vector embeddings for  RAG pipeline.

# %%
# --- Load embedding model (SentenceTransformer) ---

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(
    "Embedding model ready. Vector dimension:",
    embedding_model.get_sentence_embedding_dimension(),
)


def embed_text_list(texts: List[str]) -> np.array:
    # """Return numpy array of embeddings for the list of texts."""
    return embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


# %% [markdown]
# checking for duplicates in your Chroma vector database before inserting new embeddings ---- Deduplication

# %%
# Check existing chunk by md5 in Chroma (deduplication)


def chunk_md5_exists_in_chroma(md5_value: str) -> bool:
    #  Try to find if any metadata with md5 == md5_value exists.
    try:
        res = chroma_collection.get(
            where={"md5": md5_value}, include=["metadatas", "ids"]
        )
        return len(res.get("ids", [])) > 0
    except Exception:
        return False


# %% [markdown]
# full ingestion function—it takes a file and processes it all the way from loading → chunking → deduplication → embedding → storing in Chroma.

# %%
# ingestion function


def ingest_file_to_chroma(file_path: str, replace_existing: bool = False):
    """
    Ingest file_path into chroma collection.
    - replace_existing: if True, we delete all chunks with same filename first.
    """

    file_path = str(file_path)
    filename = Path(file_path).name
    file_id = make_file_id(file_path)
    uploaded_on = datetime.utcnow().isoformat()

    print(f"Processing file: {file_path} (id: {file_id})")

    # Delect previous entries for same filename -- for updating files.

    if replace_existing:
        try:
            chroma_collection.delete(where={"filename": filename})
            print(f"Deleted previous entries for filename: {filename}")
        except Exception:
            pass

    # Load files -> get the list of dicument object

    print(f"Loading file: {file_path}")
    docs = load_file_to_documents(file_path)
    if not docs:
        print(f"No documents found in file: {file_path}")
        return

    # Chuncking + Computes MD5 for deduplication

    chunks = chunk_docs_with_md5(docs)
    print(f"Chunks produced: {len(chunks)}")

    text_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    for idx, c in enumerate(chunks):
        md5_val = c["md5"]
        if chunk_md5_exists_in_chroma(md5_val):
            print(f"Skipping duplicate chunk {idx + 1}/{len(chunks)} (md5: {md5_val})")
            continue
        chunk_id = str(uuid.uuid4())
        text_to_add.append(c["text"])
        metadatas_to_add.append(
            {
                "file_id": file_id,
                "filename": filename,
                "chunk_index": idx,
                "md5": md5_val,
                "uploaded_on": uploaded_on,
                "source_meta": str(c.get("source_meta", {})),
            }
        )
        ids_to_add.append(chunk_id)

    if not text_to_add:
        print("No new (non-duplicate) chunks to add.")
        return

    # Compute embeddings and add to Chroma
    print("Computing embeddings for new chunks...")
    embedding_np = embed_text_list(text_to_add)
    embedding_list = embedding_np.tolist()

    print(f"Adding {len(text_to_add)} new chunks to ChromaDB...")
    chroma_collection.add(
        ids=ids_to_add,
        documents=text_to_add,
        metadatas=metadatas_to_add,
        embeddings=embedding_list,
    )

    print("Ingestion complete for file:", file_path)


# %% [markdown]
# functions to delete or update documents in Chroma

# %%
# functions to delete documents in Chroma


def delete_by_file_id(file_id: str):
    # Delete all chunks whose metadata file_id equals given file_id.

    try:
        chroma_collection.delete(where={"file_id": file_id})
        print(f"Deleted documents with file_id: {file_id}")
    except Exception as e:
        print(f"Error deleting documents with file_id {file_id}: {e}")


def delete_by_filename(filename: str):
    # Delete all chunks for a given filename (useful to update a file by same name).

    try:
        chroma_collection.delete(where={"filename": filename})
        print(f"Deleted documents with filename: {filename}")
    except Exception as e:
        print(f"Error deleting documents with filename {filename}: {e}")


def update_file_by_reupload(file_path: str):
    # Delete any previous chunks with same filename, then ingest file.
    # Use when you replace a file but keep same filename.

    filename = Path(file_path).name
    delete_by_filename(filename)
    ingest_file_to_chroma(file_path, replace_existing=False)


# %% [markdown]
# This block is about retrieving relevant text chunks from Chroma based on a query. It’s the search part of a RAG pipeline.

# %%
# Retrieval (Chroma query)


def retrieve_candidates(
    query: str, top_k: int = TOP_K_RETRIEVE, metadata_filter: Dict[str, Any] = None
):
    if metadata_filter:
        res = chroma_collection.query(
            query_texts=[query], n_results=top_k, where=metadata_filter
        )
    else:
        res = chroma_collection.query(query_texts=[query], n_results=top_k)

    documents = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]

    results = []
    for doc_text, meta, _id in zip(documents, metadatas, ids):
        results.append({" document": doc_text, "metadata": meta, "id": _id})
    return results


# %% [markdown]
# This block is about optionally re-ranking your retrieved candidates using a cross-encoder, which can improve the relevance of results at the cost of extra computation.

# %%
# Reranker initialization

reranker = None

if RERANKER_MODEL_NAME and CrossEncoder is not None:
    try:
        print("Loading reranker (Cross-Encoder) model...")
        reranker = CrossEncoder(RERANKER_MODEL_NAME)
        print("Reranker loaded.   ")

    except Exception as e:
        print("Error loading reranker model:", e)
        reranker = None

else:
    print("Reranker Disabled or CrossEncoder not available.")


def rerank_candidates(
    query: str, candidates: List[Dict[str, Any]], top_k: int = TOP_K_RERANK
):
    if reranker is None:
        return candidates[:top_k]

    pairs = [(query, c[" document"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:top_k]]


# %% [markdown]
# This block is about loading a local LLaMA model using llama-cpp-python, which lets you run the model entirely on your machine

# %%
# --- Initialize local LLaMA via llama-cpp-python  ---
print("Initializing local LLaMA model via llama-cpp-python...")

# Check that the model file exists
if not Path(LLAMA_MODEL_PATH).exists():
    raise FileNotFoundError(
        f"LLaMA model file not found at {LLAMA_MODEL_PATH}. Please check the path."
    )


llm = LlamaCpp(
    model_path=str(LLAMA_MODEL_PATH),
    n_ctx=LLAMA_N_CTX,
    n_threads=LLAMA_N_THREADS,
    n_gpu_layers=LLAMA_N_GPU_LAYERS,
    verbose=False,
)


print("LLaMA model loaded.")


# %% [markdown]
# uses the local LLaMA model to generate an answer based on the retrieved context chunks

# %%
# --- Generate answer with LLaMA using retrieved context ---


def generate_grounded_answer(
    user_query: str, contexts: List[Dict[str, Any]], max_tokens: int = LLAMA_MAX_TOKENS
):
    # Build a prompt that contains retrieved contexts and ask the local LLaMA model to answer using ONLY that context.

    ctx_texts = []

    for c in contexts:
        fname = c["metadata"].get("filename", "unknown")
        idx = c["metadata"].get("chunk_index", -1)

        chunk_text = c[" document"]
        ctx_texts.append(f"[{fname} - chunk {idx}]: {chunk_text}")

    context_block = "\n\n".join(ctx_texts)

    prompt = (
        "You are a helpful and polite assistant. Read the context below carefully and answer the question based ONLY on that information. "
        "If the answer is not contained in the context, kindly respond that you do not know, without making assumptions.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {user_query}\n\n"
        "ANSWER (use the context, be clear and concise, and be polite if unsure):"
    )

    # call local LLaMA model to generate answer
    resp = llm(prompt, max_tokens=max_tokens)
    return resp


# %% [markdown]
# This block is the final “all-in-one” function that ties together retrieval, reranking, and answer generation. It’s what you would call for any user question in your RAG pipeline.

# %%
# --- End-to-end query function: retrieve -> rerank -> generate ---


def answer_user_query(
    user_query: str,
    restrict_metadata: Dict[str, Any] = None,
    retrieve_k: int = TOP_K_RETRIEVE,
    user_rerank: bool = True,
):
    # 1 -- retrieve candidate
    candidates = retrieve_candidates(
        user_query, top_k=retrieve_k, metadata_filter=restrict_metadata
    )
    if not candidates:
        return "No relevant documents found in the database."

    # 2 -- rerank
    if user_rerank:
        top_candidates = rerank_candidates(
            user_query, candidates, top_k=min(TOP_K_RERANK, len(candidates))
        )
    else:
        top_candidates = candidates[: min(TOP_K_RERANK, len(candidates))]

    # 3 -- generate final answer
    answer_text = generate_grounded_answer(user_query, top_candidates)
    return answer_text


# %%

"""MAIN INTERACTIVE MENU
Choose whether to upload documents first or directly start chatting.
"""


def run_menu():
    print("\n--- Welcome to Your RAG Assistant ---")
    print("Options:")
    print("1) Upload documents into vector DB")
    print("2) Directly chat with AI assistant")
    choice = input("Enter your choice (1/2): ").strip()
    print(f"\nYou selected option {choice}.")

    if choice == "1":
        print("\n Please place your documents inside the 'DATA/' folder.")
        UPLOAD_DIR = "./DATA"
        Path(UPLOAD_DIR).mkdir(exist_ok=True)

        files = list(Path(UPLOAD_DIR).glob("*"))
        if not files:
            print(" No files found in DATA/. Add some documents and run again.")
        else:
            for f in tqdm(files, desc="Ingesting files"):
                ingest_file_to_chroma(str(f), replace_existing=False)
            print("\n All files processed and stored in ChromaDB.")

        # After ingestion, switch to chat
        start_chat_loop()

    elif choice == "2":
        print("\n Starting chat without uploading documents...")
        start_chat_loop()
    else:
        print(" Invalid choice. Please restart and enter 1 or 2.")


def start_chat_loop():
    print("\n--- Chat Mode ---")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        answer = answer_user_query(query)
        print(f"\n AI: {answer}\n")


if __name__ == "__main__":
    run_menu()

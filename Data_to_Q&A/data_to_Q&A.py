# data_prep.py
# Auto version using llama-cpp-python (no main.exe needed)

import os
import glob
import json
import time
from typing import List

# llama-cpp-python
from llama_cpp import Llama

# Optional imports for PDF/DOCX extraction
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx
except Exception:
    docx = None


# -------------------------
# Configuration / Defaults
# -------------------------
DEFAULT_INSTRUCTION = "Convert the following text into clear and concise Q&A pairs suitable for training a chatbot."
DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 200
DEFAULT_OUTFILE = "output_qa.jsonl"
DEFAULT_NUM_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_THREADS = 4

# --- YOUR FIXED VALUES ---
MODEL_PATH = (
    r"C:\Users\User\Desktop\Mint HRM\company_chatbot\model\llama-2-7b-chat.Q3_K_M.gguf"
)
FILES = [r"C:\Users\User\Desktop\Mint HRM\company_chatbot\DATA\PRO_032844.pdf"]
OUTPUT_PATH = DEFAULT_OUTFILE
DOMAIN_HINT = "MEDICAL"
# -------------------------


# -------------------------
# File extraction helpers
# -------------------------
def extract_text_from_pdf(path: str):
    if not fitz:
        raise RuntimeError("PyMuPDF not installed. Install with: pip install pymupdf")
    doc = fitz.open(path)
    for page in doc:
        yield page.get_text()
    doc.close()


def extract_text_from_docx(path: str) -> str:
    if not docx:
        raise RuntimeError(
            "python-docx not installed. Install with: pip install python-docx"
        )
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# -------------------------
# Chunking
# -------------------------
def chunk_text(text_iter, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP):
    buffer = ""
    for piece in text_iter:
        buffer += piece + "\n"
        while len(buffer) >= max_chars:
            chunk = buffer[:max_chars].strip()
            yield chunk
            buffer = buffer[max_chars - overlap :]
    if buffer.strip():
        yield buffer.strip()


# -------------------------
# llama-cpp-python wrapper
# -------------------------
def call_llama(model: Llama, prompt: str, n_predict: int, temp: float) -> str:
    output = model(
        prompt,
        max_tokens=n_predict,
        temperature=temp,
        stop=["</s>"],
    )
    return output["choices"][0]["text"].strip()


# -------------------------
# Prompt formatting
# -------------------------
def build_prompt(instruction: str, chunk: str, domain_hint: str = None) -> str:
    schema_note = (
        "Return the answer as a JSON array of objects. Each object must have these keys: "
        '"question" (string), "answer" (string). Example: '
        '[{"question": "Q1?", "answer": "A1."}, {"question": "Q2?", "answer": "A2."}]'
        " Keep answers concise but complete."
    )
    prompt_parts = [instruction, schema_note]
    if domain_hint:
        prompt_parts.append(
            f"Document domain / hint: {domain_hint}. Use domain-appropriate terminology."
        )
    prompt_parts.append("\n---\n")
    prompt_parts.append(chunk)
    return "\n\n".join(prompt_parts)


# -------------------------
# Parse model output
# -------------------------
def parse_json_from_model(output: str) -> List[dict]:
    first_bracket = output.find("[")
    last_bracket = output.rfind("]")
    if first_bracket == -1 or last_bracket == -1 or last_bracket <= first_bracket:
        return []
    json_text = output[first_bracket : last_bracket + 1]
    try:
        data = json.loads(json_text)
        if isinstance(data, list):
            cleaned = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                q = item.get("question") or item.get("q") or item.get("Question")
                a = item.get("answer") or item.get("a") or item.get("Answer")
                if q and a:
                    cleaned.append(
                        {"question": str(q).strip(), "answer": str(a).strip()}
                    )
            return cleaned
    except Exception:
        return []
    return []


# -------------------------
# High-level flow
# -------------------------
def process_files(
    model_path: str,
    file_paths: List[str],
    output_path: str,
    instruction: str = DEFAULT_INSTRUCTION,
    domain_hint: str = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    n_predict: int = DEFAULT_NUM_TOKENS,
    temp: float = DEFAULT_TEMPERATURE,
    threads: int = DEFAULT_THREADS,
):
    # Load model once
    print(f"[info] Loading model from {model_path} ...")
    model = Llama(
        model_path=model_path,
        n_threads=threads,
        n_ctx=2048,
        embedding=False,
    )

    all_qas = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"[warn] file not found: {path}. Skipping.")
            continue
        ext = os.path.splitext(path)[1].lower()
        print(f"[info] Reading {path} ...")
        try:
            if ext == ".pdf":
                text_iter = extract_text_from_pdf(path)
                chunks = list(
                    chunk_text(text_iter, max_chars=max_chars, overlap=overlap)
                )
            elif ext in [".docx", ".doc"]:
                text = extract_text_from_docx(path)
                chunks = list(chunk_text([text], max_chars=max_chars, overlap=overlap))
            else:
                text = extract_text_from_txt(path)
                chunks = list(chunk_text([text], max_chars=max_chars, overlap=overlap))
        except Exception as e:
            print(f"[error] Failed to extract text from {path}: {e}")
            continue

        print(f"[info] {len(chunks)} chunks created from {path}")

        for i, chunk in enumerate(chunks, 1):
            prompt = build_prompt(instruction, chunk, domain_hint=domain_hint)
            try:
                raw_output = call_llama(model, prompt, n_predict=n_predict, temp=temp)
            except Exception as e:
                print(f"[error] Llama call failed on chunk {i}: {e}")
                continue

            qas = parse_json_from_model(raw_output)
            if not qas:
                print(f"[warn] could not parse JSON for chunk {i}. Using fallback.")
                fallback = {
                    "question": f"Content chunk {i} summary/question",
                    "answer": raw_output.strip(),
                }
                all_qas.append(fallback)
            else:
                for qa in qas:
                    qa["_source_file"] = os.path.basename(path)
                    qa["_chunk_index"] = i
                    all_qas.append(qa)

            time.sleep(0.2)

    print(f"[info] Writing {len(all_qas)} Q&A items to {output_path}")
    with open(output_path, "w", encoding="utf-8") as outf:
        for item in all_qas:
            outf.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("[done] Saved output.")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    process_files(
        model_path=MODEL_PATH,
        file_paths=FILES,
        output_path=OUTPUT_PATH,
        instruction=DEFAULT_INSTRUCTION,
        domain_hint=DOMAIN_HINT,
        max_chars=DEFAULT_MAX_CHARS,
        overlap=DEFAULT_OVERLAP,
        n_predict=DEFAULT_NUM_TOKENS,
        temp=DEFAULT_TEMPERATURE,
        threads=DEFAULT_THREADS,
    )

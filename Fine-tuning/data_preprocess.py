# data_prep.py
# Converts local PDFs / DOCX / plain text into instruction-format JSONL for fine-tuning.
# Usage examples:
#  python data_prep.py --pdfs ./docs/*.pdf --out train.jsonl --maxlen 512 --val_split 0.05

import glob
import json
import argparse
import os
import random

# Optional imports for PDF/DOCX extraction
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import docx
except Exception:
    docx = None

path = os.path.dirname(
    os.path.abspath(
        r"c:\Users\User\Desktop\Mint HRM\company_chatbot\DATA\PRO_032844.pdf"
    )
)


def extract_text_from_pdf(path):
    if not fitz:
        raise RuntimeError("PyMuPDF not installed. Install with: pip install pymupdf")
    doc = fitz.open(path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)


def extract_text_from_docx(path):
    if not docx:
        raise RuntimeError(
            "python-docx not installed. Install with: pip install python-docx"
        )
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def chunk_text(text, max_chars=1500, overlap=200):
    # naive chunker by characters
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = min(i + max_chars, L)
        chunks.append(text[i:end])
        i = end - overlap
        if i < 0:
            i = 0
    return chunks


def make_examples_from_text(text, instruction_template):
    # returns list of instruction dicts from a text string
    chunks = chunk_text(text, max_chars=1500, overlap=200)
    examples = []
    for c in chunks:
        # simple pattern: ask summary/question about chunk
        inst = instruction_template
        input_text = c.strip()
        output_text = ""  # leave blank if you plan to create labels manually later
        examples.append(
            {"instruction": inst, "input": input_text, "output": output_text}
        )
    return examples


def main(args):
    all_examples = []
    # collect PDF files
    for pdfpat in args.pdfs:
        for path in glob.glob(pdfpat):
            print("Processing PDF:", path)
            text = extract_text_from_pdf(path)
            ex = make_examples_from_text(text, args.instruction)
            all_examples.extend(ex)

    # docx
    for docpat in args.docx:
        for path in glob.glob(docpat):
            print("Processing DOCX:", path)
            text = extract_text_from_docx(path)
            ex = make_examples_from_text(text, args.instruction)
            all_examples.extend(ex)

    # plaintext files
    for txtpat in args.txts:
        for path in glob.glob(txtpat):
            print("Processing TXT:", path)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            ex = make_examples_from_text(text, args.instruction)
            all_examples.extend(ex)

    # optional: shuffle and split
    random.shuffle(all_examples)
    n_val = int(len(all_examples) * args.val_split)
    val = all_examples[:n_val]
    train = all_examples[n_val:]

    print(f"Total examples: {len(all_examples)} -> train: {len(train)} val: {len(val)}")

    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    write_jsonl(args.out_train, train)
    write_jsonl(args.out_val, val)
    print("Wrote:", args.out_train, args.out_val)


if __name__ == "__main__":
    print("=== Data Prep for Fine-Tuning ===")

    pdf_input = input(
        "Enter PDF file paths (comma-separated, or leave blank if none): "
    ).strip()
    docx_input = input(
        "Enter DOCX file paths (comma-separated, or leave blank if none): "
    ).strip()
    txt_input = input(
        "Enter TXT file paths (comma-separated, or leave blank if none): "
    ).strip()

    pdfs = [p.strip() for p in pdf_input.split(",")] if pdf_input else []
    docx = [p.strip() for p in docx_input.split(",")] if docx_input else []
    txts = [p.strip() for p in txt_input.split(",")] if txt_input else []

    out_train = (
        input("Enter output training JSONL filename (default: train.jsonl): ").strip()
        or "train.jsonl"
    )
    out_val = (
        input("Enter output validation JSONL filename (default: val.jsonl): ").strip()
        or "val.jsonl"
    )
    val_split = input("Enter validation split ratio (default: 0.05): ").strip()
    val_split = float(val_split) if val_split else 0.05

    instruction = input(
        "Enter instruction prompt for the model (or leave blank for default): "
    ).strip()
    if not instruction:
        instruction = "You are an HR assistant. Convert the following text into clear and concise Q&A pairs suitable for training a chatbot. "

    class Args:
        pass

    args = Args()
    args.pdfs = pdfs
    args.docx = docx
    args.txts = txts
    args.out_train = out_train
    args.out_val = out_val
    args.val_split = val_split
    args.instruction = instruction

    main(args)

import json
import pdfplumber
import textwrap
from pathlib import Path

# ---------------------------
# CONFIG
# ---------------------------
PDF_PATH = "Chapters/CH11.pdf"   # change this to your file
OUTPUT_PATH = "pretrain/chapter11_pretrain.jsonl"
CHUNK_SIZE = 800  # approx characters per chunk (adjust as needed)

# ---------------------------
# Extract text from PDF
# ---------------------------
def extract_pdf_text(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text.strip())
    return "\n".join(text)

# ---------------------------
# Chunk text into manageable pieces
# ---------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks = textwrap.wrap(text, chunk_size, replace_whitespace=False)
    return chunks

# ---------------------------
# Save dataset in JSONL format
# ---------------------------
def save_jsonl(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            sample = {
                "text": f"<bos>\n{chunk}\n<eos>"
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    Path(OUTPUT_PATH).unlink(missing_ok=True)  # clear old file
    raw_text = extract_pdf_text(PDF_PATH)
    chunks = chunk_text(raw_text, CHUNK_SIZE)
    save_jsonl(chunks, OUTPUT_PATH)
    print(f"✅ Saved {len(chunks)} chunks to {OUTPUT_PATH}")
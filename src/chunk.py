import json
from pathlib import Path

MANUAL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "filings.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "chunked" / "chunked.json"

TARGET_SIZE = 350
OVERLAP_SIZE = 75

def build_overlap_seed(current_chunk, overlap_size):
    words = " ".join(current_chunk).split()
    seed_words = words[-overlap_size:]
    return [" ".join(seed_words)], len(seed_words)

def chunk_text(text, target_size, overlap_size) :
    paras = text.split("\n")
    chunks = []     # list of string
    current_chunk = []
    current_size = 0
    seed_only = False # true only when current_chunk == overlap 

    for para in paras:
        n = len(para.split())

        # big para > target_size
        if n > target_size:
            if current_chunk and not seed_only:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_size = build_overlap_seed(current_chunk, overlap_size) 

            words = para.split()
            while words:
                remain_size = target_size - current_size
                piece_words = words[:remain_size]
                current_chunk.append(" ".join(piece_words))
                chunks.append(" ".join(current_chunk))

                words = words[remain_size:]
                current_chunk, current_size = build_overlap_seed(current_chunk, overlap_size) 
                seed_only = True
            continue

        # small para, check if can fit with current_chunk
        if current_size + n > target_size and current_chunk:
            if not seed_only:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_size = build_overlap_seed(current_chunk, overlap_size)
            else:
                current_chunk, current_size = [], 0 # drop bare seed, start fresh
        
        current_chunk.append(para)
        current_size += n
        seed_only = False

    # final flush
    if current_chunk and not seed_only:
        chunks.append(" ".join(current_chunk))

    return chunks

def main():
    with open(MANUAL_PATH, "r", encoding="utf-8") as f:
        filings = json.load(f)
    
    # section = filings[0]
    # for i, c in enumerate(chunk_text(section["text"], TARGET_SIZE, OVERLAP_SIZE)):
    #     print(i, ":", c)

    chunks = []
    for section in filings:
        this_chunks = chunk_text(section["text"], TARGET_SIZE, OVERLAP_SIZE)

        ticker = section["ticker"]
        fiscal_year = section["fiscal_year"]
        item = section["item"]
        section_name = section["section_name"]

        for ind, chunk in enumerate(this_chunks):
            chunks.append({
                "chunk_id": f"{ticker}_{fiscal_year}_{item}_{ind}",
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "item": item,
                "section_name": section_name,
                "chunk_index": ind,
                "text": chunk,
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"\nSaved {len(chunks)} chunks to {OUT_PATH}")



if __name__ == "__main__":
    main()
    
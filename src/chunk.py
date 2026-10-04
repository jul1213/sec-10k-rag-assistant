import json
from pathlib import Path

MANUAL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "filings.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "chunked" / "chunked.json"

TARGET_SIZE = 350
OVERLAP_SIZE = 75

def build_overlap_seed(current_chunk, overlap_size):
    seed = []
    seed_size = 0
    for chunk in reversed(current_chunk):
        if seed_size >= overlap_size:
            break
        seed.append(chunk)
        seed_size += len(chunk.split())
    seed.reverse()

    return seed, seed_size

def chunk_text(text, target_size, overlap_size) :
    paras = text.split("\n")
    chunks = []     # list of string
    current_chunk = []
    current_size = 0

    for para in paras:
        if len(para.split()) > target_size:
            # flush chunk from before
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_size = build_overlap_seed(current_chunk, overlap_size) 

            words = para.split()
            while words:
                remain_size = target_size - current_size
                if remain_size <= 0:
                    current_chunk, current_size = [], 0
                    remain_size = target_size

                piece_words = words[:remain_size]
                current_chunk.append(" ".join(piece_words))
                chunks.append(" ".join(current_chunk))

                words = words[remain_size:]
                seed_words = piece_words[-overlap_size:]   # last N words of the piece just made
                current_chunk = [" ".join(seed_words)]
                current_size = len(seed_words)
            continue

        if current_size + len(para.split()) > target_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_size = build_overlap_seed(current_chunk, overlap_size)
        
        current_chunk.append(para)
        current_size += len(para.split())

    # final flush
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def main():
    with open(MANUAL_PATH, "r") as f:
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
    
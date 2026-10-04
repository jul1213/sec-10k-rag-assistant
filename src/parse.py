import json
import re
from pathlib import Path
from sec_parser import Edgar10KParser, TreeBuilder

from common import section_label, get_sections

MANUAL_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "manual"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "filings.json"

FILENAME_RE = re.compile(r"^([A-Z]+)_(\d{4})\.html?$", re.IGNORECASE)


def main():
    if not MANUAL_DIR.exists() or not any(MANUAL_DIR.iterdir()):
        print(f"No files found in {MANUAL_DIR}.")
        return

    records = []

    for html_path in sorted(MANUAL_DIR.glob("*.html*")):
        m = FILENAME_RE.match(html_path.name)
        if not m:
            print(f"  [!] skipping {html_path.name} — expected TICKER_YEAR.html, e.g. NVDA_2026.htm")
            continue

        ticker, fiscal_year = m.group(1).upper(), m.group(2)

        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        elements = Edgar10KParser().parse(html)
        tree = TreeBuilder().build(elements)
        sections = get_sections(tree)

        if not sections:
            print(f"  [!] no sections matched in {html_path.name} — "
                  f"open it and check it's the actual 10-K body, not a cover page or exhibit")

        for item_num, section_text in sections.items():
            records.append({
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "filed_date": None,   # not available in manual mode — fine for now
                "item": item_num,
                "section_name": section_label(item_num),
                "text": section_text,
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved {len(records)} sections to {OUT_PATH}")


if __name__ == "__main__":
    main()
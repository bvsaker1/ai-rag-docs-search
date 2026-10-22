import json, requests, time
from pathlib import Path

pdf_dir = Path("data/papers")
pdf_dir.mkdir(parents=True, exist_ok=True)

with open("data/pdf/arxiv/pdf_urls.json") as f:
    pdf_urls = json.load(f)

for paper_id, url in pdf_urls.items():
    dest = pdf_dir / f"{paper_id}.pdf"
    if dest.exists():
        continue
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    time.sleep(1)  # be polite to arXiv servers
    print(f"Downloaded {paper_id}")
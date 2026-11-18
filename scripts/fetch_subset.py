"""Fetch a small CGHD slice locally, for testing the vision stage."""
import sys
from huggingface_hub import snapshot_download

drafters = sys.argv[1:] or ["1"]
patterns = ["classes*.json"]
for d in drafters:
    patterns += [f"drafter_{d}/annotations/*", f"drafter_{d}/images/*"]
p = snapshot_download(
    repo_id="lowercaseonly/cghd", repo_type="dataset",
    local_dir="data/cghd", allow_patterns=patterns, max_workers=8,
)
print("DOWNLOADED_TO", p)

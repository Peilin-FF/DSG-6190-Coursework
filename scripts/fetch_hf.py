import sys
from huggingface_hub import hf_hub_download
repo, out = sys.argv[1], sys.argv[2]
for f in sys.argv[3:]:
    p = hf_hub_download(repo_id=repo, repo_type="dataset", filename=f, local_dir=out)
    print("done", p, flush=True)

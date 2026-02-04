import os
import json
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
import pyrallis
from dataclasses import dataclass
import nltk#
from nltk.corpus import wordnet
import ssl
import os

# 1. Fix SSL certificate error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 2. Force NLTK to use your workspace folder
nltk_data_dir = "/home/e12434694/rdt_workspace/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# 3. Download the resources
print("Downloading NLTK resources...")
nltk.download('wordnet', download_dir=nltk_data_dir, quiet=False)
nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=False)
@dataclass
class AuditConfig:
    bucket: str = "gresearch"
    prefix: str = "robotics/droid_raw/1.0.1/AUTOLab/success"
    audit_dir: Path = Path("/home/e12434694/rdt_workspace/data/audit_online")
    temp_cache: Path = Path("/home/e12434694/temp_cache")
    limit: int = -1 
    min_samples: int = 10  
    merge_similar: bool = True
    # Your core ManiSkill tasks
    target_tasks: str = "peg,insertion,cube,pick,stack,plug,charger,push"

def get_synonyms(word):
    """Returns a set of synonyms for a given word using WordNet."""
    synonyms = {word.lower()}
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' ').lower())
    return synonyms

def normalize_instruction(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def audit_online_dataset(cfg: AuditConfig):
    cfg.audit_dir.mkdir(parents=True, exist_ok=True)
    cfg.temp_cache.mkdir(parents=True, exist_ok=True)
    
    # --- THESAURUS EXPANSION ---
    base_keywords = [k.strip() for k in cfg.target_tasks.split(",")]
    expanded_keywords = set()
    for word in base_keywords:
        expanded_keywords.update(get_synonyms(word))
    
    print(f"Expanded {len(base_keywords)} tasks into {len(expanded_keywords)} similar words.")
    print(f"Examples: {list(expanded_keywords)[:10]}")

    client = storage.Client(project="anonymous", credentials=AnonymousCredentials())
    blobs = list(client.list_blobs(cfg.bucket, prefix=cfg.prefix))
    
    trials = {}
    for b in blobs:
        parts = b.name.split("/")
        try:
            tid = parts[parts.index("success") + 2]
        except (ValueError, IndexError): continue
        trials.setdefault(tid, {"metadata": None})
        if b.name.endswith(".json"):
            trials[tid]["metadata"] = b

    tids = sorted(trials.keys())
    if cfg.limit > 0: tids = tids[:cfg.limit]

    print(f"Processing {len(tids)} trials...")
    clean_counts = Counter()

    for tid in tqdm(tids, desc="Scanning Metadata"):
        if trials[tid]["metadata"]:
            local_meta = cfg.temp_cache / f"{tid.replace(':','_')}_meta.json"
            try:
                trials[tid]["metadata"].download_to_filename(str(local_meta))
                with open(local_meta, 'r') as f:
                    meta_data = json.load(f)
                    instr = normalize_instruction(meta_data.get("current_task", "unknown_task"))
                    
                    # Search for any expanded synonyms in the instruction
                    if any(syn in instr for syn in expanded_keywords):
                        clean_counts[instr] += 1
            except Exception as e:
                print(f"Error {tid}: {e}")
            finally:
                if local_meta.exists(): local_meta.unlink()

    # Final Sort and Save
    filtered_sorted = dict(clean_counts.most_common())
    filtered_sorted = {k: v for k, v in filtered_sorted.items() if v >= cfg.min_samples}
    
    output_file = cfg.audit_dir / "thesaurus_audit_report.json"
    with open(output_file, 'w') as f:
        json.dump({"results": filtered_sorted}, f, indent=4)

    print(f"Done! Saved report with {len(filtered_sorted)} relevant task types.")

if __name__ == "__main__":
    pyrallis.wrap()(audit_online_dataset)()
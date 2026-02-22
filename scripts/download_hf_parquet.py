#!/usr/bin/env python3
"""
Download all parquet shards of a Hugging Face dataset via datasets-server.

Example:
  python3 scripts/download_hf_parquet.py \\
    --dataset dmitva/human_ai_generated_text \\
    --out-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List


def fetch_json(url: str) -> Dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def sanitize_name(name: str) -> str:
    return name.replace("/", "__")


def ensure_download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HF dataset id, e.g. org/name")
    parser.add_argument("--out-dir", default="data/raw", help="Base output directory")
    args = parser.parse_args()

    ds = args.dataset.strip()
    enc = urllib.parse.quote(ds, safe="")
    parquet_api = f"https://datasets-server.huggingface.co/parquet?dataset={enc}"
    splits_api = f"https://datasets-server.huggingface.co/splits?dataset={enc}"

    print(f"[info] dataset={ds}")
    parquet_info = fetch_json(parquet_api)
    splits_info = fetch_json(splits_api)

    files: List[Dict] = parquet_info.get("parquet_files") or []
    if not files:
        print("[error] no parquet files found")
        return 1

    dataset_dir = Path(args.out_dir) / sanitize_name(ds)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    manifest_files = []
    for idx, f in enumerate(files):
        url = f["url"]
        filename = f.get("filename") or os.path.basename(urllib.parse.urlparse(url).path)
        out_name = f"{idx:04d}__{filename}"
        out_path = dataset_dir / out_name
        ensure_download(url, out_path)
        size = out_path.stat().st_size
        total_bytes += size
        manifest_files.append(
            {
                "index": idx,
                "config": f.get("config"),
                "split": f.get("split"),
                "source_url": url,
                "source_filename": f.get("filename"),
                "downloaded_path": str(out_path),
                "size_bytes": size,
            }
        )
        print(f"[ok] {idx+1}/{len(files)} {out_name} ({size} bytes)")

    manifest = {
        "dataset": ds,
        "splits": splits_info.get("splits", []),
        "files": manifest_files,
        "total_files": len(manifest_files),
        "total_bytes": total_bytes,
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] manifest={manifest_path} total_bytes={total_bytes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

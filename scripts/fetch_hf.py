#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download or inspect RaceShift Hugging Face sources.")
    parser.add_argument("repo_id")
    parser.add_argument("--output", default="data/raw/huggingface")
    parser.add_argument("--list", action="store_true", help="Only list repository files.")
    parser.add_argument("--allow", action="append", default=[], help="Optional allow pattern, repeatable.")
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    files = api.list_repo_files(args.repo_id, repo_type="dataset")
    print(f"{args.repo_id}: {len(files)} files")
    for f in files:
        print(f)

    if args.list:
        return

    output = Path(args.output) / args.repo_id.replace("/", "__")
    output.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output),
        allow_patterns=args.allow or None,
    )
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()

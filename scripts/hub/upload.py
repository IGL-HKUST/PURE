"""Explicitly upload an already converted model folder to a Hugging Face model repo."""

import argparse
from pathlib import Path
import json
from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument(
        "--public",
        action="store_true",
        help="Create a new public repo; default is private",
    )
    args = p.parse_args()
    configs = list(args.folder.glob("config.json")) + list(
        args.folder.glob("*/config.json")
    )
    if not configs:
        raise ValueError("No converted model config found")
    for config in configs:
        if json.loads(config.read_text()).get("model_type") != "pure":
            raise ValueError("Not a PURE model directory")
        if not list(config.parent.glob("model*.safetensors")):
            raise ValueError("Safetensors files are missing")
    api = HfApi()  # Uses the user's normal HF login / HF_TOKEN; never logs the token.
    api.create_repo(
        args.repo_id, repo_type="model", private=not args.public, exist_ok=True
    )
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.folder,
        commit_message="Publish PURE HF model directories",
        ignore_patterns=["*.bin", "*.pt", "*.pth"],
    )
    print(result)


if __name__ == "__main__":
    main()

"""Upload the assembled model folder (default ``<repo>/../project_hf/<model name>``) to the Hugging Face Hub.

    python -m model.upload --repo-id <user>/kenlang-gemma-4-12b-it-lora-v2        # dry run: lists the files
    python -m model.upload --repo-id <user>/kenlang-gemma-4-12b-it-lora-v2 --yes  # create the repo, upload

Needs ``pip install huggingface_hub`` and a login (``hf auth login`` or the ``HF_TOKEN`` variable).
The repository is created **private** unless you pass ``--public``; publishing is a decision to take
on the Hub page once you have read the model card.
"""
import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent.parent / "project_hf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--folder", default="", help="default: <DEFAULT_ROOT>/<model name>, as built by assemble")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--yes", action="store_true", help="really upload (default is a dry run)")
    a = ap.parse_args()

    folder = pathlib.Path(a.folder) if a.folder else DEFAULT_ROOT / a.repo_id.split("/")[-1]
    if not (folder / "README.md").exists():
        sys.exit(f"{folder} has no README.md; run `python -m model.assemble` first")
    if "<your-username>" in a.repo_id or "<your-username>" in (folder / "README.md").read_text(encoding="utf-8"):
        sys.exit("replace <your-username> in --repo-id, and re-run `python -m model.assemble` with the real "
                 "--repo-id/--repo-url so the card points to the right places")
    files = sorted(f for f in folder.rglob("*") if f.is_file())
    print(f"{'Uploading' if a.yes else 'Would upload'} {len(files)} files to {a.repo_id} "
          f"({'public' if a.public else 'private'}):")
    for f in files:
        print(f"  {f.relative_to(folder)}  {f.stat().st_size / 1e6:.2f} MB")
    if not a.yes:
        print("dry run; add --yes to upload")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo_id, repo_type="model", private=not a.public, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=a.repo_id, repo_type="model",
                      commit_message="init")
    print(f"done: https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()

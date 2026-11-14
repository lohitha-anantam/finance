import os
from pathlib import Path
import kagglehub

def download_kaggle_dataset(dataset_slug: str, destination: str = "data/kaggle") -> str:
    """Download a public or authorized Kaggle dataset once and copy it into destination."""
    if not dataset_slug or "/" not in dataset_slug:
        raise ValueError("Dataset slug must look like owner/dataset-name")
    token = os.getenv("KAGGLE_API_TOKEN")
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    downloaded = Path(kagglehub.dataset_download(dataset_slug))
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    # Copy only when target is empty; preserve KaggleHub cache otherwise.
    import shutil
    if not any(dest.iterdir()):
        for item in downloaded.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    return str(dest.resolve())

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Kaggle slug, e.g. owner/dataset-name")
    p.add_argument("--out", default="data/kaggle")
    a = p.parse_args()
    print(download_kaggle_dataset(a.dataset, a.out))

"""
Data Pipeline Module.

Responsible for downloading and extracting the dataset from Kaggle.
"""
import os
import subprocess
import zipfile
from pathlib import Path

from ex3.shared.config import cfg


def download_dataset() -> None:
    """Download the dataset using Kaggle API if it doesn't exist."""
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    summary_path = data_dir / "program_summary.csv"
    detailed_path = data_dir / "programs_detailed_boostcamp_kaggle.csv"

    if summary_path.exists() and detailed_path.exists():
        print("Dataset already downloaded.")
        return

    print(f"Downloading dataset {cfg.kaggle_dataset}...")
    try:
        # We use subprocess to call Kaggle CLI since the Kaggle API expects keys in a specific way
        # or via kaggle.json. Since we use dotenv, we assume environment variables are set.
        subprocess.run(
            ["uv", "run", "kaggle", "datasets", "download", "-d", cfg.kaggle_dataset, "-p", str(data_dir)],
            check=True
        )

        zip_path = data_dir / f"{cfg.kaggle_dataset.split('/')[-1]}.zip"
        if zip_path.exists():
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_path)
            print("Extraction complete.")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download Kaggle dataset: {e}") from e

if __name__ == "__main__":
    download_dataset()

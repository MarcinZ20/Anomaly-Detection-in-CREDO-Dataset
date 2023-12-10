import os
import gdown
import shutil
from zipfile import ZipFile
from pathlib import Path

URLS = ["https://drive.google.com/file/d/1vCoe9oN44j--WCI7VxZZkA7wmbot6DH0/view?usp=share_link",
        "https://drive.google.com/file/d/1zgTGeHSLX9Za2s_vjLT2NeiWvVw6NDuv/view?usp=share_link",
        "https://drive.google.com/file/d/1SOvH9ROisBqTg0Y2tJUtkvS1ba4br1ZX/view?usp=share_link",
        "https://drive.google.com/file/d/1pRM9TIirjjhQHV9y9U0H-ZVZE_Oj07Gk/view?usp=share_link",
        "https://drive.google.com/file/d/1vqpv_kB0YdCUbDtnqUXE_pqHELcEIoNV/view?usp=share_link"
        ]

FILE_NAME = "cred_data_file"
RAW_DATA_PATH = r'../../data/raw'


def download_file(file_url: str, destination: str, extension: str, in_place=False) -> Path:

    if in_place:
        destination = Path('./default' + extension)

    gdown.download(url=file_url, output=destination, quiet=False, fuzzy=True)

    return destination


def extract_from_zip(file_path: str) -> None:
    with ZipFile(Path(file_path), 'r') as f:
        f.extractall(path=RAW_DATA_PATH)


# Works, but slowly - if you prefer to keep files in one place, use it before clear_zip_archives() below
def merge_to_one_folder() -> None:
    sub_dirs = [f for f in os.listdir(RAW_DATA_PATH) if os.path.isdir(f"{RAW_DATA_PATH}/{f}")]

    for sub_dir in sub_dirs:
        for file in os.listdir(fr'{RAW_DATA_PATH}/{sub_dir}'):
            shutil.move(fr'{RAW_DATA_PATH}/{sub_dir}/{file}', fr'{RAW_DATA_PATH}/{file}')
        Path.rmdir(Path(fr'{RAW_DATA_PATH}/{sub_dir}'))
        print(f'Deleted directory {sub_dir}')


def clean_zip_archives() -> None:
    for file in os.listdir(RAW_DATA_PATH):
        if os.path.splitext(file)[1] == ".zip":
            Path.unlink(Path(f"{RAW_DATA_PATH}/{file}"))


if __name__ == "__main__":
    for index, url in enumerate(URLS):
        download_file(url, fr"{RAW_DATA_PATH}/{FILE_NAME}_{index}.zip", ".zip")
        extract_from_zip(f"{RAW_DATA_PATH}/{FILE_NAME}_{index}.zip")

    # After unpacking, clean .zip archives
    clean_zip_archives()


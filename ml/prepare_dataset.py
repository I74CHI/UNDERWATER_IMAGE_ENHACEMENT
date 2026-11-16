import argparse
import random
import shutil
import zipfile
from pathlib import Path, PurePosixPath

DATA_DIR = Path(__file__).parent / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_PAIRS = 890
SEED = 42


def _file_name(file):
    return file.filename if isinstance(file, zipfile.ZipInfo) else file.name


def _index_files(files):
    indexed = {}
    for file in files:
        name = _file_name(file)
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = Path(name).stem
        if stem in indexed:
            raise ValueError(f"Duplicate image name: {stem}")
        indexed[stem] = file
    return indexed


def _zip_files(archive, folder):
    files = (
        info
        for info in archive.infolist()
        if len(PurePosixPath(info.filename).parts) >= 2
        and PurePosixPath(info.filename).parts[-2].lower() == folder
        and not info.is_dir()
    )
    return _index_files(files)


def _copy(source, destination, archive):
    if archive is None:
        shutil.copy2(source, destination)
    else:
        with archive.open(source) as reader, destination.open("wb") as writer:
            shutil.copyfileobj(reader, writer)


def _organize(raw_files, reference_files, archive):
    stems = sorted(raw_files.keys() & reference_files.keys())
    if len(stems) != EXPECTED_PAIRS:
        raise ValueError(f"Expected 890 matched UIEB pairs; found {len(stems)}")

    pairs = [(raw_files[stem], reference_files[stem]) for stem in stems]
    random.Random(SEED).shuffle(pairs)
    train_end = int(len(pairs) * 0.70)
    val_end = train_end + int(len(pairs) * 0.15)
    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    for split, split_pairs in splits.items():
        for kind in ("raw", "reference"):
            (DATA_DIR / split / kind).mkdir(parents=True, exist_ok=True)
        for raw, reference in split_pairs:
            _copy(raw, DATA_DIR / split / "raw" / Path(_file_name(raw)).name, archive)
            _copy(
                reference,
                DATA_DIR / split / "reference"
                / Path(_file_name(reference)).name,
                archive,
            )

    counts = ", ".join(f"{key}={len(value)}" for key, value in splits.items())
    print(f"Prepared UIEB pairs: {counts}")


def main():
    parser = argparse.ArgumentParser(
        description="Split local UIEB raw-890/reference-890 images into train, val, and test."
    )
    parser.add_argument("source", type=Path, help="Folder or ZIP containing raw-890 and reference-890")
    source = parser.parse_args().source

    if source.is_dir():
        _organize(
            _index_files((source / "raw-890").iterdir()),
            _index_files((source / "reference-890").iterdir()),
            None,
        )
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            _organize(
                _zip_files(archive, "raw-890"),
                _zip_files(archive, "reference-890"),
                archive,
            )
    else:
        raise FileNotFoundError(f"Source must be a folder or ZIP archive: {source}")


if __name__ == "__main__":
    main()

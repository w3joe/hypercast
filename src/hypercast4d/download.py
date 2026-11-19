"""Download and safely extract the paper's supplementary data workbook."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


SUPPLEMENT_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41598-025-08957-5/"
    "MediaObjects/41598_2025_8957_MOESM1_ESM.zip"
)

EXPECTED_ARCHIVE_SHA256 = (
    "de5a7cc06ce26b120486da146ff39cdd0ccee5e1e7bc00fec40354a53d73e8d4"
)

MEMBERS = {
    "Experiments/data.xlsx": "paper_data.xlsx",
    "SummaryPlots/Summary.xlsx": "paper_summary.xlsx",
}


def sha256(path: Path) -> str:
    """Compute the hexadecimal SHA-256 checksum of file ``path`` in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    destination: Path,
    force: bool = False,
    allow_updated_source: bool = False,
) -> dict[str, str]:
    """Download, checksum-verify, and extract the Springer Nature supplement archive.

    Parameters
    ----------
    destination : Path
        Target directory (typically ``data/raw``) receiving ``paper_data.xlsx``,
        ``paper_summary.xlsx``, and ``checksums.json``.
    force : bool, default=False
        Re-download ``supplement.zip`` even if it already exists locally.
    allow_updated_source : bool, default=False
        Permit extraction if the publisher archive SHA-256 differs from
        ``EXPECTED_ARCHIVE_SHA256``.

    Returns
    -------
    dict[str, str]
        Recorded URL and SHA-256 checksum manifest written to ``checksums.json``.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "supplement.zip"
    if force or not archive.exists():
        request = urllib.request.Request(
            SUPPLEMENT_URL, headers={"User-Agent": "Hypercast"}
        )
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            archive.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)

    archive_checksum = sha256(archive)
    if archive_checksum != EXPECTED_ARCHIVE_SHA256 and not allow_updated_source:
        raise RuntimeError(
            "Supplement checksum changed: expected "
            f"{EXPECTED_ARCHIVE_SHA256}, received {archive_checksum}. "
            "Inspect the new archive before passing --allow-updated-source."
        )

    extracted: dict[str, str] = {}
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.namelist()
        for suffix, local_name in MEMBERS.items():
            matches = [
                member
                for member in members
                if PurePosixPath(member).as_posix().endswith(suffix)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one archive member ending {suffix!r}, got {matches}"
                )
            target = destination / local_name
            with bundle.open(matches[0]) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted[local_name] = sha256(target)

    checksums = {
        "source_url": SUPPLEMENT_URL,
        "expected_supplement_sha256": EXPECTED_ARCHIVE_SHA256,
        "supplement.zip": archive_checksum,
        **extracted,
    }
    (destination / "checksums.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )
    return checksums


def main() -> None:
    """CLI entry point for ``hypercast4d-download``."""
    from .datasets import DATASETS, NYC_TLC_PARQUET_SHA256, dataset_catalog, download_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Registered dataset ID to download (repeatable; see --list)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download Copper, all 9 TSLib benchmarks, and NYC Taxi Trip Records",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered datasets and local availability",
    )
    parser.add_argument(
        "--nyc-taxi-months",
        default=",".join(NYC_TLC_PARQUET_SHA256),
        help="Comma-separated YYYY-MM months for nyc_taxi aggregation",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-updated-source",
        action="store_true",
        help="Accept an upstream archive whose checksum differs from the pinned copy",
    )
    args = parser.parse_args()
    if args.list:
        print(json.dumps(dataset_catalog(Path.cwd()), indent=2))
        return
    if args.all or args.dataset:
        targets = list(DATASETS) if args.all else args.dataset
        months = tuple(m.strip() for m in args.nyc_taxi_months.split(",") if m.strip())
        results = [
            download_dataset(
                name,
                Path.cwd(),
                force=args.force,
                allow_updated_source=args.allow_updated_source,
                nyc_taxi_months=months,
            )
            for name in targets
        ]
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
        return
    checksums = download(args.destination, args.force, args.allow_updated_source)
    print(json.dumps(checksums, indent=2))


if __name__ == "__main__":
    main()

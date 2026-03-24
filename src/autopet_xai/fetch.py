from __future__ import annotations

import csv
import io
import random
import shutil
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from brain_tumor_xai.utils import ensure_dir

DEFAULT_AUTOPET_FDG_ZIP_URL = "http://193.196.20.155/data/autoPET/data/nifti.zip"
DEFAULT_AUTOPET_FDG_META_URL = "http://193.196.20.155/data/autoPET/data/autoPETmeta.csv"
DEFAULT_AUTOPET_FILENAMES = ("SUV.nii.gz", "CTres.nii.gz", "SEG.nii.gz")


class HTTPRangeReader(io.RawIOBase):
    """Read a remote file through HTTP range requests with a small block cache."""

    def __init__(
        self,
        url: str,
        block_size: int = 8 * 1024 * 1024,
        max_cached_blocks: int = 8,
        timeout: int = 120,
    ) -> None:
        self.url = url
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks
        self.timeout = timeout
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
        if not content_length:
            raise ValueError(f"Missing Content-Length for remote ZIP: {url}")
        self.length = int(content_length)
        self.position = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.length + offset
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported seek mode: {whence}")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""

        if size is None or size < 0:
            size = self.length - self.position

        remaining = min(size, self.length - self.position)
        chunks: List[bytes] = []
        while remaining > 0:
            block_index = self.position // self.block_size
            block = self._get_block(block_index)
            offset = self.position % self.block_size
            take = min(remaining, len(block) - offset)
            chunks.append(block[offset : offset + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        read_size = len(data)
        buffer[:read_size] = data
        return read_size

    def _get_block(self, block_index: int) -> bytes:
        cached = self._cache.get(block_index)
        if cached is not None:
            self._cache.move_to_end(block_index)
            return cached

        start = block_index * self.block_size
        end = min(self.length, start + self.block_size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            block = response.read()
        self._cache[block_index] = block
        self._cache.move_to_end(block_index)
        while len(self._cache) > self.max_cached_blocks:
            self._cache.popitem(last=False)
        return block


def load_autopet_metadata(metadata_url: str = DEFAULT_AUTOPET_FDG_META_URL) -> List[Dict[str, str]]:
    with urllib.request.urlopen(metadata_url, timeout=120) as response:
        raw = response.read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    rows = []
    for row in reader:
        normalized_row = {key: value.strip() for key, value in row.items() if key}
        if normalized_row.get("study_location"):
            rows.append(normalized_row)
    if not rows:
        raise ValueError(f"No metadata rows found at {metadata_url}")
    return rows


def study_prefix_from_location(study_location: str) -> str:
    prefix = study_location.strip().lstrip("./")
    return prefix if prefix.endswith("/") else f"{prefix}/"


def _balanced_case_targets(
    rows: Sequence[Dict[str, str]],
    target_count: int,
    negative_count: Optional[int],
    positive_count: Optional[int],
) -> Tuple[int, int]:
    negatives = [row for row in rows if row.get("diagnosis", "").upper() == "NEGATIVE"]
    positives = [row for row in rows if row.get("diagnosis", "").upper() != "NEGATIVE"]
    if negative_count is None and positive_count is None:
        target_negative = min(len(negatives), target_count // 2)
        target_positive = min(len(positives), target_count - target_negative)
        shortfall = target_count - (target_negative + target_positive)
        if shortfall > 0:
            extra_negative = min(shortfall, len(negatives) - target_negative)
            target_negative += extra_negative
            shortfall -= extra_negative
        if shortfall > 0:
            target_positive += min(shortfall, len(positives) - target_positive)
        return target_negative, target_positive

    target_negative = negative_count or 0
    target_positive = positive_count or 0
    if target_negative + target_positive == 0:
        raise ValueError("At least one of negative_count or positive_count must be positive")
    if target_negative + target_positive != target_count:
        raise ValueError("negative_count + positive_count must match target_count")
    return target_negative, target_positive


def select_autopet_fdg_cases(
    rows: Sequence[Dict[str, str]],
    target_count: int,
    seed: int = 42,
    negative_count: Optional[int] = None,
    positive_count: Optional[int] = None,
) -> List[Dict[str, str]]:
    if target_count <= 0:
        raise ValueError("target_count must be strictly positive")

    negative_rows = [row for row in rows if row.get("diagnosis", "").upper() == "NEGATIVE"]
    positive_rows = [row for row in rows if row.get("diagnosis", "").upper() != "NEGATIVE"]
    target_negative, target_positive = _balanced_case_targets(rows, target_count, negative_count, positive_count)

    if target_negative > len(negative_rows):
        raise ValueError(f"Requested {target_negative} NEGATIVE cases but only {len(negative_rows)} are available")
    if target_positive > len(positive_rows):
        raise ValueError(f"Requested {target_positive} positive cases but only {len(positive_rows)} are available")

    rng = random.Random(seed)
    selected_negative = rng.sample(negative_rows, target_negative) if target_negative else []
    selected_positive = rng.sample(positive_rows, target_positive) if target_positive else []
    selected = sorted([*selected_negative, *selected_positive], key=lambda row: row["study_location"])
    if len(selected) != target_count:
        raise ValueError(f"Expected {target_count} selected cases but got {len(selected)}")
    return selected


def _matching_entries(
    archive: zipfile.ZipFile,
    selected_prefixes: Sequence[str],
    include_filenames: Sequence[str],
) -> List[zipfile.ZipInfo]:
    prefixes = tuple(study_prefix_from_location(prefix) for prefix in selected_prefixes)
    include_names = set(include_filenames)
    return [
        info
        for info in archive.infolist()
        if not info.is_dir() and Path(info.filename).name in include_names and info.filename.startswith(prefixes)
    ]


def extract_autopet_fdg_subset(
    zip_url: str,
    selected_study_prefixes: Sequence[str],
    destination_root: Path,
    include_filenames: Sequence[str] = DEFAULT_AUTOPET_FILENAMES,
    overwrite: bool = False,
) -> Dict[str, object]:
    destination_root = ensure_dir(destination_root)
    reader = HTTPRangeReader(zip_url)
    archive = zipfile.ZipFile(reader)
    members = _matching_entries(archive, selected_study_prefixes, include_filenames)

    expected_file_count = len(selected_study_prefixes) * len(include_filenames)
    if len(members) < expected_file_count:
        raise ValueError(
            f"Only found {len(members)} files for {len(selected_study_prefixes)} studies in the remote archive; "
            f"expected at least {expected_file_count}."
        )

    extracted_files: List[str] = []
    case_to_files: Dict[str, List[str]] = {}

    for info in members:
        relative_path = Path(info.filename)
        target_path = destination_root / relative_path
        ensure_dir(target_path.parent)
        if target_path.exists() and not overwrite:
            extracted_files.append(str(target_path))
            case_to_files.setdefault(str(relative_path.parent), []).append(str(target_path))
            continue

        with archive.open(info) as source, open(target_path, "wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)

        extracted_files.append(str(target_path))
        case_to_files.setdefault(str(relative_path.parent), []).append(str(target_path))

    return {
        "zip_url": zip_url,
        "destination_root": str(destination_root),
        "study_count": len(selected_study_prefixes),
        "file_count": len(extracted_files),
        "include_filenames": list(include_filenames),
        "selected_studies": [study_prefix_from_location(prefix) for prefix in selected_study_prefixes],
        "cases": case_to_files,
    }

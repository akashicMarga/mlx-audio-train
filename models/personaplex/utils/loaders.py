# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import posixpath

from huggingface_hub import hf_hub_download
from pathlib import Path


def _validate_hf_path(path: str) -> str:
    """Validate that an hf:// relative path doesn't escape via traversal."""
    normalized = posixpath.normpath(path)
    if normalized.startswith("..") or normalized.startswith("/"):
        raise ValueError(f"path traversal detected in hf:// URL: {path!r}")
    return normalized


def hf_get(
    filename: str | Path,
    hf_repo: str | None = None,
    check_local_file_exists: bool = False,
) -> Path:
    if isinstance(filename, Path):
        return filename
    if filename.startswith("hf://"):
        parts = filename.removeprefix("hf://").split("/")
        if len(parts) < 3:
            raise ValueError(f"invalid hf:// URL, expected hf://org/repo/path: {filename!r}")
        repo_name = parts[0] + "/" + parts[1]
        rel_path = _validate_hf_path("/".join(parts[2:]))
        return Path(hf_hub_download(repo_name, rel_path))
    elif filename.startswith("file://"):
        # Provide a way to force the read of a local file.
        filename = filename.removeprefix("file://")
        return Path(filename)
    elif hf_repo is not None:
        if check_local_file_exists:
            if Path(filename).exists():
                return Path(filename)
        return Path(hf_hub_download(hf_repo, filename))
    else:
        return Path(filename)

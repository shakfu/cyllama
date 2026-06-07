#!/usr/bin/env python3
"""Download model files in parallel, verify SHA256, cancel all on failure.

Each file is downloaded using multiple connections (HTTP Range requests) for
maximum throughput, similar to aria2. Falls back to single-connection if the
server doesn't support Range.

Architecture (deadlock-free):
  - file_pool: runs download_one() — one thread per file.
  - chunk_pool: runs download_chunk() — leaf I/O tasks only.
  File threads submit chunks to the chunk pool and wait; chunk tasks never
  submit sub-tasks, so no circular dependency → no deadlock.
"""

import hashlib
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

MODELS_DIR = Path("models")
CHUNKS_PER_FILE = 2  # Number of parallel connections per file download.
MAX_CONCURRENT_FILES = 1  # Max files downloading simultaneously.
READ_CHUNK_SIZE = 1 << 23  # 8 MB — buffer size for network reads / disk writes.
MAX_HTTP_RETRIES = 3
RATE_LIMIT_BASE_SLEEP = 30.0
REQUEST_START_DELAY = 1.0


@dataclass(frozen=True)
class RetryDelay:
    seconds: float
    detail: str


# (filename, url, sha256)
MODELS = [
    (
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf",
        "432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3",
    ),
    (
        "tinygemma3-Q8_0.gguf",
        "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/tinygemma3-Q8_0.gguf",
        "7566ae7219c93ea2ecc692a931ee122d30c55261d0e2c3347acb8b939d2e9abd",
    ),
    (
        "mmproj-tinygemma3.gguf",
        "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/mmproj-tinygemma3.gguf",
        "93c2ba8c34574dd8f2dfda64931fc20943de2f941bfe03e6e9eca68951b80604",
    ),
    (
        "Qwen3-Embedding-0.6B-Q8_0.gguf",
        "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf",
        "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
    ),
    (
        "bge-reranker-v2-m3-Q2_K.gguf",
        "https://modelscope.cn/models/gpustack/bge-reranker-v2-m3-GGUF/resolve/master/bge-reranker-v2-m3-Q2_K.gguf",
        "f12135b80de836cbf94c1169dc8efda57c81040c1dfd9dedc20709d2e1725e39",
    ),
    (
        "stories15M_MOE-F16.gguf",
        "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/stories15M_MOE-F16.gguf",
        "1240dfc1957df9f3550dd6c1d9e64b466fc2f452d8bc34bd4e45e1a1e2ca6055",
    ),
    (
        "stories15M-q4_0.gguf",
        "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf",
        "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",
    ),
    (
        "moe_shakespeare15M.gguf",
        "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf",
        "d1e0617d7e10de960639d18a4620ec8c6bb56343f45692830d3634a1a3e1fe1a",
    ),
]


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest using the fastest available method."""
    with open(path, "rb") as f:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(f, "sha256").hexdigest()
        h = hashlib.sha256()
        while chunk := f.read(READ_CHUNK_SIZE):
            h.update(chunk)
        return h.hexdigest()


def _retry_after_delay(err: urllib.error.HTTPError, attempt: int) -> RetryDelay:
    """Return the server-requested or exponential sleep time for HTTP 429."""
    value = err.headers.get("Retry-After")
    if value:
        try:
            seconds = max(float(value), REQUEST_START_DELAY)
            return RetryDelay(seconds, f"Retry-After={value!r} ({seconds:.0f}s)")
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                server_now = time.time()
                response_date = err.headers.get("Date")
                if response_date:
                    try:
                        server_now = parsedate_to_datetime(response_date).timestamp()
                    except (TypeError, ValueError):
                        pass
                seconds = max(retry_at.timestamp() - server_now, REQUEST_START_DELAY)
                if response_date:
                    detail = (
                        f"Retry-After date={value!r}, response Date={response_date!r} "
                        f"({seconds:.0f}s)"
                    )
                else:
                    detail = (
                        f"Retry-After date={value!r}, no response Date ({seconds:.0f}s)"
                    )
                return RetryDelay(seconds, detail)
            except (TypeError, ValueError):
                pass

    seconds = RATE_LIMIT_BASE_SLEEP * (2**attempt)
    if value:
        detail = f"unparseable Retry-After={value!r}; using exponential backoff"
    else:
        detail = "no Retry-After header; using exponential backoff"
    return RetryDelay(seconds, detail)


def _sleep_before_request() -> None:
    if REQUEST_START_DELAY > 0:
        time.sleep(REQUEST_START_DELAY)


def open_url_with_retries(req: urllib.request.Request, timeout: int, description: str):
    """Open a URL, retrying HTTP 429 with backoff."""
    request_timeout = timeout
    for attempt in range(MAX_HTTP_RETRIES + 1):
        try:
            _sleep_before_request()
            return urllib.request.urlopen(req, timeout=request_timeout)
        except urllib.error.HTTPError as err:
            if err.code != 429 or attempt == MAX_HTTP_RETRIES:
                raise

            retry_delay = _retry_after_delay(err, attempt)
            sleep_for = retry_delay.seconds
            request_timeout = max(timeout, int(sleep_for) + 1)
            err.close()
            print(
                f"HTTP 429 while requesting {description}; "
                f"{retry_delay.detail}; "
                f"next timeout {request_timeout}s; "
                f"sleeping {sleep_for:.0f}s before retry {attempt + 1}/"
                f"{MAX_HTTP_RETRIES}...",
                flush=True,
            )
            time.sleep(sleep_for)


def get_file_info(url: str) -> tuple[str, int] | None:
    """Follow redirects and get (final_url, file_size) via HEAD request.

    Returns None if Range is not supported or Content-Length unknown.
    """
    try:
        req = urllib.request.Request(url, method="HEAD")
        with open_url_with_retries(req, timeout=30, description="file info") as resp:
            final_url = resp.url  # After redirects.
            accept_ranges = resp.headers.get("Accept-Ranges", "")
            length = resp.headers.get("Content-Length")
            if accept_ranges.lower() == "bytes" and length:
                return final_url, int(length)
    except urllib.error.HTTPError as err:
        if err.code == 429:
            raise
    except Exception:
        pass
    return None


def _write_at(fd: int, data: bytes, offset: int, lock: threading.Lock) -> None:
    """Write *data* at *offset* into *fd*, cross-platform.

    Uses os.pwrite when available (atomic positional write).
    Falls back to lock + lseek + write otherwise (e.g. Windows).
    """
    if hasattr(os, "pwrite"):
        os.pwrite(fd, data, offset)
    else:
        with lock:
            os.lseek(fd, offset, os.SEEK_SET)
            os.write(fd, data)


def download_chunk(
    url: str, start: int, end: int, fd: int, lock: threading.Lock
) -> None:
    """Download a byte range and write to fd at the correct offset."""
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={start}-{end}")
    with open_url_with_retries(
        req, timeout=300, description=f"bytes {start}-{end}"
    ) as resp:
        offset = start
        while data := resp.read(READ_CHUNK_SIZE):
            _write_at(fd, data, offset, lock)
            offset += len(data)


def download_single_connection(url: str, dest: Path) -> None:
    """Download a file using one HTTP connection with 429 retries."""
    try:
        req = urllib.request.Request(url)
        with (
            open_url_with_retries(req, timeout=300, description=dest.name) as resp,
            open(dest, "wb") as out,
        ):
            while data := resp.read(READ_CHUNK_SIZE):
                out.write(data)
    except Exception:
        dest.unlink(missing_ok=True)
        raise


def download_file(
    url: str, dest: Path, num_chunks: int, chunk_pool: ThreadPoolExecutor
) -> None:
    """Download a file using multiple parallel connections (Range requests).

    Chunk downloads are submitted to *chunk_pool*.
    Falls back to single-connection download if Range is not supported.
    """
    file_info = get_file_info(url)

    # Fallback: single connection (server doesn't support Range, or tiny file).
    if file_info is None or file_info[1] < READ_CHUNK_SIZE * num_chunks:
        download_single_connection(url, dest)
        return

    final_url, file_size = file_info

    # Pre-allocate the file.
    # O_BINARY is required on Windows to prevent \n → \r\n translation;
    # it is 0 on POSIX (no-op).
    flags = os.O_CREAT | os.O_RDWR | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    fd = os.open(str(dest), flags)
    try:
        if hasattr(os, "ftruncate"):
            os.ftruncate(fd, file_size)
        else:
            os.lseek(fd, file_size - 1, os.SEEK_SET)
            os.write(fd, b"\0")

        # Split into chunks and download in parallel.
        chunk_size = file_size // num_chunks
        lock = threading.Lock()  # Used only when os.pwrite is unavailable.

        chunk_futures: list[Future[None]] = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = file_size - 1 if i == num_chunks - 1 else (i + 1) * chunk_size - 1
            fut = chunk_pool.submit(download_chunk, final_url, start, end, fd, lock)
            chunk_futures.append(fut)

        # Wait for all chunks and propagate the first error.
        done, _ = wait(chunk_futures)
        for fut in done:
            fut.result()  # Raises if the chunk failed.
    finally:
        os.close(fd)


def download_one(
    name: str, url: str, expected_sha: str, chunk_pool: ThreadPoolExecutor
) -> str:
    """Download and verify a single model file. Returns a status message."""
    dest = MODELS_DIR / name

    # If file exists, verify integrity; redownload if mismatch.
    if dest.exists():
        actual = sha256_file(dest)
        if actual == expected_sha:
            return f"[ok] {name} (already exists, verified)"
        else:
            print(f"  {name}: SHA256 mismatch, redownloading...", flush=True)
            dest.unlink()

    print(f"Downloading {name} ({CHUNKS_PER_FILE} connections)...", flush=True)
    try:
        download_file(url, dest, CHUNKS_PER_FILE, chunk_pool)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: download failed -- {e}")

    actual = sha256_file(dest)
    if actual != expected_sha:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"{name}: SHA256 mismatch after download.\n"
            f"  Expected: {expected_sha}\n"
            f"  Got:      {actual}"
        )

    return f"  [ok] {name} downloaded and verified"


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Two pools, no deadlock:
    #   file_pool  — runs download_one (submits to chunk_pool, waits)
    #   chunk_pool — runs download_chunk (leaf tasks, never submits sub-tasks)
    with ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_FILES * CHUNKS_PER_FILE
    ) as chunk_pool:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FILES) as file_pool:
            futures = {
                file_pool.submit(download_one, name, url, sha, chunk_pool): name
                for name, url, sha in MODELS
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    msg = future.result()
                    print(msg, flush=True)
                except Exception as e:
                    for f in futures:
                        f.cancel()
                    print(
                        f"\n{'=' * 50}\n" f"ERROR: {e}\n" f"{'=' * 50}",
                        file=sys.stderr,
                    )
                    chunk_pool.shutdown(wait=False, cancel_futures=True)
                    file_pool.shutdown(wait=False, cancel_futures=True)
                    return 1

    print(f"\nAll models ready in {MODELS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

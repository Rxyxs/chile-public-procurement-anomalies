"""Downloads the raw public data and caches it in ``data/raw/``.

- **ChileCompra open data**: one ZIP per month with every purchase order (*orden de compra*)
  sent through Mercado Público, one row per order line, from
  ``https://transparenciachc.blob.core.windows.net/oc-da/<year>-<month>.zip`` (no key needed).
- **mindicador.cl**: the monthly UTM (*unidad tributaria mensual*), the unit in which the
  procurement law sets its thresholds.

The study window is fixed in ``MONTHS``: January to November 2024, when the Compra Ágil cap
was 30 UTM, and January 2025 to September 2026, after Ley 21.634 raised it to 100 UTM on
12 December 2024. December 2024 mixes both rules and is left out. A failed download raises:
there is no fallback.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OC_URL = "https://transparenciachc.blob.core.windows.net/oc-da/{year}-{month}.zip"
UTM_URL = "https://mindicador.cl/api/utm/{year}"
USER_AGENT = "public-procurement-anomaly-engine (github.com/Rxyxs)"
TIMEOUT_S = 300
MONTHS = [(2024, m) for m in range(1, 12)] + [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 10)]


def oc_path(year: int, month: int, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"oc_{year}-{month}.zip"


def utm_path(year: int, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"utm_{year}.json"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.read()


def _download(url: str, path: Path) -> int:
    """Streams ``url`` into ``path`` in 1 MB chunks (``response.read()`` on a 90 MB file
    ran at about 1 MB/s here) through a temporary file, so a broken download never
    leaves a half-written ZIP behind."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    with open(tmp, "rb") as check:
        if check.read(2) != b"PK":
            tmp.unlink()
            raise RuntimeError(f"{url} did not return a ZIP")
    tmp.replace(path)
    return path.stat().st_size


def download_all(raw_dir: Path = RAW_DIR, months: list[tuple[int, int]] = MONTHS, workers: int = 6) -> list[Path]:
    """Downloads every month and UTM year that is not cached yet; returns the paths written."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for year in sorted({y for y, _ in months}):
        path = utm_path(year, raw_dir)
        if not path.exists():
            payload = json.loads(_get(UTM_URL.format(year=year)))
            if not payload.get("serie"):
                raise RuntimeError(f"mindicador returned no UTM values for {year}")
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            written.append(path)
    pending = [(y, m) for y, m in months if not oc_path(y, m, raw_dir).exists()]

    def fetch(year_month: tuple[int, int]) -> Path:
        year, month = year_month
        path = oc_path(year, month, raw_dir)
        size = _download(OC_URL.format(year=year, month=month), path)
        print(f"downloaded {path.name} ({size / 1e6:.0f} MB)", flush=True)
        return path

    # The server serves each connection at about 1 MB/s, so a few run side by side.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        written.extend(pool.map(fetch, pending))
    return written

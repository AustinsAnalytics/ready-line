"""Stage 0 - download the public data.

* NASA C-MAPSS turbofan degradation data (PHM Society mirror of the NASA PCoE repository).
* FAA Service Difficulty Reports, one CSV per calendar year (2016-2025).

Files land in data/raw/. Re-running skips anything already downloaded.
"""
from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from common import RAW

CMAPSS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
SDR_URL = "https://external.apic4e.faa.gov/sdrs/retrieve/SDR-{year}.csv"
YEARS = range(2016, 2026)
UA = {"User-Agent": "Mozilla/5.0 (ready-line research pipeline)"}


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.relative_to(RAW.parent.parent)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get  {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def cmapss() -> None:
    out = RAW / "cmapss"
    if (out / "train_FD001.txt").exists():
        print("  have C-MAPSS")
        return
    z = out / "cmapss.zip"
    fetch(CMAPSS_URL, z)
    with zipfile.ZipFile(z) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        outer.extract(inner_name, out)
    with zipfile.ZipFile(out / inner_name) as inner:
        for name in inner.namelist():
            if name.endswith(".txt"):
                (out / Path(name).name).write_bytes(inner.read(name))
    print(f"  extracted {len(list(out.glob('*.txt')))} C-MAPSS text files")


def sdr() -> None:
    for y in YEARS:
        fetch(SDR_URL.format(year=y), RAW / "sdr" / f"SDR-{y}.csv")


if __name__ == "__main__":
    cmapss()
    sdr()

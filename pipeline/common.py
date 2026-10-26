"""Shared paths, constants and helpers for the Ready Line pipeline."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
WAREHOUSE = ROOT / "data" / "warehouse.duckdb"
SITE_DATA = ROOT / "site" / "data"

for p in (PROCESSED, SITE_DATA):
    p.mkdir(parents=True, exist_ok=True)

SEED = 20270501
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
CMAPSS_COLS = ["unit", "cycle", *SETTINGS, *SENSORS]

# Sensor descriptions from Saxena et al. (2008), Table 2.
SENSOR_INFO = {
    "s1": ("T2", "Total temperature at fan inlet", "°R"),
    "s2": ("T24", "Total temperature at LPC outlet", "°R"),
    "s3": ("T30", "Total temperature at HPC outlet", "°R"),
    "s4": ("T50", "Total temperature at LPT outlet", "°R"),
    "s5": ("P2", "Pressure at fan inlet", "psia"),
    "s6": ("P15", "Total pressure in bypass duct", "psia"),
    "s7": ("P30", "Total pressure at HPC outlet", "psia"),
    "s8": ("Nf", "Physical fan speed", "rpm"),
    "s9": ("Nc", "Physical core speed", "rpm"),
    "s10": ("epr", "Engine pressure ratio (P50/P2)", "–"),
    "s11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "s12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "s13": ("NRf", "Corrected fan speed", "rpm"),
    "s14": ("NRc", "Corrected core speed", "rpm"),
    "s15": ("BPR", "Bypass ratio", "–"),
    "s16": ("farB", "Burner fuel-air ratio", "–"),
    "s17": ("htBleed", "Bleed enthalpy", "–"),
    "s18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    "s20": ("W31", "HPT coolant bleed", "lbm/s"),
    "s21": ("W32", "LPT coolant bleed", "lbm/s"),
}

SUBSET_INFO = {
    "FD001": {"conditions": 1, "faults": "HPC degradation"},
    "FD002": {"conditions": 6, "faults": "HPC degradation"},
    "FD003": {"conditions": 1, "faults": "HPC + fan degradation"},
    "FD004": {"conditions": 6, "faults": "HPC + fan degradation"},
}

# JASC chapter titles (FAA Joint Aircraft System/Component code, derived from ATA iSpec 2200).
JASC_CHAPTERS = {
    "11": "Placards & markings", "12": "Servicing", "18": "Helicopter vibration",
    "21": "Air conditioning", "22": "Auto flight", "23": "Communications",
    "24": "Electrical power", "25": "Equipment & furnishings", "26": "Fire protection",
    "27": "Flight controls", "28": "Fuel", "29": "Hydraulic power",
    "30": "Ice & rain protection", "31": "Indicating & recording", "32": "Landing gear",
    "33": "Lights", "34": "Navigation", "35": "Oxygen", "36": "Pneumatic",
    "37": "Vacuum", "38": "Water & waste", "45": "Central maintenance system",
    "46": "Information systems", "49": "Auxiliary power unit", "51": "Structures (general)",
    "52": "Doors", "53": "Fuselage", "54": "Nacelles & pylons", "55": "Stabilizers",
    "56": "Windows", "57": "Wings", "61": "Propellers", "62": "Main rotor",
    "63": "Rotor drive", "64": "Tail rotor", "65": "Tail rotor drive",
    "67": "Rotor flight control", "71": "Powerplant", "72": "Turbine engine",
    "73": "Engine fuel & control", "74": "Ignition", "75": "Engine bleed air",
    "76": "Engine controls", "77": "Engine indicating", "78": "Engine exhaust",
    "79": "Engine oil", "80": "Starting", "81": "Turbocharging", "82": "Water injection",
    "83": "Accessory gearboxes", "85": "Reciprocating engine",
}
ENGINE_CHAPTERS = {"71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "83"}

# SDR code tables. Precautionary procedure and nature-of-condition letters follow the
# shared FAA / Transport Canada SDR code structure (TC AC 521-009, Appendix A, Block 7).
PRECAUTION_CODES = {
    "A": "Unscheduled landing", "B": "Emergency descent", "C": "Aborted take-off",
    "D": "Return to blocks", "E": "Engine shutdown / prop feather",
    "F": "Fire extinguisher activated", "G": "Oxygen masks deployed",
    "H": "System deactivated", "I": "Intentional depressurization", "J": "Fuel dumped",
    "K": "None", "L": "Aborted approach", "M": "Evasive action",
    "N": "Abnormal landing configuration", "O": "Other",
    "P": "Emergency gear extension", "Q": "Declared emergency", "R": "Autorotation",
}
# Codes that mean the flight was actually interrupted (excludes None / Other / blank).
INTERRUPTION_CODES = set(PRECAUTION_CODES) - {"K", "O"}

STAGE_GROUPS = {
    "IN": "Inspection / maintenance", "SM": "Inspection / maintenance",
    "UM": "Inspection / maintenance", "SI": "Inspection / maintenance",
    "CI": "Inspection / maintenance", "SR": "Inspection / maintenance",
    "OH": "Inspection / maintenance", "SV": "Inspection / maintenance",
    "TX": "Ground ops", "GR": "Ground ops", "ST": "Ground ops",
    "TO": "Take-off & climb", "CL": "Take-off & climb",
    "CR": "Cruise", "DE": "Descent & landing", "AP": "Descent & landing",
    "LD": "Descent & landing",
}


def rnd(x, n=4):
    """Round floats (recursively) so exported JSON stays small."""
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        if x != 0 and abs(x) < 0.01:
            return float(f"{x:.{n}g}")      # keep significant digits (p-values, small rates)
        return round(x, n)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), n)
    return x


def export(name: str, obj, n: int = 4) -> Path:
    """Write a JSON payload consumed by the website."""
    path = SITE_DATA / f"{name}.json"
    path.write_text(json.dumps(rnd(obj, n), separators=(",", ":")))
    print(f"  -> site/data/{name}.json ({path.stat().st_size / 1024:.1f} KB)")
    return path


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"[{self.label}] ...", flush=True)
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        print(f"[{self.label}] done in {self.seconds:.1f}s", flush=True)

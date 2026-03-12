from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path("/home/jherec/methane-filters-benchmark")

MULTIPLICATIVE_METRICS_CSV = ROOT / "multiplicative_all.csv"
FINAL_DATA_METRICS_CSV = ROOT / "csvs" / "final_data_matrics.csv"
BAND_N_TEMPLATE_CSV = ROOT / "csvs" / "band_n.csv"
PROCESSING_TIMES_CSV = ROOT / "processing_times_final.csv"

BAND_N_OUTPUT_CSV = ROOT / "csvs" / "band_n_updated.csv"
FINAL_DATA_METRICS_OUTPUT_CSV = ROOT / "csvs" / "final_data_matrics_updated.csv"


# User-editable: choose which methods should receive metrics from multiplicative_all.csv
MULTIPLICATIVE_METHODS = {"ACE", "MF"}

# User-editable: map processing_times_final.csv method names to band_n METHOD names.
PROCESSING_TO_BAND_METHOD = {
    "ACE_optimized": "ACE",
    "CEM_optimized": "CEM",
    "MatchedFilterOptimized": "MF",
    "mag1c_tile": "Mag1c (tile-wise)",
    "mag1c_SAS": "Mag1c-SAS",
}

# User-editable: which processing_times_final.csv column should be used for runtime.
RUNTIME_SOURCE_COLUMN = "Mean"

RUNTIME_COLUMNS_TO_DROP = {"RUNTIME_optimized (s)", "runtime visualized"}

# Only these metric columns are replaced.
METRIC_COLUMNS = [
    "AUPRC_all",
    "AUPRC_strong",
    "Recall_all_seg",
    "Precision_all_seg",
    "F1-score_all_seg",
    "Recall_strong_seg",
    "Precision_strong_seg",
    "F1-score_strong_seg",
]


def _normalize_channel(channel_value: str) -> str:
    return str(int(float(channel_value)))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_multiplicative_index(
    rows: List[Dict[str, str]],
) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    index: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        key = (
            row["METHOD"],
            _normalize_channel(row["CHANNEL_N"]),
            row["SELECT_STRATEGY"],
        )
        index[key] = row
    return index


def _patch_metrics_from_index(
    rows: List[Dict[str, str]],
    multiplicative_index: Dict[Tuple[str, str, str], Dict[str, str]],
    label: str,
) -> Tuple[int, List[Tuple[str, str, str]]]:
    updated = 0
    missing: List[Tuple[str, str, str]] = []

    for row in rows:
        method = row["METHOD"]
        if method not in MULTIPLICATIVE_METHODS:
            continue

        key = (
            method,
            _normalize_channel(row["CHANNEL_N"]),
            row["SELECT_STRATEGY"],
        )
        source_row = multiplicative_index.get(key)
        if source_row is None:
            missing.append(key)
            continue

        for col in METRIC_COLUMNS:
            row[col] = source_row[col]
        updated += 1

    print(f"[{label}] Updated rows: {updated}")
    if missing:
        print(f"[{label}] Missing multiplicative rows: {len(missing)}")
        for key in missing:
            print(f"  - METHOD={key[0]}, CHANNEL_N={key[1]}, SELECT_STRATEGY={key[2]}")

    return updated, missing


def _build_runtime_index(
    rows: List[Dict[str, str]],
) -> Dict[Tuple[str, str], Dict[str, str]]:
    index: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        processing_name = row["Method"]
        band_method = PROCESSING_TO_BAND_METHOD.get(processing_name)
        if band_method is None:
            continue

        key = (band_method, _normalize_channel(row["Channels"]))
        if key in index:
            raise ValueError(
                f"Duplicate runtime mapping for {key}. "
                "Keep only one processing method per Band N method/channel."
            )
        index[key] = row
    return index


def _patch_band_n_runtimes(
    band_rows: List[Dict[str, str]],
    runtime_index: Dict[Tuple[str, str], Dict[str, str]],
) -> Tuple[int, List[Tuple[str, str]]]:
    updated = 0
    missing: List[Tuple[str, str]] = []

    for row in band_rows:
        key = (row["METHOD"], _normalize_channel(row["CHANNEL_N"]))
        source_row = runtime_index.get(key)
        if source_row is None:
            missing.append(key)
            continue

        runtime_val = float(source_row[RUNTIME_SOURCE_COLUMN])
        row["RUNTIME_original (s)"] = f"{runtime_val:.3f}"
        updated += 1

    print(f"[band_n] Updated runtime rows: {updated}")
    if missing:
        print(f"[band_n] Missing runtime rows: {len(missing)}")
        for key in missing:
            print(f"  - METHOD={key[0]}, CHANNEL_N={key[1]}")

    return updated, missing


def main() -> None:
    multiplicative_rows = _read_csv(MULTIPLICATIVE_METRICS_CSV)
    multiplicative_index = _build_multiplicative_index(multiplicative_rows)

    band_rows = _read_csv(BAND_N_TEMPLATE_CSV)
    _patch_metrics_from_index(
        rows=band_rows,
        multiplicative_index=multiplicative_index,
        label="band_n",
    )

    processing_rows = _read_csv(PROCESSING_TIMES_CSV)
    runtime_index = _build_runtime_index(processing_rows)
    _patch_band_n_runtimes(band_rows, runtime_index)

    band_fieldnames = [k for k in band_rows[0].keys() if k not in RUNTIME_COLUMNS_TO_DROP]
    for row in band_rows:
        for col in RUNTIME_COLUMNS_TO_DROP:
            row.pop(col, None)
    _write_csv(BAND_N_OUTPUT_CSV, band_rows, band_fieldnames)
    print(f"[band_n] Wrote: {BAND_N_OUTPUT_CSV}")

    final_data_rows = _read_csv(FINAL_DATA_METRICS_CSV)
    _patch_metrics_from_index(
        rows=final_data_rows,
        multiplicative_index=multiplicative_index,
        label="final_data_matrics",
    )
    _write_csv(FINAL_DATA_METRICS_OUTPUT_CSV, final_data_rows, final_data_rows[0].keys())
    print(f"[final_data_matrics] Wrote: {FINAL_DATA_METRICS_OUTPUT_CSV}")


if __name__ == "__main__":
    main()

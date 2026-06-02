from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple


inference = "/home/jherec/methane-filters-benchmark/emit_50_inference_50_class.csv"
mag1c_sas = "/home/jherec/methane-filters-benchmark/emit_50_mag1c_sas_50_class.csv"
mag1c_original = "/home/jherec/methane-filters-benchmark/emit_50_mag1c_original_50_class.csv"
output_tex = "/home/jherec/methane-filters-benchmark/benchmark/emit_deployment/table.tex"

threshold_mag1c_sas = 1200
threshold_mag1c_original = 1200
threshold_inference = 0.5

STRATEGY_ORDER: List[Tuple[str, str]] = [
    ("starcop_filtering", "Variance Increase"),
    ("highest_transmittance", "Highest Transmittance"),
    ("evenly_spaced", "Evenly spaced"),
]

METRIC_COLUMNS: List[Tuple[str, str]] = [
    ("AUPRC_all", "AUPRC"),
    ("AUPRC_strong", "AUPRC (Strong)"),
    ("F1-score_all_seg", "F1"),
    ("F1-score_strong_seg", "F1 (Strong)"),
    ("F1-score_all_class_clas", "F1 (Classification)"),
]
LINKNET_LABEL = r"\shortstack[l]{Linknet\\(RGB+Mag1c-SAS)}"


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_float(value: str) -> float:
    return float(value)


def _format_metric(value: str) -> str:
    # Convert to percentage and truncate to 2 decimal places.
    pct = float(value) * 100.0
    pct_trunc = int(pct * 100) / 100.0
    return f"{pct_trunc:.2f}"


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _find_row(
    rows: List[Dict[str, str]],
    method_prefix: str,
    strategy: str,
    threshold: float,
) -> Dict[str, str]:
    matches: List[Dict[str, str]] = []
    expected = f"data_selected_50_{strategy}"
    for row in rows:
        method = row["METHOD"]
        if not method.startswith(method_prefix):
            continue
        if expected not in method:
            continue
        if _as_float(row["THRESHOLD"]) != threshold:
            continue
        matches.append(row)

    if not matches:
        raise ValueError(
            f"Missing row for method_prefix={method_prefix}, strategy={strategy}, threshold={threshold}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple rows for method_prefix={method_prefix}, strategy={strategy}, threshold={threshold}"
        )
    return matches[0]


def _compute_best_metric_values(rows: List[Dict[str, str]]) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for col, _ in METRIC_COLUMNS:
        best[col] = max(_as_float(r[col]) for r in rows)
    return best


def _metrics_cells(
    row: Dict[str, str],
    best_values: Dict[str, float] | None = None,
    highlight: bool = False,
) -> str:
    vals: List[str] = []
    for col, _ in METRIC_COLUMNS:
        raw = _as_float(row[col])
        txt = _format_metric(row[col])
        if highlight and best_values is not None and abs(raw - best_values[col]) <= 1e-12:
            txt = f"\\textbf{{{txt}}}"
        vals.append(txt)
    return " & ".join(vals)


def _build_table() -> str:
    inference_rows = _read_rows(inference)
    sas_rows = _read_rows(mag1c_sas)
    original_rows = _read_rows(mag1c_original)

    mag1c_original_row = _find_row(
        original_rows,
        method_prefix="MAG1Cdata_selected_50_",
        strategy="evenly_spaced",
        threshold=float(threshold_mag1c_original),
    )

    tiling_rows = [
        _find_row(
            sas_rows,
            method_prefix="MAG1C_SAS_TILINGdata_selected_50_",
            strategy=strategy,
            threshold=float(threshold_mag1c_sas),
        )
        for strategy, _ in STRATEGY_ORDER
    ]
    no_tiling_rows = [
        _find_row(
            sas_rows,
            method_prefix="MAG1C_SASdata_selected_50_",
            strategy=strategy,
            threshold=float(threshold_mag1c_sas),
        )
        for strategy, _ in STRATEGY_ORDER
    ]
    tiling_linknet_rows = [
        _find_row(
            inference_rows,
            method_prefix="INFERENCE_TILINGdata_selected_50_",
            strategy=strategy,
            threshold=float(threshold_inference),
        )
        for strategy, _ in STRATEGY_ORDER
    ]
    no_tiling_linknet_rows = [
        _find_row(
            inference_rows,
            method_prefix="INFERENCE_TILING_INFERENCE_ONLYdata_selected_50_",
            strategy=strategy,
            threshold=float(threshold_inference),
        )
        for strategy, _ in STRATEGY_ORDER
    ]
    contender_rows = tiling_linknet_rows + no_tiling_linknet_rows + tiling_rows + no_tiling_rows
    best_values = _compute_best_metric_values(contender_rows)

    header_metrics = " & ".join(
        "\\textbf{\\shortstack{" + _latex_escape(label) + "\\\\{[\\%]}}}" for _, label in METRIC_COLUMNS
    )

    lines = [
        "\\begin{table*}[ht!]",
        "\\centering",
        "\\caption{Score of methods on EMIT dataset. For entries that do not mention an ML model, the morphological baseline was applied for inference.}",
        "\\label{tab:emit-deployment-metrics}",
        "\\begin{tabular}{lll" + "c" * len(METRIC_COLUMNS) + "}",
        "\\toprule",
        "\\textbf{Setup} & \\textbf{Method} & \\textbf{Band Selection} & " + header_metrics + " \\\\",
        "\\midrule",
        "- & Mag1c (original) & Evenly spaced & "
        + _metrics_cells(mag1c_original_row, best_values=best_values, highlight=False)
        + " \\\\",
        "\\midrule",
        "\\multirow{6}{*}{\\shortstack[l]{Mag1c-SAS on tiles\\\\(512x512)}} & \\multirow{3}{*}{"
        + LINKNET_LABEL
        + "} & "
        + STRATEGY_ORDER[0][1]
        + " & "
        + _metrics_cells(tiling_linknet_rows[0], best_values=best_values, highlight=True)
        + " \\\\",
        "& & "
        + STRATEGY_ORDER[1][1]
        + " & "
        + _metrics_cells(tiling_linknet_rows[1], best_values=best_values, highlight=True)
        + " \\\\",
        "& & "
        + STRATEGY_ORDER[2][1]
        + " & "
        + _metrics_cells(tiling_linknet_rows[2], best_values=best_values, highlight=True)
        + " \\\\",
        "\\cmidrule(lr){2-8}",
        "& \\multirow{3}{*}{Mag1c-SAS} & "
        + STRATEGY_ORDER[0][1]
        + " & "
        + _metrics_cells(tiling_rows[0], best_values=best_values, highlight=True)
        + " \\\\",
        "& & "
        + STRATEGY_ORDER[1][1]
        + " & "
        + _metrics_cells(tiling_rows[1], best_values=best_values, highlight=True)
        + " \\\\",
        "& & "
        + STRATEGY_ORDER[2][1]
        + " & "
        + _metrics_cells(tiling_rows[2], best_values=best_values, highlight=True)
        + " \\\\",
        "\\midrule",
        "\\multirow{6}{*}{\\shortstack[l]{Mag1c-SAS\\\\on whole image}} & \\multirow{3}{*}{"
        + LINKNET_LABEL
        + "} & "
        + STRATEGY_ORDER[0][1]
        + " & "
        + _metrics_cells(no_tiling_linknet_rows[0], best_values=best_values, highlight=True)
        + " \\\\",
        "& & "
        + STRATEGY_ORDER[1][1]
        + " & "
        + _metrics_cells(no_tiling_linknet_rows[1], best_values=best_values, highlight=True)
        + " \\\\",
        "& & "
        + STRATEGY_ORDER[2][1]
        + " & "
        + _metrics_cells(no_tiling_linknet_rows[2], best_values=best_values, highlight=True)
        + " \\\\",
        "\\cmidrule(lr){2-8}",
        "& \\multirow{3}{*}{Mag1c-SAS} & "
        + STRATEGY_ORDER[0][1]
        + " & "
        + _metrics_cells(no_tiling_rows[0], best_values=best_values, highlight=True)
        + " \\\\",
        "& & "
        + STRATEGY_ORDER[1][1]
        + " & "
        + _metrics_cells(no_tiling_rows[1], best_values=best_values, highlight=True)
        + " \\\\",
        "& & "
        + STRATEGY_ORDER[2][1]
        + " & "
        + _metrics_cells(no_tiling_rows[2], best_values=best_values, highlight=True)
        + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "% Requires: \\usepackage{booktabs} and \\usepackage{multirow}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    table = _build_table()
    out = Path(output_tex)
    out.write_text(table, encoding="utf-8")
    print(f"Wrote LaTeX table to: {out}")


if __name__ == "__main__":
    main()

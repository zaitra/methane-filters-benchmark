from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


power_csv = "/home/jherec/methane-filters-benchmark/benchmark/emit_deployment/power_tiling.csv"
resource_usage_csv = "/home/jherec/methane-filters-benchmark/benchmark/emit_deployment/resource_usage_tiling.csv"
timestamps_csv = "/home/jherec/methane-filters-benchmark/benchmark/emit_deployment/timestamps_tiling.csv"

power_csv_tiling_for_inference_only = "/home/jherec/methane-filters-benchmark/benchmark/emit_deployment/power_tiling_inference.csv"
resource_usage_csv_tiling_for_inference_only = "/home/jherec/methane-filters-benchmark/benchmark/emit_deployment/resource_usage_tiling_inference.csv"
timestamps_csv_tiling_for_inference_only = "/home/jherec/methane-filters-benchmark/benchmark/emit_deployment/timestamps_tiling_inference.csv"

output_csv = "/home/jherec/methane-filters-benchmark/benchmark/emit_deployment/statistics.csv"
output_dir = "/home/jherec/methane-filters-benchmark/benchmark/emit_deployment"

PLOT_DPI = 300
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 28
TICK_FONTSIZE = 24
LEGEND_FONTSIZE = 19
LINE_WIDTH = 4
GRID_ALPHA = 0.30
BLACK_TELEMETRY = False


POWER_PHASES = ("idle_before", "running", "idle_after")
RESOURCE_COLUMNS = {
    "cpu_percent": "cpu_percent",
    "ram_used_MB": "ram_used_MB",
    "proc_read_MB": "proc_read_MB",
    "proc_write_MB": "proc_write_MB",
}

# User-facing labels for timestamps actions.
ACTION_VERBOSE_NAMES = {
    "prep_init": "Preparation initialization",
    "load_scene": "Load bands",
    "tile_image": "Tiling",
    "mag1c_prep": "Mag1c-SAS preprocessing",
    "mag1c_compute": "Mag1c-SAS compute",
    "mag1c_post": "Mag1c-SAS postprocessing",
    "normalize": "Normalization",
    "inference_tiles": "Model inference",
    "post_tiles": "Inference postprocessing",
    "stitch_tiles": "Stitch tiles",
    "create_out_dir": "Create output directory",
    "save_inference": "Save inference output",
    "save_mag1c": "Save Mag1c-SAS output",
    "save_valid_mask": "Save valid mask",
    "pad_image_32": "Padding to 32",
    "inference_image": "Model inference (full image)",
    "post_image": "Image postprocessing",
}

DATASETS = [
    {
        "name": "tiling",
        "power_csv": power_csv,
        "resource_csv": resource_usage_csv,
        "timestamps_csv": timestamps_csv,
    },
    {
        "name": "tiling_inference_only",
        "power_csv": power_csv_tiling_for_inference_only,
        "resource_csv": resource_usage_csv_tiling_for_inference_only,
        "timestamps_csv": timestamps_csv_tiling_for_inference_only,
    },
]


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stats(values: List[float]) -> Dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def _format(v: float | int) -> str:
    if isinstance(v, int):
        return str(v)
    return f"{v:.6f}"


def _verbose_action(action_name: str) -> str:
    return ACTION_VERBOSE_NAMES.get(action_name, action_name.replace("_", " ").title())


def _set_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _read_power_rows(csv_path: str) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _safe_float(row.get("timestamp", ""))
            pw = _safe_float(row.get("power_watts", ""))
            phase = row.get("phase", "")
            if ts is None or pw is None:
                continue
            rows.append({"timestamp": ts, "power_watts": pw, "phase": phase})
    rows.sort(key=lambda r: float(r["timestamp"]))
    return rows


def _read_resource_rows(csv_path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _safe_float(row.get("timestamp", ""))
            cpu = _safe_float(row.get("cpu_percent", ""))
            ram = _safe_float(row.get("ram_used_MB", ""))
            read_mb = _safe_float(row.get("proc_read_MB", ""))
            write_mb = _safe_float(row.get("proc_write_MB", ""))
            if None in (ts, cpu, ram, read_mb, write_mb):
                continue
            rows.append(
                {
                    "timestamp": ts,  # type: ignore[dict-item]
                    "cpu_percent": cpu,  # type: ignore[dict-item]
                    "ram_used_MB": ram,  # type: ignore[dict-item]
                    "proc_read_MB": read_mb,  # type: ignore[dict-item]
                    "proc_write_MB": write_mb,  # type: ignore[dict-item]
                }
            )
    rows.sort(key=lambda r: float(r["timestamp"]))
    return rows


def _read_timestamp_rows(csv_path: str) -> List[Dict[str, str | float]]:
    rows: List[Dict[str, str | float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _safe_float(row.get("timestamp", ""))
            if ts is None:
                continue
            rows.append(
                {
                    "id": row.get("id", ""),
                    "action_name": row.get("action_name", ""),
                    "marker": row.get("marker", ""),
                    "timestamp": ts,
                }
            )
    rows.sort(key=lambda r: float(r["timestamp"]))
    return rows


def _pair_timestamp_events(
    timestamp_rows: Iterable[Dict[str, str | float]],
    exclude_global: bool,
) -> Tuple[Dict[str, List[float]], List[str], Dict[str, List[Tuple[str, float, float]]]]:
    durations: Dict[str, List[float]] = {}
    action_order: List[str] = []
    action_seen = set()
    pending: Dict[Tuple[str, str], List[float]] = {}
    scene_windows: Dict[str, List[Tuple[str, float, float]]] = {}

    for row in timestamp_rows:
        scene_id = str(row["id"])
        action = str(row["action_name"])
        marker = str(row["marker"]).upper()
        ts = float(row["timestamp"])

        if exclude_global and scene_id == "Global":
            continue

        if action not in action_seen:
            action_seen.add(action)
            action_order.append(action)

        key = (scene_id, action)
        if marker == "START":
            pending.setdefault(key, []).append(ts)
        elif marker == "END" and pending.get(key):
            start_ts = pending[key].pop()
            if ts < start_ts:
                continue
            dur = ts - start_ts
            durations.setdefault(action, []).append(dur)
            scene_windows.setdefault(scene_id, []).append((action, start_ts, ts))

    return durations, action_order, scene_windows


def _power_statistics(power_rows: List[Dict[str, float | str]], dataset: str) -> List[Dict[str, str]]:
    by_phase: Dict[str, List[float]] = {phase: [] for phase in POWER_PHASES}

    for row in power_rows:
        phase = str(row["phase"])
        if phase in by_phase:
            by_phase[phase].append(float(row["power_watts"]))

    out: List[Dict[str, str]] = []
    for phase in POWER_PHASES:
        s = _stats(by_phase[phase])
        out.append(
            {
                "dataset": dataset,
                "source": "power",
                "group": phase,
                "group_verbose": phase.replace("_", " ").title(),
                "metric": "power_watts",
                "count": _format(s["count"]),
                "min": _format(s["min"]),
                "max": _format(s["max"]),
                "mean": _format(s["mean"]),
                "median": _format(s["median"]),
            }
        )
    return out


def _resource_statistics(resource_rows: List[Dict[str, float]], dataset: str) -> List[Dict[str, str]]:
    values: Dict[str, List[float]] = {v: [] for v in RESOURCE_COLUMNS.values()}

    for row in resource_rows:
        for col in RESOURCE_COLUMNS.values():
            values[col].append(float(row[col]))

    out: List[Dict[str, str]] = []
    for metric in RESOURCE_COLUMNS.values():
        s = _stats(values[metric])
        out.append(
            {
                "dataset": dataset,
                "source": "resource_usage",
                "group": "all",
                "group_verbose": "All samples",
                "metric": metric,
                "count": _format(s["count"]),
                "min": _format(s["min"]),
                "max": _format(s["max"]),
                "mean": _format(s["mean"]),
                "median": _format(s["median"]),
            }
        )
    return out


def _timestamp_statistics(
    dataset: str,
    durations: Dict[str, List[float]],
    action_order: List[str],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    ordered_actions = [a for a in action_order if a in durations]
    for action in ordered_actions:
        s = _stats(durations[action])
        out.append(
            {
                "dataset": dataset,
                "source": "timestamps",
                "group": action,
                "group_verbose": _verbose_action(action),
                "metric": "duration_seconds",
                "count": _format(s["count"]),
                "min": _format(s["min"]),
                "max": _format(s["max"]),
                "mean": _format(s["mean"]),
                "median": _format(s["median"]),
            }
        )
    return out


def _scene_duration_statistics(
    dataset: str,
    timestamp_rows: List[Dict[str, str | float]],
) -> List[Dict[str, str]]:
    # For each scene id, duration = last timestamp - first timestamp.
    per_scene_bounds = _scene_runtime_bounds(timestamp_rows)

    scene_durations: List[float] = []
    for t_min, t_max in per_scene_bounds.values():
        if t_max >= t_min:
            scene_durations.append(t_max - t_min)

    s = _stats(scene_durations)
    return [
        {
            "dataset": dataset,
            "source": "scene_duration",
            "group": "all_scenes",
            "group_verbose": "Per-scene total processing time",
            "metric": "scene_duration_seconds",
            "count": _format(s["count"]),
            "min": _format(s["min"]),
            "max": _format(s["max"]),
            "mean": _format(s["mean"]),
            "median": _format(s["median"]),
        }
    ]


def _scene_runtime_bounds(
    timestamp_rows: List[Dict[str, str | float]],
) -> Dict[str, Tuple[float, float]]:
    per_scene_bounds: Dict[str, Tuple[float, float]] = {}
    for row in timestamp_rows:
        scene_id = str(row["id"])
        if scene_id == "Global":
            continue
        ts = float(row["timestamp"])

        if scene_id not in per_scene_bounds:
            per_scene_bounds[scene_id] = (ts, ts)
        else:
            curr_min, curr_max = per_scene_bounds[scene_id]
            per_scene_bounds[scene_id] = (min(curr_min, ts), max(curr_max, ts))

    return per_scene_bounds


def _source_total_statistics(
    dataset: str,
    timestamp_rows: List[Dict[str, str | float]],
) -> List[Dict[str, str]]:
    if not timestamp_rows:
        total_duration = 0.0
    else:
        ts_values = [float(r["timestamp"]) for r in timestamp_rows]
        total_duration = max(ts_values) - min(ts_values)

    return [
        {
            "dataset": dataset,
            "source": "source_total",
            "group": "all_scenes_total",
            "group_verbose": "Total processing time over all scenes",
            "metric": "total_duration_seconds",
            "count": "1",
            "min": _format(total_duration),
            "max": _format(total_duration),
            "mean": _format(total_duration),
            "median": _format(total_duration),
        }
    ]


def _plot_action_pie(
    dataset: str,
    out_dir: Path,
    durations: Dict[str, List[float]],
    action_order: List[str],
) -> Path | None:
    ordered_actions = [a for a in action_order if a in durations and durations[a]]
    if not ordered_actions:
        return None

    totals = [sum(durations[a]) for a in ordered_actions]
    verbose_labels = [_verbose_action(a) for a in ordered_actions]
    total_sum = sum(totals)

    def _autopct_if_large(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= 1.0 else ""

    fig, ax = plt.subplots(figsize=(12, 8.5))
    wedges, _, autotexts = ax.pie(
        totals,
        labels=None,
        autopct=_autopct_if_large,
        startangle=90,
        textprops={"fontsize": TICK_FONTSIZE},
    )
    for t in autotexts:
        t.set_fontsize(TICK_FONTSIZE)
        t.set_weight("bold")
    ax.set_title(f"Action Time Share ({dataset})", fontsize=TITLE_FONTSIZE)
    ax.axis("equal")

    legend_labels = [f"{lbl} ({(val / total_sum) * 100:.1f}%)" for lbl, val in zip(verbose_labels, totals)]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
    )

    out_path = out_dir / f"pie_actions_{dataset}.pdf"
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    return out_path


def _to_relative(ts_values: List[float], t0: float) -> List[float]:
    return [t - t0 for t in ts_values]


def _plot_overall_timeseries(
    dataset: str,
    out_dir: Path,
    power_rows: List[Dict[str, float | str]],
    resource_rows: List[Dict[str, float]],
    timestamp_rows: List[Dict[str, str | float]],
    scene_windows: Dict[str, List[Tuple[str, float, float]]],
    action_order: List[str],
    global_action_color: Dict[str, tuple] | None = None,
) -> Path | None:
    if not power_rows or not resource_rows:
        return None

    t0_candidates = [float(power_rows[0]["timestamp"]), float(resource_rows[0]["timestamp"])]
    if timestamp_rows:
        t0_candidates.append(float(timestamp_rows[0]["timestamp"]))
    t0 = min(t0_candidates)

    p_t = _to_relative([float(r["timestamp"]) for r in power_rows], t0)
    p_y = [float(r["power_watts"]) for r in power_rows]
    r_t = _to_relative([float(r["timestamp"]) for r in resource_rows], t0)
    ram_y = [float(r["ram_used_MB"]) for r in resource_rows]
    cpu_y = [float(r["cpu_percent"]) for r in resource_rows]
    read_y = [float(r["proc_read_MB"]) for r in resource_rows]
    write_y = [float(r["proc_write_MB"]) for r in resource_rows]

    all_windows: List[Tuple[str, float, float]] = []
    for windows in scene_windows.values():
        all_windows.extend(windows)
    all_windows.sort(key=lambda x: x[1])

    ordered_actions = [a for a in action_order if any(w[0] == a for w in all_windows)]
    if global_action_color is not None:
        action_color = global_action_color
    else:
        action_palette = plt.get_cmap("tab20")
        action_color = {a: action_palette(i % 20) for i, a in enumerate(ordered_actions)}

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(16, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.65]},
    )

    axes[0].plot(p_t, p_y, color="tab:purple", lw=LINE_WIDTH)
    axes[0].set_ylabel("Power\n(W)", fontsize=LABEL_FONTSIZE + 3, fontweight="bold")
    axes[0].grid(alpha=GRID_ALPHA)
    _set_axis_style(axes[0])

    axes[1].plot(r_t, ram_y, color="tab:blue", lw=LINE_WIDTH)
    axes[1].set_ylabel("RAM\n(MB)", fontsize=LABEL_FONTSIZE + 3, fontweight="bold")
    axes[1].grid(alpha=GRID_ALPHA)
    _set_axis_style(axes[1])

    axes[2].plot(r_t, cpu_y, color="tab:green", lw=LINE_WIDTH)
    axes[2].set_ylabel("CPU\n(%)", fontsize=LABEL_FONTSIZE + 3, fontweight="bold")
    axes[2].grid(alpha=GRID_ALPHA)
    _set_axis_style(axes[2])

    axes[3].plot(r_t, read_y, color="tab:orange", lw=LINE_WIDTH, label="Disk read (MB)")
    axes[3].plot(r_t, write_y, color="tab:red", lw=LINE_WIDTH, label="Disk write (MB)")
    axes[3].set_ylabel("Disk\n(MB)", fontsize=LABEL_FONTSIZE + 3, fontweight="bold")
    axes[3].grid(alpha=GRID_ALPHA)
    axes[3].legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
    _set_axis_style(axes[3])

    axes[4].set_ylim(0, 1)
    axes[4].set_yticks([])
    axes[4].set_ylabel("Actions", fontsize=LABEL_FONTSIZE)
    axes[4].set_xlabel("Time From Start (s)", fontsize=LABEL_FONTSIZE)
    axes[4].grid(alpha=0.18)
    _set_axis_style(axes[4])

    for action, start, end in all_windows:
        rel_start = start - t0
        width = end - start
        axes[4].broken_barh([(rel_start, width)], (0, 1), facecolors=action_color[action], edgecolors="none")

    legend_handles = [Patch(facecolor=action_color[action], label=_verbose_action(action)) for action in ordered_actions]
    axes[4].legend(
        handles=legend_handles,
        ncol=4,
        fontsize=LEGEND_FONTSIZE,
        loc="lower center",
        #bbox_to_anchor=(0.55, -0.43),
        frameon=False,
    )

    fig.suptitle(f"Overall Resource Time Series ({dataset})", y=0.995, fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.98))

    out_path = out_dir / f"timeseries_overall_{dataset}.pdf"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _pick_median_scene(timestamp_rows: List[Dict[str, str | float]]) -> str | None:
    scene_totals: List[Tuple[str, float]] = []
    for scene_id, (t_min, t_max) in _scene_runtime_bounds(timestamp_rows).items():
        if t_max < t_min:
            continue
        scene_totals.append((scene_id, t_max - t_min))

    if not scene_totals:
        return None

    scene_totals.sort(key=lambda x: (x[1], x[0]))
    return scene_totals[len(scene_totals) // 2][0]


def _plot_median_scene_timeseries(
    dataset: str,
    out_dir: Path,
    power_rows: List[Dict[str, float | str]],
    resource_rows: List[Dict[str, float]],
    timestamp_rows: List[Dict[str, str | float]],
    scene_windows: Dict[str, List[Tuple[str, float, float]]],
    action_order: List[str],
    global_action_color: Dict[str, tuple] | None = None,
) -> Path | None:
    median_scene = _pick_median_scene(timestamp_rows)
    if median_scene is None:
        return None

    windows = sorted(scene_windows.get(median_scene, []), key=lambda x: x[1])
    if not windows:
        return None

    scene_start = min(start for _, start, _ in windows)
    scene_end = max(end for _, _, end in windows)

    p_scene = [r for r in power_rows if scene_start <= float(r["timestamp"]) <= scene_end]
    r_scene = [r for r in resource_rows if scene_start <= float(r["timestamp"]) <= scene_end]
    if not p_scene or not r_scene:
        return None

    ordered_actions = [a for a in action_order if any(w[0] == a for w in windows)]
    if global_action_color is not None:
        action_color = global_action_color
    else:
        action_palette = plt.get_cmap("tab20")
        action_color = {a: action_palette(i % 20) for i, a in enumerate(ordered_actions)}
    telemetry_color = "black" if BLACK_TELEMETRY else None

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(16, 14),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.6]},
    )

    p_t = _to_relative([float(r["timestamp"]) for r in p_scene], scene_start)
    p_y = [float(r["power_watts"]) for r in p_scene]
    r_t = _to_relative([float(r["timestamp"]) for r in r_scene], scene_start)
    ram_y = [float(r["ram_used_MB"]) for r in r_scene]
    cpu_y = [float(r["cpu_percent"]) for r in r_scene]
    read_y = [float(r["proc_read_MB"]) for r in r_scene]
    write_y = [float(r["proc_write_MB"]) for r in r_scene]
    disk_usage_y = [r + w for r, w in zip(read_y, write_y)]

    axes[0].plot(p_t, p_y, color=telemetry_color or "tab:purple", lw=LINE_WIDTH)
    axes[0].set_ylabel("Power\n[W]", fontsize=LABEL_FONTSIZE + 3, fontweight="bold")
    axes[0].grid(alpha=GRID_ALPHA)
    _set_axis_style(axes[0])

    axes[1].plot(r_t, ram_y, color=telemetry_color or "tab:blue", lw=LINE_WIDTH)
    axes[1].set_ylabel("RAM\n[MB]", fontsize=LABEL_FONTSIZE + 3, fontweight="bold")
    axes[1].grid(alpha=GRID_ALPHA)
    _set_axis_style(axes[1])

    axes[2].plot(r_t, cpu_y, color=telemetry_color or "tab:green", lw=LINE_WIDTH)
    axes[2].set_ylabel("CPU\n[%]", fontsize=LABEL_FONTSIZE + 3, fontweight="bold")
    axes[2].grid(alpha=GRID_ALPHA)
    _set_axis_style(axes[2])

    axes[3].plot(r_t, disk_usage_y, color=telemetry_color or "tab:orange", lw=LINE_WIDTH, label="Disk usage (MB)")
    axes[3].set_ylabel("Disk Usage\n[MB]", fontsize=LABEL_FONTSIZE + 3, fontweight="bold")
    axes[3].grid(alpha=GRID_ALPHA)
    _set_axis_style(axes[3])

    axes[4].set_ylim(0, 1)
    axes[4].set_yticks([])
    axes[4].set_ylabel("Action\n ", fontsize=LABEL_FONTSIZE + 3, fontweight="bold")
    axes[4].set_xlabel("Time From Scene Start (s)", fontsize=LABEL_FONTSIZE + 3, fontweight="bold")
    axes[4].grid(alpha=0.15)
    _set_axis_style(axes[4])

    for action, start, end in windows:
        rel_start = start - scene_start
        width = end - start
        axes[4].broken_barh([(rel_start, width)], (0, 1), facecolors=action_color[action], edgecolors="none")

    legend_handles = [Patch(facecolor=action_color[action], label=_verbose_action(action)) for action in ordered_actions]
    axes[4].legend(
        handles=legend_handles,
        ncol=3,
        fontsize=LEGEND_FONTSIZE + 2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.75),
        frameon=True,
    )

    fig.tight_layout(rect=(0.02, 0.14, 1, 1.0))

    out_path = out_dir / f"timeseries_median_scene_{dataset}.pdf"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    all_rows: List[Dict[str, str]] = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    produced_plots: List[Path] = []

    # Build a global action→color map so both graphs use the same colors per action.
    all_seen_actions: List[str] = []
    seen_set: set = set()
    for cfg in DATASETS:
        ts_rows = _read_timestamp_rows(str(cfg["timestamps_csv"]))
        _, action_order_tmp, _ = _pair_timestamp_events(ts_rows, exclude_global=True)
        for a in action_order_tmp:
            if a not in seen_set:
                seen_set.add(a)
                all_seen_actions.append(a)
    _global_palette = plt.get_cmap("tab20")
    global_action_color: Dict[str, tuple] = {a: _global_palette(i % 20) for i, a in enumerate(all_seen_actions)}

    for cfg in DATASETS:
        dataset = str(cfg["name"])
        power_rows = _read_power_rows(str(cfg["power_csv"]))
        resource_rows = _read_resource_rows(str(cfg["resource_csv"]))
        timestamp_rows = _read_timestamp_rows(str(cfg["timestamps_csv"]))
        durations, action_order, scene_windows = _pair_timestamp_events(timestamp_rows, exclude_global=True)

        all_rows.extend(_power_statistics(power_rows, dataset))
        all_rows.extend(_resource_statistics(resource_rows, dataset))
        all_rows.extend(_timestamp_statistics(dataset, durations, action_order))
        all_rows.extend(_scene_duration_statistics(dataset, timestamp_rows))
        all_rows.extend(_source_total_statistics(dataset, timestamp_rows))

        pie_path = _plot_action_pie(dataset, out_dir, durations, action_order)
        if pie_path is not None:
            produced_plots.append(pie_path)

        overall_path = _plot_overall_timeseries(
            dataset=dataset,
            out_dir=out_dir,
            power_rows=power_rows,
            resource_rows=resource_rows,
            timestamp_rows=timestamp_rows,
            scene_windows=scene_windows,
            action_order=action_order,
            global_action_color=global_action_color,
        )
        if overall_path is not None:
            produced_plots.append(overall_path)

        median_scene_path = _plot_median_scene_timeseries(
            dataset=dataset,
            out_dir=out_dir,
            power_rows=power_rows,
            resource_rows=resource_rows,
            timestamp_rows=timestamp_rows,
            scene_windows=scene_windows,
            action_order=action_order,
            global_action_color=global_action_color,
        )
        if median_scene_path is not None:
            produced_plots.append(median_scene_path)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["dataset", "source", "group", "group_verbose", "metric", "count", "min", "max", "mean", "median"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Statistics written to: {output_path}")
    print(f"Rows written: {len(all_rows)}")
    if produced_plots:
        print("Generated plots:")
        for p in produced_plots:
            print(f"- {p}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import string
import textwrap
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio


# =========================
# User Configuration
# =========================

# Dataset roots
DATASET_PATHS: Dict[str, Path] = {
    "evenly_spaced": Path("/home/jherec/emit-dataset/data_selected_50_evenly_spaced"),
    "starcop_filtering": Path("/home/jherec/emit-dataset/data_selected_50_starcop_filtering"),
    "highest_transmittance": Path("/home/jherec/emit-dataset/data_selected_50_highest_transmittance"),
}

# How dataset names are shown in plot titles
DATASET_VERBOSE: Dict[str, str] = {
    "evenly_spaced": "Evenly Spaced",
    "starcop_filtering": "Variance Increase",
    "highest_transmittance": "Highest Transmittance",
}

# Image rendering settings
RGB_PERCENTILE_LOW = 0.0
RGB_PERCENTILE_HIGH = 99.0
MAG1C_SAS_VMIN = 0.0
MAG1C_SAS_VMAX = 1750.0
INFERENCE_VMIN = 0.0
INFERENCE_VMAX = 1.0
INFERENCE_WHITE_THRESHOLD = 1.0

# Grid layout
N_COLUMNS = 4
FIGSIZE_PER_COL = 4.0
FIGSIZE_PER_ROW = 4.0
PANEL_TITLE_FONTSIZE = 15
ROW_LABEL_FONTSIZE = 13
ANNOTATION_FONTSIZE = 10
PANEL_TITLE_FONTWEIGHT = "normal"
ROW_LABEL_FONTWEIGHT = "normal"

# Output
OUTPUT_PDF = Path("/home/jherec/methane-filters-benchmark/benchmark/emit_deployment/inference_visualization.pdf")
DPI = 300


@dataclass(frozen=True)
class ProductSpec:
    key: str
    verbose_name: str
    filename: str
    cmap: str | None
    vmin: float | None
    vmax: float | None
    kind: str  # "rgb" | "npy" | "tif"


PRODUCT_SPECS: Dict[str, ProductSpec] = {
    "rgb": ProductSpec(
        key="rgb",
        verbose_name="RGB",
        filename="",
        cmap=None,
        vmin=None,
        vmax=None,
        kind="rgb",
    ),
    "inference_tiling": ProductSpec(
        key="inference_tiling",
        verbose_name="Linknet (RGB+Mag1c-SAS)",
        filename="inference_tiling.tif",
        cmap="viridis",
        vmin=INFERENCE_VMIN,
        vmax=INFERENCE_VMAX,
        kind="tif",
    ),
    "inference_tiling_inference_only": ProductSpec(
        key="inference_tiling_inference_only",
        verbose_name="Inference (tiled, inference only)",
        filename="inference_tiling_inference_only.tif",
        cmap="magma",
        vmin=INFERENCE_VMIN,
        vmax=INFERENCE_VMAX,
        kind="tif",
    ),
    "mag1c_sas": ProductSpec(
        key="mag1c_sas",
        verbose_name="Mag1c-SAS",
        filename="mag1c_sas.tif",
        cmap="viridis",
        vmin=MAG1C_SAS_VMIN,
        vmax=MAG1C_SAS_VMAX,
        kind="tif",
    ),
    "mag1c_sas_tiling": ProductSpec(
        key="mag1c_sas_tiling",
        verbose_name="Mag1c-SAS (tiled)",
        filename="mag1c_sas_tiling.tif",
        cmap="viridis",
        vmin=MAG1C_SAS_VMIN,
        vmax=MAG1C_SAS_VMAX,
        kind="tif",
    ),
    "label": ProductSpec(
        key="label",
        verbose_name="Label",
        filename="label.npy",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        kind="npy",
    ),
    # Optional product if present in a dataset scene (typically evenly_spaced only)
    "mag1c": ProductSpec(
        key="mag1c",
        verbose_name="Mag1c",
        filename="mag1c.npy",
        cmap="viridis",
        vmin=0.0,
        vmax=MAG1C_SAS_VMAX,
        kind="npy",
    ),
}


@dataclass(frozen=True)
class PanelRequest:
    product: str
    dataset: str


@dataclass(frozen=True)
class SceneRequest:
    scene_id: str
    panels: List[PanelRequest]


# -------------------------
# Edit this block only
#EMIT_L1B_RAD_001_20230215T094705_2304606_020
#EMIT_L1B_RAD_001_20230224T181429_2305512_036
# EMIT_L1B_RAD_001_20230502T042310_2312203_009
# bad EMIT_L1B_RAD_001_20230629T061850_2318004_025
# -------------------------
SCENE_REQUESTS: List[SceneRequest] = [
    SceneRequest(
        scene_id="EMIT_L1B_RAD_001_20231025T061531_2329804_012",
        panels=[
            PanelRequest("rgb", "starcop_filtering"),
            PanelRequest("mag1c_sas", "starcop_filtering"),
            PanelRequest("mag1c_sas_tiling", "starcop_filtering"),
            PanelRequest("inference_tiling", "starcop_filtering"),
        ],
    ),
    SceneRequest(
        scene_id="EMIT_L1B_RAD_001_20230824T070101_2323605_011",
        panels=[
            PanelRequest("rgb", "evenly_spaced"),
            PanelRequest("mag1c_sas", "evenly_spaced"),
            PanelRequest("mag1c_sas_tiling", "evenly_spaced"),
            PanelRequest("inference_tiling", "evenly_spaced"),
        ],
    ),
    SceneRequest(
        scene_id="EMIT_L1B_RAD_001_20230502T042310_2312203_009",
        panels=[
            PanelRequest("rgb", "highest_transmittance"),
            PanelRequest("mag1c_sas", "highest_transmittance"),
            PanelRequest("mag1c_sas_tiling", "highest_transmittance"),
            PanelRequest("inference_tiling", "highest_transmittance"),
        ],
    ),
    SceneRequest(
        scene_id="EMIT_L1B_RAD_001_20230629T061850_2318004_025",
        panels=[
            PanelRequest("rgb", "starcop_filtering"),
            PanelRequest("mag1c_sas", "starcop_filtering"),
            PanelRequest("mag1c_sas_tiling", "starcop_filtering"),
            PanelRequest("inference_tiling", "starcop_filtering"),
        ],
    ),
]


def _load_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1)
    return data.astype(np.float32)


def _load_npy(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim > 2:
        data = np.squeeze(data)
    return data.astype(np.float32)


def _stretch_rgb_percentile(rgb: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        channel = rgb[..., c]
        lo = np.percentile(channel, low_pct)
        hi = np.percentile(channel, high_pct)
        if hi <= lo:
            out[..., c] = np.clip(channel, 0.0, 1.0)
        else:
            out[..., c] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return out


def _load_rgb(scene_dir: Path) -> np.ndarray:
    r = _load_tif(scene_dir / "red.tif")
    g = _load_tif(scene_dir / "green.tif")
    b = _load_tif(scene_dir / "blue.tif")
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, None)  # Ensure no negative values
    return _stretch_rgb_percentile(rgb, RGB_PERCENTILE_LOW, RGB_PERCENTILE_HIGH)


def _panel_title(spec: ProductSpec, dataset_key: str) -> str:
    dataset_name = DATASET_VERBOSE[dataset_key]
    product_line = textwrap.fill(spec.verbose_name, width=24, break_long_words=False)
    if spec.key == "rgb":
        # RGB should not display the band-selection strategy.
        return product_line
    dataset_line = textwrap.fill(dataset_name, width=24, break_long_words=False)
    return f"{product_line}\n{dataset_line}"


def _validate_config() -> None:
    if N_COLUMNS <= 0:
        raise ValueError("N_COLUMNS must be > 0")

    for scene_req in SCENE_REQUESTS:
        for panel in scene_req.panels:
            if panel.product not in PRODUCT_SPECS:
                raise ValueError(f"Unknown product key: {panel.product}")
            if panel.dataset not in DATASET_PATHS:
                raise ValueError(f"Unknown dataset key: {panel.dataset}")


def _render_panel(ax: plt.Axes, scene_id: str, panel_req: PanelRequest) -> None:
    spec = PRODUCT_SPECS[panel_req.product]
    dataset_root = DATASET_PATHS[panel_req.dataset]
    scene_dir = dataset_root / scene_id

    ax.set_xticks([])
    ax.set_yticks([])

    if not scene_dir.exists():
        ax.text(0.5, 0.5, f"Missing scene:\n{scene_id}", ha="center", va="center", fontsize=ANNOTATION_FONTSIZE)
        ax.set_title(
            _panel_title(spec, panel_req.dataset),
            fontsize=PANEL_TITLE_FONTSIZE,
            fontweight=PANEL_TITLE_FONTWEIGHT,
        )
        return

    try:
        if spec.kind == "rgb":
            img = _load_rgb(scene_dir)
            ax.imshow(img)
        else:
            prod_path = scene_dir / spec.filename
            if not prod_path.exists():
                ax.text(0.5, 0.5, f"Missing file:\n{spec.filename}", ha="center", va="center", fontsize=ANNOTATION_FONTSIZE)
                ax.set_title(
                    _panel_title(spec, panel_req.dataset),
                    fontsize=PANEL_TITLE_FONTSIZE,
                    fontweight=PANEL_TITLE_FONTWEIGHT,
                )
                return

            if spec.kind == "npy":
                img = _load_npy(prod_path)
            else:
                img = _load_tif(prod_path)

            ax.imshow(img, cmap=spec.cmap, vmin=spec.vmin, vmax=spec.vmax)

            # Highlight confident inference pixels in white.
            if panel_req.product in {"inference_tiling", "inference_tiling_inference_only"}:
                mask = img > INFERENCE_WHITE_THRESHOLD
                if np.any(mask):
                    overlay = np.zeros((*img.shape, 4), dtype=np.float32)
                    overlay[..., :3] = 1.0  # white RGB
                    overlay[..., 3] = mask.astype(np.float32)  # alpha only where mask is True
                    ax.imshow(overlay)
    except Exception as exc:  # noqa: BLE001
        ax.text(0.5, 0.5, f"Error:\n{exc}", ha="center", va="center", fontsize=ANNOTATION_FONTSIZE, color="crimson")

    ax.set_title(
        _panel_title(spec, panel_req.dataset),
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight=PANEL_TITLE_FONTWEIGHT,
    )


def _row_label(row_idx: int) -> str:
    letters = string.ascii_lowercase
    if row_idx < len(letters):
        return f"{letters[row_idx]})"
    return f"{row_idx + 1})"


def build_visualization() -> Path:
    _validate_config()

    n_rows = len(SCENE_REQUESTS)
    n_cols = N_COLUMNS
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * FIGSIZE_PER_COL, n_rows * FIGSIZE_PER_ROW),
        squeeze=False,
    )

    for row_idx, scene_req in enumerate(SCENE_REQUESTS):
        row_panels = scene_req.panels[:n_cols]

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(row_panels):
                _render_panel(ax, scene_req.scene_id, row_panels[col_idx])
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "No panel", ha="center", va="center", fontsize=ANNOTATION_FONTSIZE, color="gray")

            if col_idx == 0:
                ax.set_ylabel(
                    _row_label(row_idx),
                    fontsize=ROW_LABEL_FONTSIZE,
                    fontweight=ROW_LABEL_FONTWEIGHT,
                    rotation=0,
                    labelpad=20,
                    va="center",
                )

    fig.tight_layout(rect=(0.02, 0.01, 1.0, 0.99))

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PDF


def main() -> None:
    out_path = build_visualization()
    print(f"Visualization written to: {out_path}")


if __name__ == "__main__":
    main()

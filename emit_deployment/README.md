# EMIT Deployment Pipeline

## Prerequisites

The deployment and measurement scripts depend on the `onboard-methane-detection` library. Install it first:

```bash
pip install onboard-methane-detection
```

## 1. Prepare EMIT Data

Generate the band-selected dataset from raw EMIT `.nc` files:

```bash
python3 prepare_emit_data.py \
    --csv-path /home/jherec/emit-dataset/dataset_info.csv \
    --data-dir /home/jherec/emit-dataset/data/data \
    --output-dir /home/jherec/emit-dataset/data_selected_50_highest_transmittance \
    --num-bands 50 \
    --strategy highest-transmittance
```

This reads the raw NetCDF scenes, selects bands according to the chosen strategy (`highest-transmittance`, `highest-variance`, or `evenly-spaced`), generates the methane spectrum, and exports per-band GeoTIFFs along with RGB bands and labels.

## 2. Run Inference

Run the deployment script directly to process the prepared data:

```bash
python3 deployment_script.py \
    --data-dir /tmp/emmc/jonas/TGRS/data_selected_50_variance \
    --spectrum-path /tmp/emmc/jonas/TGRS/data_selected_50_variance/emit_50_selected_methane_spectrum.npy \
    --output-dir /tmp/emmc/jonas/TGRS/data_selected_50_variance \
    --tile-for-inference \
    --tile-for-mag1c-sas \
    --save-mag1c-sas-output \
    --save-valid-mask \
    --mark-timestamps
```

The `--mark-timestamps` flag records start/end timestamps for each processing step (loading, mag1c, normalization, inference, stitching, saving) and writes them to a CSV with per-action timing statistics.

## 3. Measure Resources and Power

The measurement scripts are layered -- you only call `measure_power.py`, which internally calls `measure_resources.py`, which in turn calls `deployment_script.py`:

```
measure_power.py  -->  measure_resources.py  -->  deployment_script.py
  (power W)              (CPU, RAM, disk I/O)       (per-step timestamps)
```

### Tiling for both inference and mag1c SAS:

```bash
python3 measure_power.py \
    -o power_tiling.csv \
    -c "python3 measure_resources.py --tile-for-inference --tile-for-mag1c-sas --csv resource_usage_tiling.csv"
```

### Tiling for inference only:

```bash
python3 measure_power.py \
    -o power_tiling_inference_only.csv \
    -c "python3 measure_resources.py --tile-for-inference --csv resource_usage_tiling_inference_only.csv"
```

### No tiling:

Note: This runs out of memory on Q8J with 4GB RAM.

```bash
python3 measure_power.py \
    -o power.csv \
    -c "python3 measure_resources.py --csv resource_usage.csv"
```

Each run produces three output files:
- **power CSV** -- power consumption over time (idle before, running, idle after)
- **resource usage CSV** -- CPU%, RAM usage, disk read/write per sample
- **timestamps CSV** -- per-step timing from `deployment_script.py`

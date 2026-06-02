import csv
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil


# If True, store measurements in memory and write at the end
STORE_IN_MEMORY = True


process = None


def cleanup(*args, **kwargs):
    print("\nTerminating inference process and all subprocesses...\n")

    global process

    if process is not None:
        try:
            proc = psutil.Process(process.pid)

            for child in proc.children(recursive=True):
                child.terminate()
                child.wait()

            process.terminate()
            process.wait()
        except psutil.NoSuchProcess:
            pass

    sys.exit(0)


process = None


def _sum_process_io(proc):
    """Return total (read_bytes, write_bytes) for proc and recursive children."""
    try:
        if not proc.is_running():
            return 0, 0
        procs = [proc] + proc.children(recursive=True)
        r = 0
        w = 0
        for p in procs:
            try:
                io = p.io_counters()
                r += getattr(io, "read_bytes", 0)
                w += getattr(io, "write_bytes", 0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return r, w
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, 0


def run_inference_and_monitor(command, csv_file="resource_usage.csv"):
    """Run a command, monitor CPU, RAM, and per-process disk usage, save CSV, and print statistics."""

    global process
    process = subprocess.Popen(command)

    csv_path = Path(csv_file)
    header = ["timestamp", "cpu_percent", "ram_used_MB", "proc_read_MB", "proc_write_MB"]
    if not csv_path.exists() and not STORE_IN_MEMORY:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    measurements = [] if STORE_IN_MEMORY else None

    ram = psutil.virtual_memory()
    beginning_ram = float(ram.used)

    # psutil.Process for subprocess
    try:
        proc = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        proc = None

    # initial disk counters for this process
    prev_read, prev_write = (0, 0)
    if proc is not None:
        prev_read, prev_write = _sum_process_io(proc)

    while True:
        # CPU and RAM
        cpu_usage = psutil.cpu_percent(interval=0.1, percpu=False)
        ram = psutil.virtual_memory()

        # per-process disk I/O
        curr_read, curr_write = (0, 0)
        if proc is not None:
            curr_read, curr_write = _sum_process_io(proc)

        # delta since last sample
        read_mb = max((curr_read - prev_read) / (1024 * 1024), 0)
        write_mb = max((curr_write - prev_write) / (1024 * 1024), 0)
        prev_read, prev_write = curr_read, curr_write

        row = [
            time.time(),
            float(cpu_usage),
            ram.used / (1024 * 1024),
            read_mb,
            write_mb,
        ]
        if STORE_IN_MEMORY:
            measurements.append(row)
        else:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

        # break if process has ended
        if process.poll() is not None:
            if proc is not None:
                children = []
                try:
                    children = proc.children(recursive=True)
                except psutil.NoSuchProcess:
                    pass
                if not children:
                    break
            else:
                break

    # ======================
    # Compute statistics
    # ======================
    if STORE_IN_MEMORY:
        # Write all measurements at the end
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(measurements)
        rows = measurements
    else:
        data = list(csv.reader(open(csv_path)))
        rows = data[1:]

    cpu_array = np.array([float(row[1]) for row in rows])
    ram_used = np.array([float(row[2]) for row in rows])
    disk_read = np.array([float(row[3]) for row in rows])
    disk_write = np.array([float(row[4]) for row in rows])

    print(
        f"CPU: min={cpu_array.min():.1f}%, max={cpu_array.max():.1f}%, "
        f"median={np.median(cpu_array):.1f}%, mean={cpu_array.mean():.1f}%"
    )
    print(
        f"RAM used: min={ram_used.min():.2f}MB, max={ram_used.max():.2f}MB, "
        f"median={np.median(ram_used):.2f}MB, mean={ram_used.mean():.2f}MB"
    )
    print(
        f"Disk read MB: min={disk_read.min():.2f}, max={disk_read.max():.2f}, "
        f"median={np.median(disk_read):.2f}, mean={disk_read.mean():.2f}"
    )
    print(
        f"Disk write MB: min={disk_write.min():.2f}, max={disk_write.max():.2f}, "
        f"median={np.median(disk_write):.2f}, mean={disk_write.mean():.2f}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deployment and measure resource usage")
    parser.add_argument(
        "--tile-for-inference",
        action="store_true",
        default=False,
        help="Enable tiling for inference",
    )
    parser.add_argument(
        "--tile-for-mag1c-sas",
        action="store_true",
        default=False,
        help="Enable tiling for mag1c SAS",
    )
    parser.add_argument(
        "--csv",
        dest="csv_file",
        default="resource_usage.csv",
        help="CSV file to write resource measurements to",
    )

    args = parser.parse_args()

    command = [
        "python3",
        "deployment_script.py",
        "--data-dir",
        "/tmp/emmc/jonas/TGRS/data_selected_50",
        "--spectrum-path",
        "/tmp/emmc/jonas/TGRS/data_selected_50/emit_50_selected_methane_spectrum.npy",
        "--output-dir",
        "/tmp/emmc/jonas/TGRS/data_selected_50",
        "--mark-timestamps",
    ]

    if args.tile_for_inference:
        command += ["--tile-for-inference"]
    if args.tile_for_mag1c_sas:
        command += ["--tile-for-mag1c-sas"]

    run_inference_and_monitor(command, csv_file=args.csv_file)
    cleanup()

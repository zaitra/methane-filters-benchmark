#!/usr/bin/env python3
"""
Power monitoring script for Q8

Tiling for both inference and mag1c SAS:
python3 measure_power.py -o power_tiling.csv -c "python3 measure_resources.py --tile-for-inference --tile-for-mag1c-sas --csv resource_usage_tiling.csv"

Tiling for inference only:
python3 measure_power.py -o power_tiling_inference_only.csv -c "python3 measure_resources.py --tile-for-inference --csv resource_usage_tiling_inference_only.csv"

No tiling:
python3 measure_power.py -o power.csv -c "python3 measure_resources.py --csv resource_usage.csv"
"""
import re
import subprocess
import sys
import threading
import time
from datetime import datetime


def read_power():
    """
    Read power from the board using sensors.
    Returns:
        float: power in Watts, or None if not found
    """
    # Run sensors on the correct adapter
    result = subprocess.run(
        ["sensors", "iio_hwmon_adc0-isa-0000"], capture_output=True, text=True, timeout=5
    )

    vcc_in = None
    cur_vcc_in = None

    for line in result.stdout.splitlines():
        # Voltage (vcc_in)
        match_v = re.search(r"vcc_in:\s*([\d.]+)\s*V", line)
        if match_v:
            vcc_in = float(match_v.group(1))

        # Current (cur_vcc_in)
        match_i = re.search(r"cur_vcc_in:\s*([\d.]+)\s*mA", line)
        if match_i:
            cur_vcc_in = float(match_i.group(1))

    if vcc_in is not None and cur_vcc_in is not None:
        return vcc_in * (cur_vcc_in / 1000.0)  # convert mA → A

    # Debug info if parsing fails
    print("Could not read power from sensors. Output:")
    print(result.stdout)
    return None


def get_power_value():
    """
    Get power1 value from ina226 sensor
    Returns: float value in Watts, or None if not found
    """
    try:
        return read_power()
    except Exception as e:
        print(f"Error reading sensor: {e}")
        return None


def monitor_power(interval=1, output_file=None, stop_event=None):
    """
    Monitor power consumption at specified interval

    Args:
        interval: Seconds between measurements (default: 1)
        output_file: Optional file path to log data (CSV format)
        stop_event: Threading event to signal when to stop monitoring
    """
    #print("Power Monitoring Started (Ctrl+C to stop)")
    #print("=" * 50)
    #print(f"{'Timestamp':<20} {'Power (W)':<12}")
    #print("-" * 50)

    # Open log file if specified
    log_file = None
    if output_file:
        log_file = open(output_file, "w")
        log_file.write("timestamp,power_watts\n")

    try:
        while True:
            if stop_event and stop_event.is_set():
                break

            timestamp = time.time()
            power = get_power_value()

            if power is not None:
                # Display to console
                # print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} {power:<12.2f}")

                # Write to log file
                if log_file:
                    log_file.write(f"{timestamp},{power:.2f}\n")
                    log_file.flush()
            else:
                pass
                # print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} {'N/A':<12}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("Monitoring stopped")
    finally:
        if log_file:
            log_file.close()
            print(f"Data saved to: {output_file}")


def measure_idle_power(duration, interval=1):
    """
    Measure idle power for specified duration

    Args:
        duration: Duration in seconds to measure
        interval: Seconds between measurements

    Returns:
        List of power measurements
    """
    measurements = []
    end_time = time.time() + duration

    while time.time() < end_time:
        timestamp = time.time()
        power = get_power_value()

        if power is not None:
            measurements.append({"timestamp": timestamp, "power": power})
            # print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} {power:<12.2f}")

        time.sleep(interval)

    return measurements


def run_command_with_monitoring(
    command, interval=0.01, output_file=None, idle_duration=3, num_runs=1
):
    """
    Run a command while monitoring power consumption
    Measures idle power before and after command execution

    Args:
        command: Command string or list to execute
        interval: Seconds between power measurements (default: 1)
        output_file: Optional file path to log power data (CSV format)
        idle_duration: Seconds to measure idle power before/after (default: 3)
        num_runs: Number of times to run the command (default: 1)
    """
    all_measurements = []

    # Measure idle power BEFORE
    #print("=" * 50)
    #print(f"Measuring IDLE power for {idle_duration} seconds (BEFORE)")
    #print("=" * 50)
    #print(f"{'Timestamp':<20} {'Power (W)':<12}")
    #print("-" * 50)

    idle_before = measure_idle_power(idle_duration, interval)
    all_measurements.extend([{**m, "phase": "idle_before"} for m in idle_before])

    if idle_before:
        avg_idle_before = sum(m["power"] for m in idle_before) / len(idle_before)
        #print(f"\nAverage idle power (before): {avg_idle_before:.3f} W\n")

    # Run command with monitoring
    #print("=" * 50)
    if num_runs > 1:
        #print(
        #    f"Running command {num_runs} times: {command if isinstance(command, str) else ' '.join(command)}"
        #)
        pass
    else:
        #print(f"Running command: {command if isinstance(command, str) else ' '.join(command)}")
        pass
    #print("=" * 50)
    #print(f"{'Timestamp':<20} {'Power (W)':<12}")
    #print("-" * 50)

    # Create stop event for monitoring thread
    stop_event = threading.Event()
    command_measurements = []

    def monitor_with_storage():
        """Monitor power and store measurements"""
        while not stop_event.is_set():
            timestamp = time.time()
            power = get_power_value()

            if power is not None:
                measurement = {"timestamp": timestamp, "power": power, "phase": "running"}
                command_measurements.append(measurement)
                # print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} {power:<12.2f}")

            time.sleep(interval)

    # Start power monitoring in separate thread
    monitor_thread = threading.Thread(target=monitor_with_storage)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Give monitor a moment to start
    time.sleep(0.5)

    # Run the command num_runs times
    command_success = False
    try:
        for run in range(1, num_runs + 1):
            if num_runs > 1:
                #print(f"\n--- Run {run} of {num_runs} ---")
                pass

            if isinstance(command, str):
                result = subprocess.run(command, shell=True)
            else:
                result = subprocess.run(command)

            if result.returncode != 0:
                #print(f"Warning: Run {run} exited with code {result.returncode}")
                pass

        command_success = True
        #print()
        #print("=" * 50)
        if num_runs > 1:
            #print(f"Completed {num_runs} runs")
            pass
        else:
            #print(f"Command completed with exit code: {result.returncode}")
            pass
        #print("=" * 50)

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("Command interrupted by user")
        print("=" * 50)
    except Exception as e:
        print(f"\nError running command: {e}")
    finally:
        # Stop monitoring
        stop_event.set()
        monitor_thread.join(timeout=2)
        all_measurements.extend(command_measurements)

    # Measure idle power AFTER
    #print()
    #print("=" * 50)
    #print(f"Measuring IDLE power for {idle_duration} seconds (AFTER)")
    #print("=" * 50)
    #print(f"{'Timestamp':<20} {'Power (W)':<12}")
    #print("-" * 50)

    idle_after = measure_idle_power(idle_duration, interval)
    all_measurements.extend([{**m, "phase": "idle_after"} for m in idle_after])

    if idle_after:
        avg_idle_after = sum(m["power"] for m in idle_after) / len(idle_after)
        #print(f"\nAverage idle power (after): {avg_idle_after:.3f} W\n")

    # Print summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if idle_before:
        avg_idle_before = sum(m["power"] for m in idle_before) / len(idle_before)
        print(f"Idle power (before):     {avg_idle_before:.3f} W")

    if command_measurements:
        avg_running = sum(m["power"] for m in command_measurements) / len(command_measurements)
        print(f"Running power (average): {avg_running:.3f} W")
        if idle_before:
            print(
                f"Power increase:          {avg_running - avg_idle_before:.3f} W ({((avg_running/avg_idle_before - 1) * 100):.1f}%)"
            )

    if idle_after:
        avg_idle_after = sum(m["power"] for m in idle_after) / len(idle_after)
        print(f"Idle power (after):      {avg_idle_after:.3f} W")

    print("=" * 50)

    # Save all measurements to CSV if requested
    if output_file and all_measurements:
        with open(output_file, "w") as f:
            f.write("timestamp,power_watts,phase\n")
            for m in all_measurements:
                f.write(f"{m['timestamp']},{m['power']:.2f},{m['phase']}\n")
        print(f"\nData saved to: {output_file}")

    print("\nPower monitoring completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor power consumption on ZCU104")
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=0.1,
        help="Measurement interval in seconds (default: 0.1)",
    )
    parser.add_argument("-o", "--output", type=str, help="Output CSV file for logging data")
    parser.add_argument(
        "-c",
        "--command",
        type=str,
        help='Command to run while monitoring power (e.g., "xdputil benchmark model.xmodel 1")',
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=1,
        help="Number of times to run the command (default: 1)",
    )
    parser.add_argument(
        "--idle-duration",
        type=float,
        default=3.0,
        help="Duration in seconds to measure idle power before/after command (default: 3)",
    )

    args = parser.parse_args()

    if args.command:
        # Run command with power monitoring and idle measurements
        run_command_with_monitoring(
            args.command,
            interval=args.interval,
            output_file=args.output,
            idle_duration=args.idle_duration,
            num_runs=args.num_runs,
        )
    else:
        # Continuous monitoring mode
        monitor_power(interval=args.interval, output_file=args.output)

#!/usr/bin/env python3
"""Validate and measure the OpenVINO CPU pipeline on a single host."""

import argparse
import csv
import hashlib
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ("banana.bmp", "dog.bmp")
ORDERS = ((1, 2, 4, 8), (2, 8, 1, 4), (4, 1, 8, 2), (8, 4, 2, 1))
WARMUP_REPEATS = 50
REPEATS = 1000
VALIDATION_JOBS = (1, 33, REPEATS * len(IMAGES))
THREADS = 8


def run(args, name, requests, jobs, top10=None):
    cmd = [str(args.binary), str(args.model), str(args.images), "CPU", str(THREADS),
           f"jobs:{jobs}", str(requests), str(WARMUP_REPEATS)]
    if top10 is not None:
        cmd.append(str(top10))
    with (args.output / f"{name}.stdout").open("w") as stdout, (
        args.output / f"{name}.stderr"
    ).open("w") as stderr:
        run_result = subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True, cwd=ROOT)

    log = (args.output / f"{name}.stdout").read_text(encoding="utf-8")
    if run_result.returncode != 0:
        raise ValueError(f"{name} exited {run_result.returncode}")
    properties = {
        key.strip(): value.strip() for key, value in
        (line.split(":", 1) for line in log.splitlines() if ":" in line)
    }
    for key, expected in {"INFERENCE_NUM_THREADS": str(THREADS), "NUM_STREAMS": "1",
                          "PERFORMANCE_HINT": "LATENCY"}.items():
        if properties.get(key) != expected:
            raise ValueError(f"{name}: {key} is {properties.get(key)!r}, expected {expected!r}")

    rows = [line.split(",")[1:] for line in log.splitlines() if line.startswith("P1_RESULT,")]
    if len(rows) != 1 or len(rows[0]) != 7:
        raise ValueError(f"{name}: missing or malformed P1_RESULT")
    mode, warmup, expected, submitted, completed = map(int, rows[0][:5])
    elapsed, cpu = map(float, rows[0][5:])
    if (mode, warmup, expected, submitted, completed) != (
        requests, WARMUP_REPEATS * len(IMAGES), jobs, jobs, jobs
    ):
        raise ValueError(f"{name}: unexpected request count or job totals")
    if not (math.isfinite(elapsed) and elapsed > 0 and math.isfinite(cpu) and cpu >= 0):
        raise ValueError(f"{name}: invalid timing data")
    return (expected, submitted, completed, elapsed, cpu, completed / elapsed, cpu / elapsed)


def read_top10(path, jobs):
    results = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["job_id", "image", "rank", "class_id", "score"]:
            raise ValueError(f"{path.name}: unexpected CSV header")
        for row in reader:
            job, rank, class_id = (int(row[key]) for key in ("job_id", "rank", "class_id"))
            score = float(row["score"])
            key = (job, rank)
            if not (0 <= job < jobs and 1 <= rank <= 10 and class_id >= 0 and math.isfinite(score)):
                raise ValueError(f"{path.name}: invalid job, rank, class, or score at {key}")
            if row["image"] != IMAGES[job % len(IMAGES)] or key in results:
                raise ValueError(f"{path.name}: wrong image or duplicate row at {key}")
            results[key] = (class_id, score)
    if len(results) != jobs * 10:
        raise ValueError(f"{path.name}: expected {jobs * 10} unique Top-10 rows, got {len(results)}")
    return results


def validate(args):
    with (args.output / "validation.csv").open("w", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(("jobs", "requests", "top10_rows", "exact_score_matches", "max_abs_error"))
        for jobs in VALIDATION_JOBS:
            reference = None
            for requests in ORDERS[0]:
                name = f"verify-{jobs:04d}-requests-{requests}"
                top10 = args.output / f"{name}.csv"
                run(args, name, requests, jobs, top10)
                results = read_top10(top10, jobs)
                if requests == 1:
                    reference = results
                exact = 0
                max_error = 0.0
                for key, (ref_class, ref_score) in reference.items():
                    class_id, score = results[key]
                    error = abs(score - ref_score)
                    if class_id != ref_class or error > 1e-6 + 1e-5 * abs(ref_score):
                        raise ValueError(f"{name}: job/rank {key} got {class_id}/{score}, "
                                         f"expected {ref_class}/{ref_score}")
                    exact += score == ref_score
                    max_error = max(max_error, error)
                writer.writerow((jobs, requests, len(results), exact, max_error))
                report.flush()


def measure(args):
    samples = {mode: [] for mode in ORDERS[0]}
    with (args.output / "runs.csv").open("w", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(("round", "position", "requests", "expected", "submitted", "completed",
                         "elapsed_s", "cpu_s", "images_per_s", "cpu_cores", "exit_code"))
        for round_number, order in enumerate(ORDERS * 2, 1):
            for position, requests in enumerate(order, 1):
                name = f"round-{round_number:02d}-requests-{requests}"
                result = run(args, name, requests, REPEATS * len(IMAGES))
                writer.writerow((round_number, position, requests, *result, 0))
                report.flush()
                samples[requests].append(result)

    baseline = statistics.mean(result[-2] for result in samples[1])
    rows = []
    with (args.output / "summary.csv").open("w", newline="") as summary:
        writer = csv.writer(summary)
        columns = ("Mode", "Runs", "Mean img/s", "SD img/s", "CV %", "Mean CPU cores",
                   "vs. sequential %", "Top-10")
        writer.writerow(columns)
        rows.append(columns)
        for requests, results in samples.items():
            throughput = [result[-2] for result in results]
            mean = statistics.mean(throughput)
            deviation = statistics.stdev(throughput)
            row = ("Sequential" if requests == 1 else f"Async {requests}", len(results),
                   f"{mean:.3f}", f"{deviation:.3f}", f"{100 * deviation / mean:.2f}",
                   f"{statistics.mean(result[-1] for result in results):.3f}",
                   f"{100 * (mean / baseline - 1):+.2f}", "Pass")
            writer.writerow(row)
            rows.append(row)

    rows.insert(1, ("---",) * len(rows[0]))
    markdown = "".join("| " + " | ".join(map(str, row)) + " |\n" for row in rows)
    (args.output / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")


def command_output(*cmd):
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def save_environment(args):
    paths = (args.binary, args.model, *(args.images / name for name in IMAGES))
    cpu_info = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    cpu_model = re.search(r"^model name\s*:\s*(.+)$", cpu_info, re.MULTILINE)
    metadata = [
        f"commit={command_output('git', 'rev-parse', 'HEAD')}",
        f"platform={platform.platform()}",
        f"cpu={cpu_model[1] if cpu_model else platform.machine()}",
        f"power_mode_operator_confirmed={args.power_mode}",
        "build=Release, OpenCV disabled (see documented build command)",
        "inference_threads=8",
        "performance_hint=LATENCY",
        "num_streams=1",
        "per_image_performance_log=off",
        "version_and_resolved_properties=per-run stdout",
        f"runner_command={sys.argv!r}",
        "tracked_worktree_status:",
        command_output("git", "status", "--porcelain", "--untracked-files=no"),
        "sha256:",
        *(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}" for path in paths),
    ]
    (args.output / "environment.txt").write_text("\n".join(metadata) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--model", type=Path, default=ROOT / "models/squeezenet1.1-7.onnx")
    parser.add_argument("--images", type=Path, default=ROOT / "data/images")
    parser.add_argument("--power-mode", choices=("ac-balanced",), required=True,
                        help="confirm Windows is on AC power and in Balanced mode")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for name in ("binary", "output", "model", "images"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        save_environment(args)
        validate(args)
        if not args.verify_only:
            measure(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"P1 stopped: {error}. See {args.output} for retained logs.\n")


if __name__ == "__main__":
    main()
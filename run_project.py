import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_FOLDER = ROOT / "logs"
HISTORY_FILE = LOG_FOLDER / "run_history.csv"

# The scripts, in the order they must run. Later scripts read files that
# earlier scripts write (for example Triage.py reads customer_segments.csv).
STEPS = [
    "generate_data.py",
    "src/Part_A/Data_cleaning.py",
    "src/Part_A/Data_validation.py",
    "src/Part_A/Sql_analysis.py",
    "src/Part_A/Pandas_analysis.py",
    "src/Part_B/Segments.py",
    "src/Part_B/Lapse.py",
    "src/Part_C/Triage.py",
    "src/Part_D/Policy_rag.py",
    "src/Part_D/Refund_agent.py",
]


# Prints one line on screen and writes the same line to the log file.
def emit(log, text=""):
    print(text, flush=True)
    log.write(text + "\n")
    log.flush()


# Runs one script, copies everything it prints to the screen and the log,
# and measures how long it took. Returns (status, seconds).
def run_step(number, script, log):
    emit(log, f"\n{'=' * 70}")
    emit(log, f"STEP {number}/{len(STEPS)}: python {script}")
    emit(log, "=" * 70)

    started_at = datetime.now()
    started = time.perf_counter()
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    process = subprocess.Popen(
        [sys.executable, script], cwd=ROOT, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")
    for line in process.stdout:
        emit(log, line.rstrip("\n"))
    exit_code = process.wait()

    seconds = time.perf_counter() - started
    status = "OK" if exit_code == 0 else f"FAILED (exit code {exit_code})"
    emit(log, f"-- {status} | started {started_at:%H:%M:%S} | took {seconds:.1f}s")
    return status, seconds


# Adds one row for this run to logs/run_history.csv (creates the file the first time).
def add_to_history(run_id, started_at, results, total_seconds, outcome, log_path):
    is_new = not HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["run_id", "started", "steps_run", "steps_ok",
                             "total_seconds", "outcome", "log_file"])
        steps_ok = sum(1 for _, status, _ in results if status == "OK")
        writer.writerow([run_id, started_at.isoformat(timespec="seconds"), len(results),
                         steps_ok, f"{total_seconds:.1f}", outcome, log_path.name])


# Runs every step in order, stops at the first failure, writes the log and the history.
def main():
    LOG_FOLDER.mkdir(exist_ok=True)
    started_at = datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_FOLDER / f"run_{run_id}.log"
    results = []
    run_started = time.perf_counter()

    with open(log_path, "w", encoding="utf-8") as log:
        emit(log, f"RUN {run_id}")
        emit(log, f"started: {started_at:%Y-%m-%d %H:%M:%S}")
        emit(log, f"python:  {sys.version.split()[0]} ({sys.executable})")

        for number, script in enumerate(STEPS, start=1):
            status, seconds = run_step(number, script, log)
            results.append((script, status, seconds))
            if status != "OK":
                break

        total_seconds = time.perf_counter() - run_started
        finished = len(results) == len(STEPS) and all(s == "OK" for _, s, _ in results)
        outcome = ("ALL STEPS FINISHED" if finished
                   else f"STOPPED AT STEP {len(results)}: {results[-1][0]}")

        emit(log, f"\n{'=' * 70}")
        emit(log, "SUMMARY")
        emit(log, "=" * 70)
        for number, (script, status, seconds) in enumerate(results, start=1):
            emit(log, f"{number:>2}  {script:<34} {status:<22} {seconds:>6.1f}s")
        emit(log, f"total time: {total_seconds:.1f}s")
        emit(log, outcome)

    add_to_history(run_id, started_at, results, total_seconds, outcome, log_path)
    print(f"log written to {log_path.relative_to(ROOT)}")
    sys.exit(0 if finished else 1)


# Start here.
if __name__ == "__main__":
    main()
"""24/7 Autonomous Orchestrator for PLE Agent Harness on Oracle OCI VM (Phase 8)."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from src.evolution.gemini_evolver import evolve_verifiers_once


ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
STATE_PATH = ROOT_DIR / "artifacts" / "autonomous_state.json"
SUBMISSION_LOG_PATH = ROOT_DIR / "docs" / "submission_log.csv"
LOG_PATH = ROOT_DIR / "logs" / "orchestrator.log"

COMPETITION = "physical-engineerings-last-exam"
DAILY_LIMIT = 5


def load_dotenv(path: Path) -> None:
    """Load key=value pairs from .env into os.environ without external deps."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            os.environ.setdefault(k, v)


def log_msg(msg: str) -> None:
    """Append timestamped line to stdout and logs/orchestrator.log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_unit_tests() -> tuple[bool, str]:
    """Run deterministic unit tests for metric and engineering verifiers."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=60,
        )
        ok = proc.returncode == 0
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-5:])
        return ok, tail
    except Exception as exc:
        return False, f"unittest_error: {exc}"


def poll_kaggle_competition() -> Dict[str, Any]:
    """Query Kaggle CLI for competition kernels & submissions without blocking."""
    info: Dict[str, Any] = {
        "competition": COMPETITION,
        "kernels_top_votes": 0,
        "submissions_today_utc": 0,
        "status": "ok",
    }
    kaggle_bin = os.environ.get("KAGGLE_BIN", "kaggle")
    try:
        k_proc = subprocess.run(
            [kaggle_bin, "kernels", "list", "--competition", COMPETITION, "--sort-by", "voteCount", "--csv"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if k_proc.returncode == 0:
            lines = [ln for ln in k_proc.stdout.strip().splitlines() if ln.strip()]
            info["kernels_top_votes"] = max(0, len(lines) - 1)
    except Exception as exc:
        info["status"] = f"kaggle_kernels_err: {exc}"

    try:
        s_proc = subprocess.run(
            [kaggle_bin, "competitions", "submissions", "-c", COMPETITION, "--csv"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if s_proc.returncode == 0:
            today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
            reader = csv.DictReader(s_proc.stdout.splitlines())
            info["submissions_today_utc"] = sum(
                1 for row in reader if str(row.get("date", "")).startswith(today)
            )
    except Exception:
        pass
    return info


def seconds_until_next_utc_reset() -> int:
    now = dt.datetime.now(dt.timezone.utc)
    nxt = (now + dt.timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
    return max(60, int((nxt - now).total_seconds()))


def append_submission_log(cycle: int, tests_ok: bool, serve_sha: str, kernels_cnt: int) -> None:
    SUBMISSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = SUBMISSION_LOG_PATH.exists()
    with SUBMISSION_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp_utc", "cycle", "tests_ok", "serve_sha256", "public_kernels"])
        writer.writerow([
            dt.datetime.now(dt.timezone.utc).isoformat(),
            cycle,
            tests_ok,
            serve_sha[:12],
            kernels_cnt,
        ])


def write_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_cycle(cycle: int, run_evolution: bool = False) -> Dict[str, Any]:
    evo_result: Optional[Dict[str, Any]] = None
    if run_evolution:
        log_msg(f"Cycle {cycle}: triggering Gemini Evolution-in-Code step...")
        evo_result = evolve_verifiers_once()
        log_msg(
            f"Cycle {cycle} evolution | accepted={evo_result.get('accepted')} | "
            f"fn={evo_result.get('function_name', '-')} | reason={evo_result.get('reason', 'ok')}"
        )

    tests_ok, test_summary = run_unit_tests()
    serve_sha = sha256_file(ROOT_DIR / "harness" / "serve.py")
    verifier_sha = sha256_file(ROOT_DIR / "harness" / "verifiers.py")
    kaggle_info = poll_kaggle_competition()

    state = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cycle": cycle,
        "competition": COMPETITION,
        "daily_limit": DAILY_LIMIT,
        "tests_ok": tests_ok,
        "test_summary": test_summary,
        "harness_serve_sha256": serve_sha,
        "harness_verifiers_sha256": verifier_sha,
        "last_evolution": evo_result,
        "kaggle": kaggle_info,
        "next_utc_reset_sec": seconds_until_next_utc_reset(),
    }
    write_state(state)
    append_submission_log(cycle, tests_ok, serve_sha, kaggle_info.get("kernels_top_votes", 0))
    log_msg(
        f"Cycle {cycle} complete | tests_ok={tests_ok} | serve_sha={serve_sha[:10]} | "
        f"kernels={kaggle_info.get('kernels_top_votes', 0)} | next_utc_reset={state['next_utc_reset_sec']}s"
    )
    gc.collect()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="PLE Autonomous Orchestrator")
    parser.add_argument("--daemon", action="store_true", help="Run continuously 24/7")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--evolve-once", action="store_true", help="Run a single cycle with Gemini evolution")
    parser.add_argument("--evolve-every", type=int, default=10, help="Run Gemini evolution every N cycles in daemon mode")
    parser.add_argument("--cooldown", type=int, default=120, help="Seconds between cycles")
    args = parser.parse_args()

    load_dotenv(ENV_PATH)
    log_msg(f"Starting PLE Autonomous Orchestrator (daemon={args.daemon}, cooldown={args.cooldown}s)")

    cycle = 0
    while True:
        cycle += 1
        do_evolve = args.evolve_once or (args.daemon and args.evolve_every > 0 and cycle % args.evolve_every == 1)
        try:
            run_cycle(cycle, run_evolution=do_evolve)
        except Exception as exc:
            log_msg(f"ERROR in cycle {cycle}: {exc}")
            gc.collect()
        if args.once or args.evolve_once or not args.daemon:
            break
        time.sleep(max(10, args.cooldown))


if __name__ == "__main__":
    main()

"""FunSearch-style Evolution-in-Code Mutator using GEMINI_API_KEY from .env."""

from __future__ import annotations

import ast
import datetime as dt
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from src.utils.checkpoint_sync import CheckpointManager


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
VERIFIERS_PATH = ROOT_DIR / "harness" / "verifiers.py"
TESTS_PATH = ROOT_DIR / "tests" / "test_verifiers.py"
HISTORY_PATH = ROOT_DIR / "artifacts" / "evolution_history.jsonl"
CHECKPOINT_DIR = ROOT_DIR / "artifacts" / "evolution_checkpoints"

DEFAULT_MODEL = "gemini-3.8-flash"


def _call_gemini(prompt: str, api_key: str, model: str = DEFAULT_MODEL, timeout: int = 120) -> str:
    """Call Gemini generateContent REST API using pure stdlib urllib."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Empty Gemini response: {data}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _run_tests() -> tuple[bool, int, str]:
    """Run unittest suite and return (ok, num_tests, output_tail)."""
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    m = re.search(r"Ran\s+(\d+)\s+test", combined)
    num_tests = int(m.group(1)) if m else 0
    tail = "\n".join(combined.splitlines()[-6:])
    return proc.returncode == 0, num_tests, tail


def _append_history(entry: Dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _extract_json(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text)


def evolve_verifiers_once(model: Optional[str] = None) -> Dict[str, Any]:
    """Ask Gemini to propose one evolutionary verifier + unit test, then gate on AST & tests."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"accepted": False, "reason": "GEMINI_API_KEY_missing"}

    target_model = model or os.environ.get("GEMINI_EVOLVE_MODEL", DEFAULT_MODEL)
    orig_verifiers = VERIFIERS_PATH.read_text(encoding="utf-8")
    orig_tests = TESTS_PATH.read_text(encoding="utf-8")
    ok_before, count_before, _ = _run_tests()
    if not ok_before:
        return {"accepted": False, "reason": "baseline_tests_failing"}

    prompt = f"""You are a Kaggle Grandmaster & Licensed Physical Engineer evolving a deterministic code-gate verifier library for Physical-Engineering's Last Exam (PLE: Structural, Civil, Electrical, Mechanical).

Current `harness/verifiers.py`:
```python
{orig_verifiers}
```

Current `tests/test_verifiers.py`:
```python
{orig_tests}
```

Task:
Propose ONE new, high-value, pure-Python (stdlib/math only) deterministic engineering code-gate verifier function to append to `harness/verifiers.py`, AND one corresponding `unittest` method to add inside `TestEngineeringVerifiers` in `tests/test_verifiers.py`.
Focus on high-frequency PLE hidden-test gaps (Mechanical ASHRAE HVAC/cooling/duct velocity/pressure drop, Electrical NEC ampacity/transformer/solar interconnection, Structural AISC column buckling/shear-lag, or Civil runoff/grading/retaining wall stability).
Do NOT duplicate existing function names. Handle edge cases (zero/negative/NaN inputs) cleanly.

Return strictly valid JSON with keys:
- "function_name": string (snake_case, unique)
- "rationale": string (1-2 sentences)
- "verifier_code": string (top-level Python function definition with type hints and docstring)
- "test_method_code": string (indented 4 spaces, `def test_...(self) -> None:` method calling `from harness.verifiers import <function_name>` or using module-level import)
"""

    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        raw = _call_gemini(prompt, api_key=api_key, model=target_model)
        proposal = _extract_json(raw)
        fn_name = str(proposal["function_name"]).strip()
        verifier_code = str(proposal["verifier_code"]).rstrip()
        test_code = str(proposal["test_method_code"]).rstrip()
        rationale = str(proposal.get("rationale", "")).strip()

        if f"def {fn_name}(" in orig_verifiers:
            raise ValueError(f"Duplicate function_name: {fn_name}")

        new_verifiers = orig_verifiers.rstrip() + "\n\n\n" + verifier_code + "\n"
        # Import the new function at top of test_verifiers.py and insert test method before if __name__ == "__main__":
        import_anchor = "from harness.verifiers import ("
        if import_anchor not in orig_tests:
            raise ValueError("Missing import block in tests/test_verifiers.py")
        patched_tests = orig_tests.replace(
            import_anchor,
            f"{import_anchor}\n    {fn_name},",
            1,
        )
        main_anchor = 'if __name__ == "__main__":'
        if main_anchor not in patched_tests:
            raise ValueError("Missing __main__ block in tests/test_verifiers.py")
        patched_tests = patched_tests.replace(
            main_anchor,
            f"{test_code}\n\n\n{main_anchor}",
            1,
        )

        # AST syntax gate before writing/testing
        ast.parse(new_verifiers)
        ast.parse(patched_tests)

        VERIFIERS_PATH.write_text(new_verifiers, encoding="utf-8")
        TESTS_PATH.write_text(patched_tests, encoding="utf-8")

        ok_after, count_after, tail = _run_tests()
        if not ok_after or count_after <= count_before:
            VERIFIERS_PATH.write_text(orig_verifiers, encoding="utf-8")
            TESTS_PATH.write_text(orig_tests, encoding="utf-8")
            result = {
                "timestamp_utc": ts,
                "accepted": False,
                "model": target_model,
                "function_name": fn_name,
                "reason": f"test_gate_failed (ok={ok_after}, {count_before}->{count_after})",
                "tail": tail,
            }
            _append_history(result)
            gc.collect()
            return result

        ckpt = CheckpointManager(local_dir=CHECKPOINT_DIR)
        ckpt.save_if_improved(
            score=float(count_after),
            payload={"function_name": fn_name, "rationale": rationale, "model": target_model},
            tag="evolve_verifiers",
        )
        ckpt.shutdown()
        result = {
            "timestamp_utc": ts,
            "accepted": True,
            "model": target_model,
            "function_name": fn_name,
            "rationale": rationale,
            "tests_before": count_before,
            "tests_after": count_after,
        }
        _append_history(result)
        gc.collect()
        return result
    except Exception as exc:
        VERIFIERS_PATH.write_text(orig_verifiers, encoding="utf-8")
        TESTS_PATH.write_text(orig_tests, encoding="utf-8")
        result = {
            "timestamp_utc": ts,
            "accepted": False,
            "model": target_model,
            "reason": f"exception: {exc}",
        }
        _append_history(result)
        gc.collect()
        return result

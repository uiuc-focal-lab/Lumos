#!/usr/bin/env python3
"""
Compare lines of code (excluding comments) between Python scripts in one directory
and DSL (.lumos) programs in another. Goes over both dirs and reports per-file and totals.

Usage:
  python nl2lumos/compare_loc.py [py_dir] [dsl_dir]
  Defaults: py_dir=nl2lumos/py_specs_claude, dsl_dir=nl2lumos/spec_progs_claude
"""

import argparse
from pathlib import Path


def count_code_lines(path: Path, is_python: bool) -> int:
    """Count non-empty lines excluding comment-only lines. Inline comments stripped."""
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8")
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if is_python and "#" in stripped:
            stripped = stripped.split("#", 1)[0].strip()
        if not is_python and "#" in stripped:
            stripped = stripped.split("#", 1)[0].strip()
        if stripped:
            count += 1
    return count


def main():
    project_root = Path(__file__).resolve().parents[1]
    nl2lumos = project_root / "nl2lumos"
    default_py = nl2lumos / "py_specs_claude"
    default_dsl = nl2lumos / "spec_progs_claude"

    parser = argparse.ArgumentParser(description="Compare LOC (excl. comments) between Python and DSL dirs.")
    parser.add_argument("py_dir", nargs="?", type=Path, default=default_py, help="Directory of .py files")
    parser.add_argument("dsl_dir", nargs="?", type=Path, default=default_dsl, help="Directory of .lumos files")
    args = parser.parse_args()

    py_dir = Path(args.py_dir)
    dsl_dir = Path(args.dsl_dir)

    stems_py = {p.stem for p in py_dir.glob("*.py")}
    stems_dsl = {p.stem for p in dsl_dir.glob("*.lumos")}
    stems = sorted(stems_py | stems_dsl)

    if not stems:
        print("No .py or .lumos files found in the given directories.")
        return

    print("--- Lines of code (excluding comments) ---")
    print(f"{'Spec':<30} {'Python':>8} {'DSL (.lumos)':>12}")
    print("-" * 52)
    total_py = 0
    total_dsl = 0
    for stem in stems:
        py_path = py_dir / f"{stem}.py"
        dsl_path = dsl_dir / f"{stem}.lumos"
        py_loc = count_code_lines(py_path, is_python=True)
        dsl_loc = count_code_lines(dsl_path, is_python=False)
        total_py += py_loc
        total_dsl += dsl_loc
        print(f"{stem:<30} {py_loc:>8} {dsl_loc:>12}")
    print("-" * 52)
    print(f"{'Total':<30} {total_py:>8} {total_dsl:>12}")


if __name__ == "__main__":
    main()

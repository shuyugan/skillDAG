#!/usr/bin/env python3
"""Batch generate skill DAGs for all tasks in a harbor job directory."""

import subprocess
import sys
from pathlib import Path
from collections import defaultdict

TRAJ_DIR = Path("/space3/shuyu/project/skillDAG/harbor/skills-jobs/swebench-claude-hierarchical")
OUTPUT_BASE = Path("/space3/shuyu/project/skillDAG/workspace_state/skill_lib")
PIPELINE_SCRIPT = Path("/space3/shuyu/project/skillDAG/workspace_state/agents/skill_constructor.py")

def main():
    # Group attempt dirs by task id
    task_dirs: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(TRAJ_DIR.iterdir()):
        if not d.is_dir():
            continue
        # Parse task_id: everything before the last __hash segment
        parts = d.name.rsplit("__", 1)
        if len(parts) == 2 and len(parts[1]) == 7:
            # e.g., django__django-11119__6vgXR7E -> django__django-11119
            task_id = d.name[:-(len(parts[1]) + 2)]
        else:
            continue
        task_dirs[task_id].append(d)

    print(f"Found {len(task_dirs)} tasks:")
    for tid, dirs in sorted(task_dirs.items()):
        print(f"  {tid}: {len(dirs)} trajectories")

    # Skip tasks that already have a dag.json
    to_run = {}
    for tid, dirs in sorted(task_dirs.items()):
        out = OUTPUT_BASE / tid
        if (out / "dag.json").exists():
            print(f"  [SKIP] {tid} — dag.json already exists")
        else:
            to_run[tid] = dirs

    print(f"\nWill generate {len(to_run)} new skill DAGs\n")

    success = []
    failed = []
    for i, (tid, dirs) in enumerate(sorted(to_run.items()), 1):
        output_path = OUTPUT_BASE / tid
        print(f"[{i}/{len(to_run)}] {tid} ({len(dirs)} trajectories) -> {output_path}")

        cmd = [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--from-scratch",
            "--model", "gpt-5.2",
            "-o", str(output_path),
        ] + [str(d) for d in dirs]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(f"  OK")
                success.append(tid)
            else:
                print(f"  FAILED (rc={result.returncode})")
                print(f"  stderr: {result.stderr[-500:]}")
                failed.append(tid)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (>600s)")
            failed.append(tid)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(tid)

    print(f"\n=== Summary ===")
    print(f"Success: {len(success)}")
    print(f"Failed:  {len(failed)}")
    if failed:
        print(f"Failed tasks: {failed}")


if __name__ == "__main__":
    main()
